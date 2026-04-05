import os
import sys
import types
import torch

# Extend sys.path so DSKDv2 modules are importable
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # WARD/
DSKDV2_DIR = os.path.join(REPO_DIR, "methods", "dskdv2")
if DSKDV2_DIR not in sys.path:
    sys.path.insert(0, DSKDV2_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

# Import DSKDv2 components  (sys.path must be set first)
import distillation                          # DSKDv2/distillation.py
import criterions as _criterions             # DSKDv2/criterions/__init__.py
from distiller import Distiller              # DSKDv2/distiller.py
from arguments import (                      # DSKDv2/arguments.py
    add_model_args, add_runtime_args, add_data_args,
    add_hp_args, add_gen_args, add_peft_args,
)

# Import DTW components from distillm (WARD root is already on sys.path
#    when this file is run via torchrun from WARD/)
from methods.distillm.losses import dtw_distillation_loss
from methods.distillm.projector import Projector

CRITERION_NAME = "dual_space_kd_v2"

# Argument helpers
def _force_criterion_in_argv():
    """Ensure --criterion dual_space_kd_v2 is in sys.argv (remove any prior value)."""
    cleaned, skip = [], False
    for arg in sys.argv[1:]:
        if skip:
            skip = False
            continue
        if arg == "--criterion":
            skip = True
            continue
        if arg.startswith("--criterion="):
            continue
        cleaned.append(arg)
    cleaned += ["--criterion", CRITERION_NAME]
    sys.argv = [sys.argv[0]] + cleaned


def _add_dtw_args(parser):
    """Add Soft-DTW arguments to DSKDv2's ArgumentParser."""
    g = parser.add_argument_group("dtw", "Soft-DTW distillation on hidden states")
    g.add_argument("--dtw-weight", type=float, default=1.0,
                   help="Weight multiplied on DTW loss (0 = disabled)")
    g.add_argument("--dtw-window", type=int, default=32,
                   help="Sakoe-Chiba band half-width (None = full matrix)")
    g.add_argument("--dtw-gamma", type=float, default=0.1,
                   help="Softmin temperature γ for Soft-DTW")
    g.add_argument("--dtw-distance", type=str, default="cosine",
                   choices=["cosine", "l2"],
                   help="Pairwise cost metric")
    g.add_argument("--dtw-normalize", action="store_true",
                   help="Normalise DTW loss by 2*seq_len")
    g.add_argument("--dtw-unitization", action="store_true",
                   help="Pool consecutive tokens into BPE units before DTW")
    g.add_argument("--dtw-use-divergence", action="store_true",
                   help="Compute DTW divergence: DTW(s,t)-0.5*(DTW(s,s)+DTW(t,t))")
    g.add_argument("--d-student", type=int, default=None,
                   help="Student hidden dimension (needed when != teacher dim)")
    g.add_argument("--d-teacher", type=int, default=None,
                   help="Teacher hidden dimension (needed when != student dim)")
    g.add_argument("--dtw-projector-path", type=str, default=None,
                   help="Optional path to a pre-trained DTW projector .pth file")
    return parser


def _validate_args(args):
    if args.teacher_model_path is None:
        raise ValueError("--teacher-model-path is required for DSKDv2+DTW.")
    if args.dtw_weight > 0:
        if args.d_student is None or args.d_teacher is None:
            raise ValueError(
                "--d-student and --d-teacher are required when --dtw-weight > 0.")


# DTW projector
def _build_dtw_projector(args, device):
    """Return a Projector(d_student → d_teacher), or None when dims match."""
    if args.d_student == args.d_teacher:
        return None
    proj = Projector(d_student=args.d_student, d_teacher=args.d_teacher).to(device)
    if args.dtw_projector_path and os.path.exists(args.dtw_projector_path):
        proj.load_state_dict(
            torch.load(args.dtw_projector_path, map_location=device))
        print(f"[DTW] Loaded projector from {args.dtw_projector_path}")
    else:
        print("[DTW] Projector randomly initialised (no pretrained weights found).")
    return proj


# Criterion patch
def _patch_criterion_with_dtw(criterion, args, device):
    """
    Wrap criterion.forward() to inject Soft-DTW loss.

    DSKDv2 criterion signature: forward(self, distiller, batch, logging_output)
    The criterion internally runs both models.

    Flow per batch
    --------------
    a) Run student forward with output_hidden_states=True → cache student_outputs
    b) Run teacher forward with output_hidden_states=True → cache teacher_outputs
       (inside torch.no_grad())
    c) Temporarily replace model.forward with a closure that returns cached
       outputs, so DSKDv2 criterion re-uses them without a second forward pass.
    d) Call the original DSKDv2 criterion forward → get dskdv2_loss.
    e) Restore model.forward.
    f) Compute dtw_loss on the last hidden states.
    g) Return  dskdv2_loss + dtw_weight * dtw_loss / loss_denom
    """
    dtw_projector = _build_dtw_projector(args, device)
    original_forward = criterion.__class__.forward  # unbound method

    def patched_forward(self_crit, distiller, batch, logging_output):

        s_model = distiller.student_model
        t_model = distiller.teacher_model

        # Determine which input key to use for this batch
        if "op_input_batch" in batch:
            s_input = batch["op_input_batch"]
            t_input = batch["op_teacher_input_batch"]
        else:
            s_input = batch["input_batch"]
            t_input = batch["teacher_input_batch"]

        # (a) student forward
        s_out = s_model(
            **s_input,
            output_hidden_states=True,
            use_cache=False,
        )

        # (b) teacher forward — no grad
        with torch.no_grad():
            t_model.eval()
            t_out = t_model(
                **t_input,
                output_hidden_states=True,
                use_cache=False,
            )

        # (c) replace model.forward with cached-output stubs
        _s_orig = s_model.forward
        _t_orig = t_model.forward

        s_model.forward = lambda *__, **___: s_out   # type: ignore[method-assign]
        t_model.forward = lambda *__, **___: t_out   # type: ignore[method-assign]

        # (d) run original DSKDv2 criterion
        try:
            loss, logging_output = original_forward(
                self_crit, distiller, batch, logging_output,
            )
        finally:
            # (e) restore model.forward unconditionally
            s_model.forward = _s_orig
            t_model.forward = _t_orig

        # (f) Soft-DTW on last hidden states
        if (args.dtw_weight > 0
                and s_out.hidden_states is not None
                and t_out.hidden_states is not None):

            # Use attention mask from the student input
            if "op_input_batch" in batch:
                attn_mask = batch["op_input_batch"].get("attention_mask", None)
            else:
                attn_mask = batch["input_batch"].get("attention_mask", None)

            if attn_mask is not None:
                attn_mask = attn_mask.float()

            try:
                dtw_loss = dtw_distillation_loss(
                    student_hiddens=s_out.hidden_states,
                    teacher_hiddens=t_out.hidden_states,
                    student_schedule=None,
                    teacher_schedule=None,
                    attention_mask=attn_mask,
                    projector=dtw_projector,
                    window_size=args.dtw_window,
                    gamma=args.dtw_gamma,
                    distance=args.dtw_distance,
                    normalize=args.dtw_normalize,
                    use_divergence=args.dtw_use_divergence,
                )
            except Exception as exc:
                print(f"[DTW] Warning – DTW loss skipped: {exc}")
                ref_device = loss.device if attn_mask is None else attn_mask.device
                dtw_loss = torch.tensor(0.0, device=ref_device)

            if torch.isnan(dtw_loss) or torch.isinf(dtw_loss):
                dtw_loss = torch.tensor(0.0, device=dtw_loss.device)

            # Normalise by the same loss_denom DSKDv2 uses internally
            loss_denom = batch["label_batch"]["loss_denom"]
            loss = loss + args.dtw_weight * dtw_loss / loss_denom

            # log alongside DSKDv2 fields
            if "dtw_loss" not in logging_output:
                logging_output["dtw_loss"] = []
            if isinstance(logging_output["dtw_loss"], list):
                logging_output["dtw_loss"].append(dtw_loss.item())

        return loss, logging_output

    criterion.forward = types.MethodType(patched_forward, criterion)
    return criterion


# Entry point
def main():
    _force_criterion_in_argv()

    # Build parser with DSKDv2 schema + our DTW arguments
    import argparse
    import deepspeed as _ds

    parser = argparse.ArgumentParser(
        description="DSKDv2 + Soft-DTW knowledge distillation")
    parser = add_model_args(parser)
    parser = add_runtime_args(parser)
    parser = add_data_args(parser)
    parser = add_hp_args(parser)
    parser = add_gen_args(parser)
    parser = add_peft_args(parser)
    parser = _ds.add_config_arguments(parser)
    parser = Distiller.add_distiller_args(parser)
    parser = _add_dtw_args(parser)

    args, unknown = parser.parse_known_args()
    assert all("--" not in x for x in unknown), \
        f"Unknown args (fix or remove): {unknown}"

    args.local_rank = int(os.getenv("LOCAL_RANK", "0"))
    args.n_gpu = args.n_gpu * args.n_nodes
    args.criterion = CRITERION_NAME

    _validate_args(args)

    # Patch build_criterion so distillation.finetune() uses the DTW-wrapped version
    _orig_build = _criterions.build_criterion

    def _patched_build(a):
        crit = _orig_build(a)
        dev  = torch.cuda.current_device()
        return _patch_criterion_with_dtw(crit, args, dev)

    _criterions.build_criterion = _patched_build
    distillation.build_criterion = _patched_build   # patched at module level too

    # Supply the pre-parsed args so distillation.main() does not re-parse
    distillation.get_args = lambda: args

    distillation.main()


if __name__ == "__main__":
    main()
