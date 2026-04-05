import os
import sys
import types
import torch

# Extend sys.path so DSKD modules are importable
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
DSKD_DIR = os.path.join(REPO_DIR, "DSKD")
if DSKD_DIR not in sys.path:
    sys.path.insert(0, DSKD_DIR)

# Import DSKD components  (sys.path must be set first)
import distillation                          # DSKD/distillation.py
import criterions as _criterions             # DSKD/criterions/__init__.py
from distiller import Distiller              # DSKD/distiller.py
from arguments import (                      # DSKD/arguments.py
    add_model_args, add_runtime_args, add_data_args,
    add_hp_args, add_gen_args, add_peft_args,
)

# Import DTW components from distillm (WARD root is already on sys.path
#    when this file is run via torchrun from WARD/)
from distillm.losses import dtw_distillation_loss
from distillm.projector import Projector

CRITERION_NAME = "min_edit_dis_kld"

# Argument helpers
def _force_criterion_in_argv():
    """Ensure --criterion min_edit_dis_kld is in sys.argv (remove any prior value)."""
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
    """Add Soft-DTW arguments to DSKD's ArgumentParser."""
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
        raise ValueError("--teacher-model-path is required for DSKD+DTW.")
    if args.teacher_to_student_id_mapping is None:
        raise ValueError(
            "--teacher-to-student-id-mapping (JSON) is required for DSKD+DTW.")
    if not os.path.exists(args.teacher_to_student_id_mapping):
        raise ValueError(
            f"Mapping file not found: {args.teacher_to_student_id_mapping}")
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

    Flow per batch
    --------------
    a) Run student forward with output_hidden_states=True → cache student_outputs
    b) Run teacher forward with output_hidden_states=True → cache teacher_outputs
       (inside torch.no_grad())
    c) Temporarily replace model.forward with a closure that returns the cached
       outputs, so the DSKD criterion re-uses them without a second pass.
    d) Call the original DSKD criterion forward → get dskd_loss.
    e) Restore model.forward.
    f) Compute dtw_loss on the last hidden states.
    g) Return  dskd_loss + dtw_weight * dtw_loss / batch_denom
    """
    dtw_projector = _build_dtw_projector(args, device)
    original_forward = criterion.__class__.forward  # unbound method

    def patched_forward(self_crit, distiller, input_data, output_data,
                        logging_output, batch_denom):

        s_model = distiller.student_model
        t_model = distiller.teacher_model
        t_type  = distiller.teacher_model_type

        # (a) student forward
        s_out = s_model(
            input_data["input_ids"],
            attention_mask=input_data["attention_mask"],
            position_ids=input_data.get("position_ids", None),
            output_hidden_states=True,
            use_cache=False,
        )

        # (b) teacher forward — no grad
        with torch.no_grad():
            t_model.eval()
            t_out = t_model(
                input_data[f"teacher_{t_type}_input_ids"],
                attention_mask=input_data[f"teacher_{t_type}_attention_mask"],
                position_ids=input_data.get(
                    f"teacher_{t_type}_position_ids", None),
                output_hidden_states=True,
                use_cache=False,
            )

        # (c) replace model.forward with cached-output stubs
        _s_orig = s_model.forward
        _t_orig = t_model.forward

        # Replace with closures that return cached outputs regardless of call args.
        # Pylance unused-param hints on stub signatures are expected and harmless.
        s_model.forward = lambda *__, **___: s_out   # type: ignore[method-assign]
        t_model.forward = lambda *__, **___: t_out   # type: ignore[method-assign]

        # (d) run original DSKD criterion
        try:
            loss_scaled, logging_output = original_forward(
                self_crit, distiller, input_data, output_data,
                logging_output, batch_denom,
            )
        finally:
            # (e) restore model.forward unconditionally
            s_model.forward = _s_orig
            t_model.forward = _t_orig

        # (f) Soft-DTW on last hidden states
        if (args.dtw_weight > 0
                and s_out.hidden_states is not None
                and t_out.hidden_states is not None):

            attn_mask = input_data["attention_mask"].float()
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
                dtw_loss = torch.tensor(0.0, device=attn_mask.device)

            if torch.isnan(dtw_loss) or torch.isinf(dtw_loss):
                dtw_loss = torch.tensor(0.0, device=attn_mask.device)

            # (g) add DTW loss (same batch_denom normalisation as DSKD)
            loss_scaled = loss_scaled + args.dtw_weight * dtw_loss / batch_denom

            # log alongside DSKD fields
            if "dtw_loss" not in logging_output:
                logging_output["dtw_loss"] = []
            if isinstance(logging_output["dtw_loss"], list):
                logging_output["dtw_loss"].append(dtw_loss.item())

        return loss_scaled, logging_output

    criterion.forward = types.MethodType(patched_forward, criterion)
    return criterion


# Entry point
def main():
    _force_criterion_in_argv()

    # Build parser with DSKD schema + our DTW arguments
    import argparse
    import deepspeed as _ds

    parser = argparse.ArgumentParser(
        description="DSKD + Soft-DTW knowledge distillation")
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
