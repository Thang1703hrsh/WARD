# WARD: Warp-Aligned Representation Distillation for Large Language Model Compression

Official PyTorch/JAX implementation of our paper:

> **WARD: Warp-Aligned Representation Distillation for Large Language Model Compression**

WARD is a plug-and-play alignment regularizer that augments existing knowledge distillation methods with Soft-DTW sequence warping on intermediate hidden states. It supports both **same-tokenizer** and **cross-tokenizer** distillation regimes.

---

## Repository Structure

```
WARD/
├── training/                         # Training entry points
│   ├── finetune.py                   # Same-tokenizer training (DistiLLM, FDD, WARD-DTW)
│   ├── finetune_dskd_dtw.py          # Cross-tokenizer DSKD + WARD-DTW
│   ├── finetune_dskdv2_dtw.py        # Cross-tokenizer DSKDv2 + WARD-DTW
│   ├── finetune_alm_dtw.py           # Cross-tokenizer ALM + WARD-DTW (JAX/Flax)
│   └── arguments.py                  # CLI argument definitions
│
├── methods/                          # Core implementation of each KD method
│   ├── distillm/                     # Same-tokenizer: losses, projector, sampler, buffer
│   │   ├── losses.py                 # Soft-DTW, FDD, forward/reverse KL, DistiLLM losses
│   │   ├── projector.py              # Linear projector (maps student → teacher hidden dim)
│   │   ├── sampler.py                # On-policy student generation
│   │   └── buffer.py                 # Replay buffer for adaptive sampling
│   ├── distillm2/                    # Same-tokenizer: DistiLLM-2 (SGO-based) losses
│   ├── dskd/                         # Cross-tokenizer: DSKD baseline
│   │   ├── criterions/               # min_edit_dis_kld (Logits Alignment via Min-Edit)
│   │   ├── data_utils/               # Cross-tokenizer dataset loaders
│   │   ├── distillation.py           # DSKD training loop
│   │   └── distiller.py              # Model wrapper (student + teacher)
│   ├── dskdv2/                       # Cross-tokenizer: DSKDv2 baseline
│   │   ├── criterions/               # dual_space_kd_v2 (Dual-Space KD v2)
│   │   ├── data_utils/
│   │   ├── distillation.py
│   │   └── distiller.py
│   └── alm/                          # Cross-tokenizer: ALM baseline (JAX/Flax)
│       ├── cross_tokenizer_distill.py  # ALM training entry (JAX)
│       ├── tokenkit/                 # ALM core library (tokenizer alignment, losses, models)
│       └── configs/                  # YAML configs for ALM experiments
│
├── data_utils/                       # Dataset loading for same-tokenizer training
│   ├── lm_datasets.py                # LMTrainDataset (Dolly, instruction-following)
│   ├── prompt_datasets.py            # PromptDataset for evaluation
│   └── distributed_indexed.py        # Distributed indexed dataset utilities
│
├── scripts/                          # Shell scripts for all experiments
│   ├── same_tokenizer/               # GPT-2, LLaMA-2, OpenLLaMA-2 experiments
│   │   ├── gpt2/                     # fdd/, distillm/, distillm2/, dtw/, sft/, eval/
│   │   ├── llama2/                   # fdd/, distillm/, distillm2/, dtw/, sft/, eval/
│   │   └── openllama2/               # fdd/, distillm/, distillm2/, dtw/, sft/, eval/
│   └── cross_tokenizer/              # Cross-tokenizer experiments
│       ├── ALM/                      # ALM + WARD-DTW (5 teacher→student pairs)
│       ├── DSKD/                     # DSKD + WARD-DTW (5 pairs)
│       └── DSKD2/                    # DSKDv2 + WARD-DTW (5 pairs)
│
├── configs/
│   └── deepspeed/                    # DeepSpeed ZeRO configs (stage 2, fp16/bf16)
│
├── tools/                            # Data preprocessing utilities
│   ├── process_data_dolly.py         # Process Dolly instruction dataset
│   ├── process_data_pretrain.py      # Process OpenWebText pre-training data
│   └── push_to_hub.py                # Upload checkpoints to HuggingFace Hub
│
├── utils.py                          # Shared utilities (model loading, tokenizer, logging)
├── rouge_metric.py                   # ROUGE evaluation metric
├── wandb_logger.py                   # Weights & Biases logging helper
└── README.md
```

**Key design principles:**
- All training entry points live in `training/` and are invoked from the repo root (`WARD/`) as `BASE_PATH`
- `methods/distillm/losses.py` contains the **Soft-DTW implementation** shared across all same-tokenizer experiments
- `methods/alm/tokenkit/training/losses.py` contains the **JAX Soft-DTW implementation** for ALM cross-tokenizer experiments
- DSKD/DSKDv2 modules are loaded at runtime via `sys.path` injection from their respective `methods/dskd*/` directories

---

## Environment Setup

```bash
bash install.sh
```

---

## Data

### Resources
- Instruction-following training and evaluation data: follow the [DSKD repository](https://github.com/songmzhang/DSKD) to download and prepare **Databricks-Dolly-15k** (11,435 train / 1,000 val / 500 test samples).
- Out-of-distribution evaluation sets (Self-Instruct, Vicuna, S-NI, UnNI) are also provided in the DSKD repository.
- Pre-training plain-text corpus: [OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext). For convenience, a smaller subset is available at [openwebtext-100k](https://huggingface.co/datasets/Elriggs/openwebtext-100k).
- Mathematical reasoning: [MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA) (50k training samples sampled from GSM8K + MATH training splits).
- Code generation: Python subset of [Magicoder](https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K) (10k training samples).

### Data Processing

```bash
# Get plain-text corpus (subset)
python3 tools/get_openwebtext.py

# Process Dolly instruction data (change gpt2 → llama2 / openllama2 as needed)
bash scripts/gpt2/tools/process_data_dolly.sh ${BASE_PATH} ${MASTER_PORT} ${GPU_NUM}

# Process OpenWebText pre-training data
bash scripts/gpt2/tools/process_data_pretrain.sh ${BASE_PATH} ${MASTER_PORT} ${GPU_NUM}
```

---

## Model Pairs

WARD is evaluated on **eight teacher→student pairs** spanning two tokenizer regimes:

### Same-Tokenizer Regime
| Teacher | Student | Vocabulary |
|---|---|---|
| GPT-2 1.5B | GPT-2 120M | 50,257 tokens |
| LLaMA2-13B | LLaMA2-7B | 32,000 tokens |
| OpenLLaMA2-7B | OpenLLaMA2-3B | 32,000 tokens |

### Cross-Tokenizer Regime (Small-Scale)
| Teacher | Student |
|---|---|
| Qwen1.5-1.8B (vocab: 151,936) | GPT-2 120M (vocab: 50,257) |
| Qwen1.5-1.8B (vocab: 151,936) | GPT-2 340M (vocab: 50,257) |

### Cross-Tokenizer Regime (Large-Scale)
| Teacher | Student |
|---|---|
| Mistral 7B (vocab: 32,000) | TinyLLaMA 1.1B (vocab: 32,000†) |
| Qwen2.5-7B (vocab: 152,064) | GPT-2 1.5B (vocab: 50,257) |
| Qwen2.5-7B-Instruct (vocab: 152,064) | OPT 2.7B (vocab: 50,272) |

†Mistral and TinyLLaMA share vocabulary size but use different tokenization functions in practice.

---

## Training

### Command Arguments
- `BASE_PATH`: Path to top-level `WARD` directory. Pass `.` if running from this directory.
- `MASTER_PORT`: Port number (e.g. `2012`). Use different ports for concurrent runs.
- `GPU_NUM`: Number of GPUs.

---

### Same-Tokenizer Baselines

#### Teacher SFT
```bash
bash scripts/gpt2/sft/sft_xlarge.sh ${BASE_PATH} ${MASTER_PORT} ${GPU_NUM}
bash scripts/llama2/sft/sft_13B.sh  ${BASE_PATH} ${MASTER_PORT} ${GPU_NUM}
```

#### Student Warm-up
```bash
bash scripts/gpt2/init/init_base.sh     ${BASE_PATH} ${MASTER_PORT} ${GPU_NUM}
bash scripts/llama2/init/init_7B.sh     ${BASE_PATH} ${MASTER_PORT} ${GPU_NUM}
bash scripts/openllama2/init/init_3B.sh ${BASE_PATH} ${MASTER_PORT} ${GPU_NUM}
```

#### FDD
```bash
bash scripts/gpt2/fdd/fdd_base.sh                              ${BASE_PATH} ${MASTER_PORT} ${GPU_NUM}
bash scripts/llama2/fdd/dtw_7B_13B_teacher_lora.sh             ${BASE_PATH} ${MASTER_PORT} ${GPU_NUM}
bash scripts/openllama2/fdd/dtw_3B_7B_teacher_lora.sh          ${BASE_PATH} ${MASTER_PORT} ${GPU_NUM}
```

#### DistiLLM
```bash
bash scripts/gpt2/distillm/distill_base.sh                     ${BASE_PATH} ${MASTER_PORT} ${GPU_NUM}
bash scripts/llama2/distillm/dtw_7B_13B_teacher_lora.sh        ${BASE_PATH} ${MASTER_PORT} ${GPU_NUM}
bash scripts/openllama2/distillm/dtw_3B_7B_teacher_lora.sh     ${BASE_PATH} ${MASTER_PORT} ${GPU_NUM}
```

#### DistiLLM-2
```bash
# Generate TGOs/SGOs first
bash scripts/gpt2/distillm2/generate_data.sh ${BASE_PATH}

bash scripts/gpt2/distillm2/dtw_0.1B_1.5B.sh                  ${BASE_PATH} ${MASTER_PORT} ${GPU_NUM}
bash scripts/llama2/distillm2/dtw_7B_13B_lora.sh               ${BASE_PATH} ${MASTER_PORT} ${GPU_NUM}
bash scripts/openllama2/distillm2/dtw_3B_7B_lora.sh            ${BASE_PATH} ${MASTER_PORT} ${GPU_NUM}
```

---

### Same-Tokenizer + WARD

WARD adds a Soft-DTW hidden-state regularizer on top of each base method.
The velocity field projector can be trained once and reused across FDD + WARD and DistiLLM + WARD.

#### Train DTW Projector (velocity field)
```bash
bash scripts/gpt2/distillm2/train_velocity_field_distillm2.sh ${BASE_PATH} ${MASTER_PORT} ${GPU_NUM}
```

#### FDD + WARD
```bash
bash scripts/gpt2/fdd/dtw_0.1B_1.5B.sh                        ${BASE_PATH} ${MASTER_PORT} ${GPU_NUM}
bash scripts/llama2/fdd/dtw_7B_13B_teacher_lora.sh             ${BASE_PATH} ${MASTER_PORT} ${GPU_NUM}
bash scripts/openllama2/fdd/dtw_3B_7B_teacher_lora.sh          ${BASE_PATH} ${MASTER_PORT} ${GPU_NUM}
```

#### DistiLLM + WARD
```bash
bash scripts/gpt2/distillm/dtw_0.1B_1.5B.sh                   ${BASE_PATH} ${MASTER_PORT} ${GPU_NUM}
bash scripts/llama2/distillm/dtw_7B_13B_teacher_lora.sh        ${BASE_PATH} ${MASTER_PORT} ${GPU_NUM}
bash scripts/openllama2/distillm/dtw_3B_7B_teacher_lora.sh     ${BASE_PATH} ${MASTER_PORT} ${GPU_NUM}
```

#### DistiLLM-2 + WARD
```bash
# Generate TGOs/SGOs (can reuse from baseline)
bash scripts/gpt2/distillm2/generate_data.sh ${BASE_PATH}

# Train projector
bash scripts/gpt2/distillm2/train_velocity_field_distillm2.sh ${BASE_PATH} ${MASTER_PORT} ${GPU_NUM}

bash scripts/gpt2/distillm2/dtw_0.1B_1.5B.sh                  ${BASE_PATH} ${MASTER_PORT} ${GPU_NUM}
bash scripts/llama2/distillm2/dtw_7B_13B_lora.sh               ${BASE_PATH} ${MASTER_PORT} ${GPU_NUM}
bash scripts/openllama2/distillm2/dtw_3B_7B_lora.sh            ${BASE_PATH} ${MASTER_PORT} ${GPU_NUM}
```

---

### Cross-Tokenizer Baselines & + WARD

Cross-tokenizer distillation requires a **vocab mapping file** (DSKD only).
Build with `python DSKD/tools/build_token_mapping.py` (see DSKD README).

#### DSKD + WARD
```bash
# Small-scale
bash scripts/cross_tokenizer/DSKD/dskd_dtw_qwen1.5_1.8B_gpt2_120M.sh  ${BASE_PATH} ${MASTER_PORT} ${GPU_NUM}
bash scripts/cross_tokenizer/DSKD/dskd_dtw_qwen1.5_1.8B_gpt2_340M.sh  ${BASE_PATH} ${MASTER_PORT} ${GPU_NUM}
# Large-scale
bash scripts/cross_tokenizer/DSKD/dskd_dtw_qwen2.5_7B_gpt2_1.5B.sh          ${BASE_PATH} ${MASTER_PORT} ${GPU_NUM}
bash scripts/cross_tokenizer/DSKD/dskd_dtw_qwen2.5_7B_instruct_opt_2.7B.sh   ${BASE_PATH} ${MASTER_PORT} ${GPU_NUM}
bash scripts/cross_tokenizer/DSKD/dskd_dtw_mistral_7B_tinyllama_1.1B.sh      ${BASE_PATH} ${MASTER_PORT} ${GPU_NUM}
```

Entry point: `finetune_dskd_dtw.py` (torchrun).

#### DSKDv2 + WARD
```bash
bash scripts/cross_tokenizer/DSKD2/dskdv2_dtw_qwen1.5_1.8B_gpt2_120M.sh ${BASE_PATH} ${MASTER_PORT} ${GPU_NUM}
bash scripts/cross_tokenizer/DSKD2/dskdv2_dtw_qwen1.5_1.8B_gpt2_340M.sh ${BASE_PATH} ${MASTER_PORT} ${GPU_NUM}
bash scripts/cross_tokenizer/DSKD2/dskdv2_dtw_qwen2.5_7B_gpt2_1.5B.sh         ${BASE_PATH} ${MASTER_PORT} ${GPU_NUM}
bash scripts/cross_tokenizer/DSKD2/dskdv2_dtw_qwen2.5_7B_instruct_opt_2.7B.sh  ${BASE_PATH} ${MASTER_PORT} ${GPU_NUM}
bash scripts/cross_tokenizer/DSKD2/dskdv2_dtw_mistral_7B_tinyllama_1.1B.sh     ${BASE_PATH} ${MASTER_PORT} ${GPU_NUM}
```

Entry point: `finetune_dskdv2_dtw.py` (torchrun). No vocab mapping required.

#### ALM + WARD
ALM uses JAX/Flax. Flax checkpoints are required (convert from HuggingFace PyTorch weights via `ALM/scripts/export_checkpoint.py` if needed).

```bash
bash scripts/cross_tokenizer/ALM/alm_dtw_qwen1.5_1.8B_gpt2_120M.sh  ${BASE_PATH} ${N_DATA_PARALLEL} ${N_MODEL_PARALLEL}
bash scripts/cross_tokenizer/ALM/alm_dtw_qwen1.5_1.8B_gpt2_340M.sh  ${BASE_PATH} ${N_DATA_PARALLEL} ${N_MODEL_PARALLEL}
bash scripts/cross_tokenizer/ALM/alm_dtw_qwen2.5_7B_gpt2_1.5B.sh          ${BASE_PATH} ${N_DATA_PARALLEL} ${N_MODEL_PARALLEL}
bash scripts/cross_tokenizer/ALM/alm_dtw_qwen2.5_7B_instruct_opt_2.7B.sh   ${BASE_PATH} ${N_DATA_PARALLEL} ${N_MODEL_PARALLEL}
bash scripts/cross_tokenizer/ALM/alm_dtw_mistral_7B_tinyllama_1.1B.sh      ${BASE_PATH} ${N_DATA_PARALLEL} ${N_MODEL_PARALLEL}
```

Entry point: `finetune_alm_dtw.py` (python3, JAX/Flax). Soft-DTW is re-implemented in JAX for ALM compatibility.

---

## WARD Hyperparameters

The following default values are used in all experiments unless otherwise noted:

| Hyperparameter | Value | Description |
|---|---|---|
| Soft-DTW smoothing γ | 1.0 | Softmin temperature |
| Band penalty λ\_band | 1.0 | Sakoe-Chiba band strength |
| Hidden-state weight λ\_H | 1.0 | Weight on hidden-state alignment loss |
| Dynamics weight λ\_Δ | 0 | Layer-wise delta alignment (disabled by default) |
| λ\_warp | 0.5 | Overall WARD regularizer weight |

**Same-tokenizer regime:** band half-width b₀ = 3 tokens; embedding weight λ\_E = 0 (exact lexical match, no embedding alignment needed).

**Cross-tokenizer regime:** band half-width b₀ = 5 word-level units; correspondence temperature τ\_A = 0.1; banding blend α = 0.7; entropy-adaptive scaling β\_ent = 2.0; embedding weight λ\_E = 0.5.

---

## Evaluation

### Instruction-Following (ROUGE-L)

Responses are generated via sampling (temperature = 1.0, top-p = 1.0) with 5 random seeds {10, 20, 30, 40, 50}.
ROUGE-L is reported against human-annotated references on five benchmarks: Dolly, Vicuna, Self-Instruct, S-NI, UnNI.

```bash
bash scripts/gpt2/eval/run_eval.sh        ${GPU_IDX} ${CKPT_PATH}
bash scripts/llama2/eval/run_eval.sh      ${GPU_IDX} ${CKPT_PATH}
bash scripts/openllama2/eval/run_eval.sh  ${GPU_IDX} ${CKPT_PATH}
```

Checkpoint paths follow the pattern:
- Baselines: `results/<model-type>/train/<method>/<student-teacher-size>/`
- WARD: `results/<model-type>/train/dtw/<method>/<student-teacher-size>/`
- Cross-tokenizer: `results/cross_tokenizer/<method>/<teacher-student>/`

### Mathematical Reasoning (Accuracy)

Evaluated on MATH-500 and GSM8K using the [Qwen2.5-Math toolkit](https://github.com/QwenLM/Qwen2.5-Math).
Teacher: Qwen2.5-Math-7B. Students: Qwen2.5-1.5B (same tokenizer) and LLaMA-3.2-1B (cross-tokenizer).

### Code Generation (pass@1)

Evaluated on HumanEval and MBPP using the [Qwen2.5-Coder toolkit](https://github.com/QwenLM/Qwen2.5-coder).
Teacher: Qwen2.5-Coder-7B. Students: Qwen2.5-1.5B and LLaMA-3.2-1B.

---

## Main Results Summary

### Same-Tokenizer (ROUGE-L, avg. across 5 benchmarks)

| Method | GPT-2 1.5B→120M | LLaMA2-13B→7B | OpenLLaMA2-7B→3B |
|---|---|---|---|
| FDD | 18.47 | 29.19 | 27.43 |
| FDD + **WARD** | **19.77** | **30.17** | **28.74** |
| DistiLLM | 20.49 | 29.81 | 27.34 |
| DistiLLM + **WARD** | **21.64** | **31.19** | **28.75** |
| DistiLLM-2 | 19.30 | 30.15 | 28.06 |
| DistiLLM-2 + **WARD** | **20.72** | **31.28** | **29.54** |

### Cross-Tokenizer Small-Scale (ROUGE-L, avg.)

| Method | Qwen1.5-1.8B→GPT-2 120M | Qwen1.5-1.8B→GPT-2 340M |
|---|---|---|
| DSKD | 17.32 | 17.89 |
| DSKD + **WARD** | **18.34** | **18.83** |
| DSKDv2 | 18.72 | 19.17 |
| DSKDv2 + **WARD** | **19.27** | **19.84** |
| ALM | 16.84 | 17.39 |
| ALM + **WARD** | **17.85** | **18.14** |

### Cross-Tokenizer Large-Scale (ROUGE-L, avg.)

| Method | Qwen2.5-7B→GPT-2 1.5B | Qwen2.5-7B-Inst→OPT 2.7B | Mistral 7B→TinyLLaMA 1.1B |
|---|---|---|---|
| DSKD | 22.26 | 23.39 | 25.10 |
| DSKD + **WARD** | **23.23** | **24.33** | **25.85** |
| DSKDv2 | 23.40 | 24.31 | 26.76 |
| DSKDv2 + **WARD** | **24.09** | **24.94** | **27.43** |
| ALM | 20.74 | 21.79 | 23.03 |
| ALM + **WARD** | **21.51** | **22.72** | **24.00** |

### Mathematical Reasoning & Code Generation

| Setting | Method | Math Avg. | Code Avg. |
|---|---|---|---|
| Same-tok: Qwen2.5-Math-7B→Qwen2.5-1.5B | Off-Policy DSKD | 55.22 | 41.66 |
| | + **WARD** | **56.05** | **42.70** |
| Cross-tok: Qwen2.5-Math-7B→LLaMA-3.2-1B | Off-Policy DSKD-ETA | 26.41 | 15.64 |
| | + **WARD** | **27.07** | **16.53** |

---

## Training Configurations

### Cross-Tokenizer (DSKD, DSKDv2, ALM)

| Setting | Qwen1.5-1.8B → GPT-2 120M/340M | Mistral 7B → TinyLLaMA | Qwen2.5-7B(-Inst) → GPT-2 1.5B / OPT 2.7B |
|---|---|---|---|
| Epochs | 20 | 15 | 15 |
| LR | 5×10⁻⁴ | 1×10⁻³ | 1×10⁻³ |
| Projector LR | 1×10⁻³ | 1×10⁻³ | 1×10⁻³ |
| Batch Size | 8 | 8 | 8 |
| LR Scheduler | Cosine | Cosine | Cosine |
| Fine-Tuning | Full | LoRA (rank=256, α=8) | LoRA (rank=256, α=8) |
| λ\_warp | 0.5 | 0.5 | 0.5 |
| τ | 2.0 | 2.0 | 2.0 |

### Same-Tokenizer — FDD / DistiLLM

| Setting | GPT-2 1.5B → 120M | LLaMA2-13B → 7B | OpenLLaMA2-7B → 3B |
|---|---|---|---|
| Epochs | 20 | 20 | 20 |
| LR | 1×10⁻⁴ | 1×10⁻³ | 1×10⁻³ |
| Projector LR | 5×10⁻⁴ | 1×10⁻³ | 1×10⁻³ |
| Batch Size | 16 | 16 | 16 |
| Fine-Tuning | Full | LoRA (rank=256, α=8) | LoRA (rank=256, α=8) |

### Same-Tokenizer — DistiLLM-2

Same as FDD / DistiLLM except **Batch Size = 8** (due to on-policy rollout memory overhead).

### Math / Code (DSKD & DSKD-ETA)

| Setting | Value |
|---|---|
| Epochs | 3 (math), 1 (code) |
| LR | 1×10⁻⁵ |
| Projector LR | 1×10⁻³ |
| Batch Size | 128 (math), 128/32 (code) |
| Fine-Tuning | Full |
| KD Rate | 0.5 |
| KD Temperature | 1.0 |
| λ\_warp | 0.5 |

---

## Infrastructure

All experiments are conducted on **8 × NVIDIA A100 (40 GB) GPUs**.
Same-tokenizer models use PyTorch + HuggingFace Transformers with DeepSpeed.
Cross-tokenizer ALM experiments use JAX/Flax.

---

## Acknowledgements

Our code builds on:
- [MiniLLM / DistiLLM](https://github.com/microsoft/LMOps/tree/main/minillm) (Gu et al., ICML 2024)
- [DistiLLM-2](https://github.com/jongwooko/distillm-2)
<!-- - [FDD](https://github.com/gongliym/fdd) (Gong et al., 2025) -->
- [DSKD](https://github.com/songmzhang/DSKD) (Zhang et al., 2024)
- [DSKDv2](https://github.com/songmzhang/DSKD) (Zhang et al., 2025)
- [ALM / tokenkit](https://github.com/bminixhofer/tokenkit) (Minixhofer et al., 2025)
