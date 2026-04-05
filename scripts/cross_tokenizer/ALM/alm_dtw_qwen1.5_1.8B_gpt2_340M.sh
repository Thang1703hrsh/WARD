#!/usr/bin/env bash
# ALM + Soft-DTW  |  Teacher: Qwen1.5-1.8B  →  Student: GPT-2 340M
#
# Usage:
#   bash scripts/cross_tokenizer/ALM/alm_dtw_qwen1.5_1.8B_gpt2_340M.sh [BASE_PATH] [N_DATA_PARALLEL] [N_MODEL_PARALLEL]

BASE_PATH=${1:-$(pwd)}
N_DATA_PARALLEL=${2:-1}
N_MODEL_PARALLEL=${3:-1}

NAME="alm_dtw_qwen1.5_1.8B_gpt2_340M"

STUDENT_PATH="${BASE_PATH}/methods/alm/model_hub/gpt2/gpt2-340M-flax"
TEACHER_PATH="${BASE_PATH}/methods/alm/model_hub/qwen/Qwen1.5-1.8B-flax"

STUDENT_TOKENIZER="gpt2-medium:source=GPT2"
TEACHER_TOKENIZER="Qwen/Qwen1.5-1.8B:source=Qwen2"
TARGET_TOKENIZER="${STUDENT_TOKENIZER}"

OUTPUT_DIR="${BASE_PATH}/results/cross_tokenizer/alm_dtw/qwen1.5_1.8B_gpt2_340M"
CONFIG="${BASE_PATH}/methods/alm/configs/cross_tokenizer_distill.yaml"

mkdir -p "${OUTPUT_DIR}"

export PYTHONPATH="${BASE_PATH}:${BASE_PATH}/methods/alm"
export TF_CPP_MIN_LOG_LEVEL=3
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "======================================================================="
echo "  ALM + Soft-DTW  |  Teacher: Qwen1.5-1.8B  →  Student: GPT-2 340M"
echo "  Output: ${OUTPUT_DIR}"
echo "======================================================================="

python3 "${BASE_PATH}/training/finetune_alm_dtw.py" \
    --config="${CONFIG}" \
    --overrides \
    name="${NAME}" \
    output="${OUTPUT_DIR}" \
    losses="[sft,alm_unconstrained,dtw]" \
    loss_weights="[1.0,1.0,1.0]" \
    alm_mode="merge_by_space_prob+append_space" \
    alm_diff_fn="binary_ce" \
    binarization_temp=100.0 \
    tokenizer_pair_bias_threshold=0.1 \
    dtw_window=32 \
    dtw_gamma=0.1 \
    dtw_normalize=true \
    dtw_use_divergence=false \
    student.pretrained_model_name_or_path="${STUDENT_PATH}" \
    student.tokenizer_name="${STUDENT_TOKENIZER}" \
    teacher.pretrained_model_name_or_path="${TEACHER_PATH}" \
    teacher.tokenizer_name="${TEACHER_TOKENIZER}" \
    target_tokenizer_name="${TARGET_TOKENIZER}" \
    steps=5000 \
    warmup_steps=500 \
    max_teacher_length=512 \
    max_student_length=512 \
    train_model_mode="lora" \
    model_lora_rank=64 \
    model_lora_alpha=64 \
    train_embeddings=true \
    hypernet.architecture="identity" \
    n_data_parallel="${N_DATA_PARALLEL}" \
    n_model_parallel="${N_MODEL_PARALLEL}" \
    data.batch_size=16 \
    data.num_workers=4 \
    data.kind="hf" \
    data.streaming=false \
    data.shuffle_buffer_size=10000 \
    "data.dataset_configs=[{lang_code: en, kwargs: {path: allenai/tulu-3-sft-mixture, split: train}}]" \
    optimizer.type="adamw" \
    optimizer.learning_rate=5.e-5 \
    optimizer.weight_decay=0.01 \
    optimizer.b1=0.9 \
    optimizer.b2=0.95 \
    optimizer.eps=1.e-8 \
    optimizer.grad_acc_steps=4 \
    optimizer.max_grad_norm=1.0 \
    log_interval=10 \
    sync_interval=100 \
    eval_interval=1000 \
    save_interval=1000 \
    dtype="bfloat16" \
    seed=42 \
    num_workers=4 \
    use_chat_template=true \
    chat_template_mode="direct_encode" \
    eval.tasks="[arc_easy,arc_challenge,piqa,hellaswag,boolq]" \
    eval.lengths="[128,256,512]" \
    eval.tokens_per_batch=4096 \
    eval.add_bos=true \
    eval.chat_template_mode="surround_instruct" \
    eval.confirm_run_unsafe_code=true
