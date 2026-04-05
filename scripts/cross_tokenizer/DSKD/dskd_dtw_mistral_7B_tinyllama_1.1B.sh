#! /bin/bash
# DSKD + Soft-DTW  |  Teacher: Mistral 7B  →  Student: TinyLLaMA 1.1B

MASTER_ADDR=localhost
MASTER_PORT=${2:-2012}
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${3:-1}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

BASE_PATH=${1:-$(pwd)}

# Student
CKPT_NAME="tinyllama-1.1B"
CKPT="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
PEFT_CKPT_NAME="tinyllama-1.1B"
PEFT_CKPT="${BASE_PATH}/results/tinyllama/train/init/${PEFT_CKPT_NAME}/"

# Teacher
TEACHER_CKPT_NAME="mistral-7B"
TEACHER_CKPT="mistralai/Mistral-7B-v0.1"
TEACHER_PEFT_CKPT_NAME="mistral-7B"
TEACHER_PEFT_CKPT="${BASE_PATH}/results/mistral/train/sft/${TEACHER_PEFT_CKPT_NAME}/"

# Mistral and TinyLLaMA share the same tokenizer (LLaMA tokenizer family),
# so a trivial identity mapping can be used. Replace with your actual mapping
# if tokenizers differ.
TOKEN_MAPPING="${BASE_PATH}/data/token_mapping/mistral_7B_to_tinyllama_id_mapping.json"
DTW_PROJECTOR="${BASE_PATH}/results/tinyllama/train/velocity_field/mistral_7B_tinyllama_1.1B/projector.pth"

DATA_DIR="${BASE_PATH}/processed_data/dolly/full/llama2/"
LM_DATA_DIR="${BASE_PATH}/processed_data/openwebtext/llama2/512/22.87K/"
SAVE_PATH="${BASE_PATH}/results/cross_tokenizer/dskd_dtw/mistral_7B_tinyllama_1.1B"

BATCH_SIZE=4
GRAD_ACC=4
LR=0.0001
EVAL_BATCH_SIZE=8
MAX_LENGTH=512
NUM_EPOCHS=10
SEED=42

OPTS=""
OPTS+=" --model-path ${CKPT}"
OPTS+=" --model-type llama2"
OPTS+=" --model-dtype bf16"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --teacher-model-path ${TEACHER_CKPT}"
OPTS+=" --teacher-model-type mistral"
OPTS+=" --teacher-ckpt-name ${TEACHER_CKPT_NAME}"
OPTS+=" --teacher-model-fp16"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --gradient-checkpointing"

OPTS+=" --peft lora"
OPTS+=" --peft-name ${PEFT_CKPT_NAME}"
OPTS+=" --peft-path ${PEFT_CKPT}"
OPTS+=" --teacher-peft-name ${TEACHER_PEFT_CKPT_NAME}"
OPTS+=" --teacher-peft-path ${TEACHER_PEFT_CKPT}"

OPTS+=" --teacher-to-student-id-mapping ${TOKEN_MAPPING}"

OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --lm-data-dir ${LM_DATA_DIR}"
OPTS+=" --num-workers 4"
OPTS+=" --dev-num 1000"
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-prompt-length 256"

OPTS+=" --do-train"
OPTS+=" --do-valid"
OPTS+=" --eval-gen"
OPTS+=" --num-epochs ${NUM_EPOCHS}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --lr ${LR}"
OPTS+=" --lr-decay-style cosine"
OPTS+=" --warmup-iters 0"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --save-interval -1"
OPTS+=" --eval-interval -1"
OPTS+=" --log-interval 10"
OPTS+=" --keep-best-n-checkpoints 3"
OPTS+=" --save-dir ${SAVE_PATH}"
OPTS+=" --seed ${SEED}"

OPTS+=" --criterion min_edit_dis_kld"
OPTS+=" --kd-rate 0.5"
OPTS+=" --kd-temperature 1.0"
OPTS+=" --kd-objective forward_kl"

# d-student=2048 (TinyLLaMA 1.1B), d-teacher=4096 (Mistral 7B)
OPTS+=" --dtw-weight 1.0"
OPTS+=" --dtw-window 32"
OPTS+=" --dtw-gamma 0.1"
OPTS+=" --dtw-distance cosine"
OPTS+=" --dtw-normalize"
OPTS+=" --dtw-unitization"
OPTS+=" --d-student 2048"
OPTS+=" --d-teacher 4096"
if [ -f "${DTW_PROJECTOR}" ]; then
    OPTS+=" --dtw-projector-path ${DTW_PROJECTOR}"
fi

OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"

OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"
OPTS+=" --base-path ${BASE_PATH}"

export NCCL_DEBUG=""
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/training/finetune_dskd_dtw.py ${OPTS} $@"

echo "======================================================================"
echo "  DSKD + Soft-DTW  |  Teacher: ${TEACHER_CKPT_NAME}  →  Student: ${CKPT_NAME}"
echo "  Save: ${SAVE_PATH}"
echo "======================================================================"
echo "${CMD}"
echo ""
mkdir -p "${SAVE_PATH}"
${CMD}
