#! /bin/bash

MASTER_ADDR=localhost
MASTER_PORT=${2-2012}
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${3-16}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

# 2. PATH & MODEL CONFIGURATION
BASE_PATH=${1:-$(pwd)}

#  STUDENT & TEACHER (Use HF ID for stability) 
CKPT_NAME="gpt2-120M-dolly"
# Use HF ID instead of old local path
CKPT="bachthetrollface/gpt2-120M-init-dolly"

TEACHER_CKPT_NAME="gpt2-1.5B-dolly"
# Use HF ID instead of old local path
TEACHER_CKPT="bachthetrollface/gpt2-1.5B-teacher-dolly"

VF_DIR="${BASE_PATH}/results/gpt2/train/velocity_field/distillm2"
PROJECTOR_CKPT="${VF_DIR}/projector.pth"

#  DATA PATHS 
# Formatted DistiLLM-2 data (JSONL pairs)
DISTILLM2_DATA_DIR="${BASE_PATH}/data/distillm2/gpt2/formatted"
# Original data for evaluation (Ground Truth)
DATA_DIR="${BASE_PATH}/processed_data/dolly/full/gpt2/"
# OpenWebText data (Optional)
LM_DATA_DIR="${BASE_PATH}/processed_data/openwebtext/gpt2/512/22.87K/"

# 3. HYPERPARAMETERS
BATCH_SIZE=8      # 16 * 2 GPU = Global Batch 32 (Fine for A100)
LR=0.0005
GRAD_ACC=2
EVAL_BATCH_SIZE=64
MAX_LENGTH=512
SAVE_PATH="${BASE_PATH}/results/gpt2/train/contra/distillm2/dtw_0.1B_1.5B"
SEED=10

# DistiLLM-2 Specs
LOSS_TYPE="distillm_v2"
BASE_ALPHA_1=0.1
BASE_ALPHA_2=0.1

# 4. CREATE OPTIONS
OPTS=""
# Model Config
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --teacher-model-path ${TEACHER_CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --teacher-ckpt-name ${TEACHER_CKPT_NAME}"
OPTS+=" --teacher-model-fp16"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --model-type gpt2"

# [CRITICAL FIX] Disable Gradient Checkpointing to avoid errors with DeepSpeed + Frozen Modules
# OPTS+=" --gradient-checkpointing" 

# Data Config
OPTS+=" --data-dir ${DISTILLM2_DATA_DIR}"
if [ -n "$LM_DATA_DIR" ]; then
    OPTS+=" --lm-data-dir ${LM_DATA_DIR}"
fi
OPTS+=" --gt-data-dir ${DATA_DIR}"
OPTS+=" --num-workers 4"
OPTS+=" --dev-num 1000"

# Training Config
OPTS+=" --lr ${LR}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --warmup-iters 0"
OPTS+=" --lr-decay-style cosine"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --epochs 20"
OPTS+=" --kd-ratio 1.0"

# DistiLLM-2 Config
OPTS+=" --distillm2-loss-type ${LOSS_TYPE}"
OPTS+=" --base-alpha-1 ${BASE_ALPHA_1}"
OPTS+=" --base-alpha-2 ${BASE_ALPHA_2}"
OPTS+=" --gradual-beta"

# Runtime Config
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-prompt-length 256"

OPTS+=" --do-train"
OPTS+=" --do-valid"
OPTS+=" --eval-gen"
OPTS+=" --save-interval -1"
OPTS+=" --eval-interval -1"
OPTS+=" --log-interval 10"
OPTS+=" --mid-log-num -1"
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --seed ${SEED}"

# DeepSpeed Config
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"

# Method Type
OPTS+=" --type distillm2-v2-dtw"

# Generation Config
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"


OPTS+=" --student-gen"
OPTS+=" --gen-num-beams 1"
OPTS+=" --gen-top-p 1.0"
OPTS+=" --init-threshold 0.0"
OPTS+=" --loss-eps 0.1"
OPTS+=" --capacity 1000"


# DTW Config
OPTS+=" --dtw-weight 1.0"
OPTS+=" --dtw-window 32"
OPTS+=" --dtw-gamma 0.1"
OPTS+=" --dtw-distance cosine"
OPTS+=" --dtw-normalize"

OPTS+=" --dtw-unitization"
# OPTS+=" --dtw-importance-weights teacher_entropy"

OPTS+=" --d-teacher 1600"
OPTS+=" --d-student 768"
OPTS+=" --projector-path ${PROJECTOR_CKPT}"

# 5. ENVIRONMENT & RUN COMMAND
export NCCL_DEBUG=""
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}

# Fix GLIBCXX library error
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/hungpv/miniconda3/envs/warp/lib

# Fix OOM and memory fragmentation errors
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/training/finetune.py ${OPTS} $@"

echo "-------------------------------------------------------"
echo "Starting DistiLLM-2 + Contra-KD Finetuning"
echo "Student: ${CKPT}"
# echo "Velocity Field: ${VELOCITY_FIELD_CKPT}"
echo "Save Path: ${SAVE_PATH}"
echo "-------------------------------------------------------"
echo ${CMD}

mkdir -p ${SAVE_PATH}
# Run command
${CMD}