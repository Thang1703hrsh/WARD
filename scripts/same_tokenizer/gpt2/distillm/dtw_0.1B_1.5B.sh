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


# PATH & MODEL CONFIGURATION
# Use the current directory as the base path
BASE_PATH=${1:-$(pwd)}

# STUDENT (GPT-2 120M)
CKPT_NAME="gpt2-120M-dolly"
CKPT="bachthetrollface/gpt2-120M-init-dolly"

# TEACHER (GPT-2 1.5B)
TEACHER_CKPT_NAME="gpt2-1.5B-dolly"
TEACHER_CKPT="bachthetrollface/gpt2-1.5B-teacher-dolly"

# PROJECTOR (from previous training step)
VF_SAVE_DIR="${BASE_PATH}/results/gpt2/train/velocity_field"
PROJECTOR_CKPT="${VF_SAVE_DIR}/projector.pth"

# DATA
DATA_DIR="${BASE_PATH}/processed_data/dolly/full/gpt2/"

# OpenWebText data for LM loss
LM_DATA_DIR="${BASE_PATH}/processed_data/openwebtext/gpt2/512/22.87K/"


# HYPERPARAMETERS
BATCH_SIZE=16
LR=0.0005
GRAD_ACC=1         # Increase gradient accumulation to compensate for small batch size
EVAL_BATCH_SIZE=64
MAX_LENGTH=512
SAVE_PATH="${BASE_PATH}/results/llama/train/dtw/distillm/0.1B_1.5B"
SEED=10


# CREATE OPTIONS

OPTS=""
# Model Config
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --teacher-model-path ${TEACHER_CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --teacher-ckpt-name ${TEACHER_CKPT_NAME}"
OPTS+=" --teacher-model-fp16"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"

OPTS+=" --gradient-checkpointing"

# Data Config
OPTS+=" --data-dir ${DATA_DIR}"
if [ -n "$LM_DATA_DIR" ]; then
    OPTS+=" --lm-data-dir ${LM_DATA_DIR}"
fi
OPTS+=" --num-workers 4"
OPTS+=" --dev-num 1000"

# HP Config
OPTS+=" --lr ${LR}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --warmup-iters 0"  # Add warmup for stable training
OPTS+=" --lr-decay-style cosine"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --epochs 20"
OPTS+=" --kd-ratio 1.0"

# Length Config
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-prompt-length 256"

# Runtime Config
OPTS+=" --do-train"
OPTS+=" --do-valid"
OPTS+=" --eval-gen"
OPTS+=" --save-interval -1"    # Save every 2 epochs to track progress
OPTS+=" --eval-interval -1"     # Evaluate every epoch
OPTS+=" --log-interval 10"
OPTS+=" --mid-log-num -1"
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --seed ${SEED}"

# DeepSpeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"

# Type: Skewed Reverse KL + DTW (without adaptive threshold)
OPTS+=" --type adaptive-srkl-dtw"
OPTS+=" --skew-alpha 0.1"

# Generation Config (for evaluation only, not training)
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


# ENVIRONMENT & RUN COMMAND

export NCCL_DEBUG=""
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}

# Fix GLIBCXX library error (from previous step)
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/hungpv/miniconda3/envs/warp/lib

# Fix OOM error
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/training/finetune.py ${OPTS} $@"

echo "-------------------------------------------------------"
echo "Starting DTW Distill Finetuning"
echo "Student: ${CKPT}"
echo "Teacher: ${TEACHER_CKPT}"
echo "Projector: ${PROJECTOR_CKPT}"
echo "Data: ${DATA_DIR}"
echo "Save Path: ${SAVE_PATH}"
echo "-------------------------------------------------------"
echo ${CMD}

mkdir -p ${SAVE_PATH}
CODE_BASE=HF ${CMD}