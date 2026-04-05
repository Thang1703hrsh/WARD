#! /bin/bash

# =========================================================
# 1. CẤU HÌNH MẠNG & GPU
# =========================================================
MASTER_ADDR=localhost
MASTER_PORT=2012
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=2  # Giữ nguyên 2 GPU như bạn đang dùng

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

# =========================================================
# 2. CẤU HÌNH ĐƯỜNG DẪN & MODEL (ĐÃ SỬA)
# =========================================================
# Lấy thư mục hiện tại làm gốc
BASE_PATH=${1:-$(pwd)}

# --- STUDENT (GPT-2 120M) ---
CKPT_NAME="gpt2-120M-dolly"
# Sử dụng model trên HF thay vì đường dẫn local cũ
CKPT="bachthetrollface/gpt2-120M-init-dolly"

# --- TEACHER (GPT-2 1.5B/XL) ---
TEACHER_NAME="gpt2-1.5B-dolly"
# Sử dụng model trên HF
TEACHER_MODEL_PATH="bachthetrollface/gpt2-1.5B-teacher-dolly"

# --- DATA ---
DATA_DIR="${BASE_PATH}/processed_data/dolly/full/gpt2/"

# =========================================================
# 3. HYPERPARAMETERS
# =========================================================
BATCH_SIZE=16        # Giảm xuống 8 vì đang chạy 2 GPU (Tổng batch = 16)
LR=0.0005
GRAD_ACC=2
EVAL_BATCH_SIZE=32

# Length
MAX_LENGTH=512

# Output Path
SAVE_PATH="${BASE_PATH}/results/gpt2/train/velocity_field"
SEED=10

# =========================================================
# 4. TẠO OPTIONS
# =========================================================
OPTS=""
# Model Config
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --teacher-model-path ${TEACHER_MODEL_PATH}"
# OPTS+=" --gradient-checkpointing" # Tắt Checkpointing cho model nhỏ 120M

# Data Config
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --num-workers 4"  # Tăng worker lên 4 để load data nhanh hơn (tránh = 0)
OPTS+=" --dev-num 1000"

# Optimizer Config
OPTS+=" --lr ${LR}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --warmup-iters 0"
OPTS+=" --lr-decay-style cosine"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --epochs 3"

# Length Config
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-prompt-length 256"

# Runtime Config
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

OPTS+=" --type lm"

# Gen Config
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"

# --- CONTRA-KD ARCHITECTURE CONFIG ---
# Cấu hình chính xác cho cặp GPT-2 120M vs GPT-2 1.5B

OPTS+=" --d-teacher 1600"        # GPT-2 XL (1.5B) hidden size
OPTS+=" --d-student 768"         # GPT-2 Base (120M) hidden size
OPTS+=" --num-distill-layers 6"
OPTS+=" --num-teacher-layers 48" # GPT-2 XL layers
OPTS+=" --num-student-layers 12" # GPT-2 Base layers
OPTS+=" --teacher-device 0"
OPTS+=" --student-device 0"

# =========================================================
# 5. CHẠY LỆNH
# =========================================================
export NCCL_DEBUG=""
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}

# Thêm config quản lý bộ nhớ để tránh OOM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/train_velocity_field.py ${OPTS} $@"

echo "-------------------------------------------------------"
echo "Starting Velocity Field Training"
echo "Student: ${CKPT}"
echo "Teacher: ${TEACHER_MODEL_PATH}"
echo "Save Path: ${SAVE_PATH}"
echo "-------------------------------------------------------"
echo ${CMD}

mkdir -p ${SAVE_PATH}
${CMD}