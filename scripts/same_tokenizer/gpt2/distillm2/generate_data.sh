set -e

# Use the current directory as the base path if no parameter is passed
BASE_PATH=${1:-$(pwd)}

# SYSTEM LIBRARY ERROR FIX
# By pointing to the lib directory of the Conda 'warp' environment
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/hungpv/miniconda3/envs/warp/lib

# Helps avoid memory fragmentation errors (OOM) during long runs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Teacher: GPT-2 XL (1.5B) Fine-tuned (SFT)
TEACHER_MODEL="bachthetrollface/gpt2-1.5B-teacher-dolly"

# Student: GPT-2 Base (120M) initialized (Init)
STUDENT_MODEL="bachthetrollface/gpt2-120M-init-dolly"

# Data paths
DATA_DIR="${BASE_PATH}/data/dolly" 
OUTPUT_DIR="${BASE_PATH}/data/distillm2/gpt2"

mkdir -p ${OUTPUT_DIR}

echo "================================================"
echo "DistiLLM-2 Data Generation for GPT-2"
echo "Teacher: ${TEACHER_MODEL}"
echo "Student: ${STUDENT_MODEL}"
echo "Data Dir: ${DATA_DIR}"
echo "Config: LD_LIBRARY_PATH set to fix GLIBCXX error"
echo "================================================"

# Generate teacher and student responses for train and dev splits
echo ""
echo "Generating teacher and student responses..."

for SPLIT in train dev; do
    echo "Processing ${SPLIT} split..."
    
    # Check if the input file exists
    if [ ! -f "${DATA_DIR}/${SPLIT}.jsonl" ]; then
        echo "Error: File ${DATA_DIR}/${SPLIT}.jsonl not found!"
        exit 1
    fi

    python generate_distillm2_data.py \
        --teacher-model ${TEACHER_MODEL} \
        --student-model ${STUDENT_MODEL} \
        --data-path ${DATA_DIR}/${SPLIT}.jsonl \
        --output-dir ${OUTPUT_DIR} \
        --split-type ${SPLIT} \
        --temperature 1.0 \
        --top-p 0.95 \
        --max-tokens 512 \
        --use-vllm \
        --tensor-parallel-size 1 
done

echo ""
echo "================================================"
echo "Reformatting data for training..."
echo "================================================"

python reformat_distillm2_data.py \
    --input-dir ${OUTPUT_DIR} \
    --output-dir ${OUTPUT_DIR}/formatted

echo ""
echo "================================================"
echo "Data generation complete!"
echo "Output directory: ${OUTPUT_DIR}/formatted"
echo " - train.json (reformatted paired data)"
echo "================================================"