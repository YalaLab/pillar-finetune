# Download models from huggingface
hf download YalaLab/Pillar0-Sybil-1.5 --local-dir logs/checkpoints 

# Configuration
CONFIG_FILE="${1:-configs/csv_dataset_setup.yaml}"

OMP_NUM_THREADS=$OMP_NUM_THREADS \
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    scripts/train.py \
    $CONFIG_FILE \
    --resume logs/checkpoints/seed0/epoch=2.ckpt \
    --evaluate \
    --opts \
    experiment.name seed0 \
    engine.max_epochs 3 