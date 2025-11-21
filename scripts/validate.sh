# Download models from huggingface
uv run hf download YalaLab/Pillar0-Sybil-1.5 --local-dir logs/checkpoints 

# Configuration
CONFIG_FILE="${1:-configs/csv_dataset.yaml}"

OMP_NUM_THREADS=$OMP_NUM_THREADS \
uv run torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    scripts/train.py \
    $CONFIG_FILE \
    --resume logs/checkpoints/seed0/epoch=2.ckpt \
    --evaluate \
    --opts \
    experiment.name seed0 \
    engine.max_epochs 3 

OMP_NUM_THREADS=$OMP_NUM_THREADS \
uv run torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    scripts/train.py \
    $CONFIG_FILE \
    --resume logs/checkpoints/seed1/epoch=2.ckpt \
    --evaluate \
    --opts \
    experiment.name seed1 \
    engine.max_epochs 3 

OMP_NUM_THREADS=$OMP_NUM_THREADS \
uv run torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    scripts/train.py \
    $CONFIG_FILE \
    --resume logs/checkpoints/seed2/epoch=2.ckpt \
    --evaluate \
    --opts \
    experiment.name seed2 \
    engine.max_epochs 3 

uv run scripts/ensemble.py \
    --results \
    logs/csv/seed0/checkpoints/3 \
    logs/csv/seed1/checkpoints/3 \
    logs/csv/seed2/checkpoints/3 \
    --config $CONFIG_FILE
