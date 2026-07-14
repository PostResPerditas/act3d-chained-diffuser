#!/usr/bin/env bash
set -euo pipefail

root=/mnt/ssd/Bennkyou/PROJECT/act3d-chained-diffuser
cd "$root"

dataset=$root/datasets/packaged/smoke_reach_target_train
valset=$root/datasets/packaged/smoke_reach_target_val

instructions=$root/datasets/artifacts/reach_target/instructions_reach_target.pkl
bounds=$root/datasets/artifacts/reach_target/reach_target_location_bounds.json

task=reach_target
image_size=128,128

main_dir=act3d_smoke
run_dir=reach_target_5train_2val

batch_size=1
batch_size_val=1
lr=1e-4

# 最小化测试配置
train_iters=100
val_freq=20
max_episodes_per_task=5
max_episode_length=3

num_ghost_points=100
num_ghost_points_val=500

weight_tying=1
gp_emb_tying=1
num_sampling_level=3
regress_position_offset=0
symmetric_rotation_loss=0
embedding_dim=60
seed=0

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port="$((20000 + RANDOM % 20000))" \
    main_keypose.py \
    --tasks "$task" \
    --variations 0 \
    --dataset "$dataset" \
    --valset "$valset" \
    --instructions "$instructions" \
    --gripper_loc_bounds "$bounds" \
    --image_size "$image_size" \
    --max_episodes_per_task "$max_episodes_per_task" \
    --max_episode_length "$max_episode_length" \
    --num_workers 0 \
    --cache_size 0 \
    --cache_size_val 0 \
    --batch_size "$batch_size" \
    --batch_size_val "$batch_size_val" \
    --train_iters "$train_iters" \
    --val_freq "$val_freq" \
    --lr "$lr" \
    --use_instruction 0 \
    --weight_tying "$weight_tying" \
    --gp_emb_tying "$gp_emb_tying" \
    --num_sampling_level "$num_sampling_level" \
    --num_ghost_points "$num_ghost_points" \
    --num_ghost_points_val "$num_ghost_points_val" \
    --symmetric_rotation_loss "$symmetric_rotation_loss" \
    --regress_position_offset "$regress_position_offset" \
    --embedding_dim "$embedding_dim" \
    --image_rescale 1.0,1.0 \
    --position_loss_coeff 1 \
    --seed "$seed" \
    --exp_log_dir "$main_dir" \
    --run_log_dir "$run_dir"