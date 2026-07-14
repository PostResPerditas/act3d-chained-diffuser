#!/usr/bin/env bash
set -euo pipefail

root=/mnt/ssd/Bennkyou/PROJECT/act3d-chained-diffuser
cd "$root"

# 和 Act3D 使用同一套 packaged 数据
dataset=$root/datasets/packaged/smoke_reach_target_train
valset=$root/datasets/packaged/smoke_reach_target_val

instructions=$root/datasets/artifacts/reach_target/instructions_reach_target.pkl
bounds=$root/datasets/artifacts/reach_target/reach_target_location_bounds.json

task=reach_target
image_size=128,128

main_dir=trajectory_smoke
run_dir=reach_target_5train_2val

lr=1e-4

# 轨迹处理
dense_interpolation=1
interpolation_length=50

# 最小测试配置
batch_size=1
batch_size_val=1
train_iters=100
val_freq=20
max_episodes_per_task=5
max_episode_length=3
num_workers=0

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

torchrun \
    --standalone \
    --nproc_per_node=1 \
    main_trajectory.py \
    --tasks "$task" \
    --variations 0 \
    --dataset "$dataset" \
    --valset "$valset" \
    --instructions "$instructions" \
    --gripper_loc_bounds "$bounds" \
    --image_size "$image_size" \
    --image_rescale 1.0,1.0 \
    --max_episodes_per_task "$max_episodes_per_task" \
    --max_episode_length "$max_episode_length" \
    --num_workers "$num_workers" \
    --train_iters "$train_iters" \
    --val_freq "$val_freq" \
    --embedding_dim 120 \
    --action_dim 7 \
    --num_query_cross_attn_layers 6 \
    --num_vis_ins_attn_layers 2 \
    --use_instruction 1 \
    --use_goal 1 \
    --use_goal_at_test 1 \
    --feat_scales_to_use 1 \
    --attn_rounds 1 \
    --weight_tying 1 \
    --rotation_parametrization 6D \
    --diffusion_timesteps 100 \
    --dense_interpolation "$dense_interpolation" \
    --interpolation_length "$interpolation_length" \
    --batch_size "$batch_size" \
    --batch_size_val "$batch_size_val" \
    --cache_size 0 \
    --cache_size_val 0 \
    --lr "$lr" \
    --exp_log_dir "$main_dir" \
    --run_log_dir "$run_dir"