#!/usr/bin/env bash
set -euo pipefail

root=/mnt/ssd/Bennkyou/PROJECT/act3d-chained-diffuser
cd "$root"

export PYTHONPATH="$root:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

dataset=$root/datasets/packaged/reach_target_train100_128
valset=$root/datasets/packaged/reach_target_val20_128
instructions=$root/datasets/artifacts/reach_target/instructions_reach_target.pkl
bounds=$root/datasets/artifacts/reach_target_100x20_128/reach_target_location_bounds.json

task=reach_target
image_size=128,128

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
    --image_rescale 0.9,1.1 \
    --max_episodes_per_task 100 \
    --max_episode_length 5 \
    --num_workers 2 \
    --cache_size 20 \
    --cache_size_val 10 \
    --batch_size 4 \
    --batch_size_val 4 \
    --train_iters 50000 \
    --val_freq 1000 \
    --lr 1e-4 \
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
    --dense_interpolation 1 \
    --interpolation_length 50 \
    --exp_log_dir trajectory_reach_target \
    --run_log_dir train100_val20_128