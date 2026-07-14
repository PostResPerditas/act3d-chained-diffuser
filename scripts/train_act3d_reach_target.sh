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
    main_keypose.py \
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
    --batch_size 8 \
    --batch_size_val 4 \
    --train_iters 30000 \
    --val_freq 500 \
    --lr 1e-4 \
    --use_instruction 0 \
    --weight_tying 1 \
    --gp_emb_tying 1 \
    --num_sampling_level 3 \
    --num_ghost_points 1000 \
    --num_ghost_points_val 10000 \
    --symmetric_rotation_loss 0 \
    --regress_position_offset 0 \
    --embedding_dim 60 \
    --position_loss_coeff 1 \
    --seed 0 \
    --exp_log_dir act3d_reach_target \
    --run_log_dir train100_val20_128