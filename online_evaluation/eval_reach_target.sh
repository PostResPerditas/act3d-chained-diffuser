#!/usr/bin/env bash
set -euo pipefail

root=/mnt/ssd/Bennkyou/PROJECT/act3d-chained-diffuser
cd "$root"
export PYTHONPATH="$root:${PYTHONPATH:-}"

task=reach_target
# exp=reach_target_smoke

# 在线评估必须读取 RLBench raw validation demos
# data_dir=$root/datasets/raw/reach_target_val20_128
data_dir=${DATA_DIR:-$root/datasets/raw/reach_target_val20_128}
robot_setup=${ROBOT_SETUP:-panda}
offline=${OFFLINE:-0}

instructions=$root/datasets/artifacts/reach_target/instructions_reach_target.pkl
# bounds=$root/datasets/artifacts/reach_target_100x20_128/reach_target_location_bounds.json
# act3d_checkpoint=$root/train_logs/act3d_reach_target/train100_val20_128/best.pth
# diff_checkpoint=$root/train_logs/trajectory_reach_target/train100_val20_128/best.pth

bounds=${BOUNDS_FILE:-$root/datasets/artifacts/reach_target_100x20_128/reach_target_location_bounds.json}
act3d_checkpoint=${ACT3D_CHECKPOINT:-$root/train_logs/act3d_reach_target/train100_val20_128/best.pth}
diff_checkpoint=${DIFF_CHECKPOINT:-$root/train_logs/trajectory_reach_target/train100_val20_128/best.pth}
exp=${EXP_NAME:-reach_target_evaluation}

image_size=128,128
cameras=left_shoulder,right_shoulder,wrist

num_episodes=${NUM_EPISODES:-20}
max_tries=3
interpolation_length=50

# 可通过环境变量切换评估模式
predict_keypose=${PREDICT_KEYPOSE:-1}
predict_traj=${PREDICT_TRAJ:-1}

echo "Evaluation configuration:"
echo "  robot_setup=$robot_setup"
echo "  data_dir=$data_dir"
echo "  num_episodes=$num_episodes"
echo "  predict_keypose=$predict_keypose"
echo "  predict_traj=$predict_traj"
echo "  bounds=$bounds"
echo "  act3d_checkpoint=$act3d_checkpoint"
echo "  diff_checkpoint=$diff_checkpoint"

headless=${HEADLESS:-0}
if [[ -z "${DISPLAY:-}" ]]; then
    echo "DISPLAY is empty. Run evaluation from an Ubuntu desktop terminal."
    exit 1
fi
echo "DISPLAY:           $DISPLAY"
echo "Headless:          $headless"

if [[ "$predict_keypose" == "1" && "$predict_traj" == "1" ]]; then
    mode=full_chain
elif [[ "$predict_keypose" == "1" ]]; then
    mode=keypose_only
elif [[ "$predict_traj" == "1" ]]; then
    mode=trajectory_gt_keypose
else
    # echo "predict_keypose 和 predict_traj 不能同时为 0"
    # exit 1
    mode=ground_truth_keypose
fi

for file in \
    "$data_dir" \
    "$instructions" \
    "$bounds" \
    "$act3d_checkpoint" \
    "$diff_checkpoint"
do
    if [[ ! -e "$file" ]]; then
        echo "Missing: $file"
        exit 1
    fi
done

mkdir -p "$root/eval_logs/$exp"

echo "Task:              $task"
echo "Mode:              $mode"
echo "Raw validation:    $data_dir"
echo "Act3D checkpoint:  $act3d_checkpoint"
echo "Diff checkpoint:   $diff_checkpoint"
echo "Image size:        $image_size"

# export DISPLAY=${DISPLAY:-:1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

python online_evaluation/eval1.py \
    --tasks "$task" \
    --variations 0 \
    --data_dir "$data_dir" \
    --num_episodes "$num_episodes" \
    --image_size "$image_size" \
    --cameras "$cameras" \
    --instructions "$instructions" \
    --act3d_checkpoint "$act3d_checkpoint" \
    --diff_checkpoint "$diff_checkpoint" \
    --model act3d \
    --traj_model diffusion \
    --predict_keypose "$predict_keypose" \
    --predict_traj "$predict_traj" \
    --action_dim 7 \
    --collision_checking 0 \
    --single_task_gripper_loc_bounds 1 \
    --gripper_loc_bounds_file "$bounds" \
    --act3d_gripper_loc_bounds_file "$bounds" \
    --use_instruction 1 \
    --act3d_use_instruction 0 \
    --dense_interpolation 1 \
    --interpolation_length "$interpolation_length" \
    --offline "$offline" \
    --headless "$headless" \
    --max_tries "$max_tries" \
    --max_steps -1 \
    --verbose 1 \
    --base_log_dir "$root/eval_logs" \
    --exp_log_dir "$exp" \
    --run_log_dir "${task}_${mode}" \
    --output_file "$root/eval_logs/$exp/${task}_${mode}.json"