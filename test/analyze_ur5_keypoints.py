from collections import Counter
from pathlib import Path
from typing import Dict, List

import os
import sys

import numpy as np


ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.append(ROOT_DIR)

from online_evaluation.utils_with_rlbench import RLBenchEnv
from utils.utils_with_rlbench import keypoint_discovery


ROOT = Path(
    "/mnt/ssd/Bennkyou/PROJECT/"
    "act3d-chained-diffuser"
)

TASK = "reach_target"
VARIATION = 0
IMAGE_SIZE = [128, 128]

DATASETS = {
    "train": (
        ROOT
        / "datasets/raw/"
        "ur5_reach_target_train100_128"
    ),
    "val": (
        ROOT
        / "datasets/raw/"
        "ur5_reach_target_val20_128"
    ),
}

STOPPING_DELTAS = [
    0.10,
    0.05,
    0.02,
    0.01,
]

# 只详细分析这些关键帧数量。
# 当前主要关注 4-keypoint episode。
DETAIL_KEYPOINT_COUNTS = {4}

# 是否同时详细打印 2-keypoint episode。
# 调试夹爪关闭来源时可临时改成 True。
ANALYZE_TWO_KEYPOINT_EPISODES = False


def count_episodes(data_dir: Path) -> int:
    episodes_dir = (
        data_dir
        / TASK
        / f"variation{VARIATION}"
        / "episodes"
    )

    return len(list(episodes_dir.glob("episode*")))


def quaternion_angle_deg(
    q1: np.ndarray,
    q2: np.ndarray,
) -> float:
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)

    q1 /= np.linalg.norm(q1) + 1e-12
    q2 /= np.linalg.norm(q2) + 1e-12

    # q 与 -q 表示相同旋转，因此取绝对值。
    dot = np.clip(
        np.abs(np.dot(q1, q2)),
        0.0,
        1.0,
    )

    return float(
        np.degrees(
            2.0 * np.arccos(dot)
        )
    )


def get_gripper_change_frames(
    demo,
) -> List[int]:
    """找出 gripper_open 数值发生变化的帧。"""

    change_frames: List[int] = []

    for frame_id in range(1, len(demo)):
        previous_state = float(
            demo[frame_id - 1].gripper_open
        )
        current_state = float(
            demo[frame_id].gripper_open
        )

        if current_state != previous_state:
            change_frames.append(frame_id)

    return change_frames


def get_frame_reasons(
    demo,
    frame_id: int,
    gripper_change_frames: List[int],
    stopping_delta: float,
) -> List[str]:
    """
    给关键帧添加诊断标签。

    注意：
    这里用于诊断，并不替代项目中的 keypoint_discovery。
    """

    reasons: List[str] = []

    if frame_id in gripper_change_frames:
        reasons.append("gripper_change")

    if frame_id == len(demo) - 1:
        reasons.append("terminal")

    joint_velocities = np.asarray(
        demo[frame_id].joint_velocities,
        dtype=np.float64,
    )

    if np.all(
        np.abs(joint_velocities)
        <= stopping_delta
    ):
        reasons.append("joint_stopped")

    if not reasons:
        reasons.append("other")

    return reasons


def print_episode_detail(
    split_name: str,
    episode: int,
    demo,
    keypoints: List[int],
    stopping_delta: float = 0.1,
) -> None:
    """打印单条 Demo 的详细关键帧及夹爪状态。"""

    gripper_change_frames = (
        get_gripper_change_frames(demo)
    )

    initial_gripper = float(
        demo[0].gripper_open
    )
    final_gripper = float(
        demo[-1].gripper_open
    )

    print("\n" + "=" * 80)
    print(
        f"[Episode Detail] "
        f"split={split_name}, "
        f"episode={episode}"
    )

    print(
        f"demo_length={len(demo)}, "
        f"keypoints={keypoints}, "
        f"num_keypoints={len(keypoints)}"
    )

    print(
        f"initial_gripper={initial_gripper:.1f}, "
        f"final_gripper={final_gripper:.1f}"
    )

    print(
        f"gripper_change_frames="
        f"{gripper_change_frames}"
    )

    for frame_id in gripper_change_frames:
        previous_state = float(
            demo[frame_id - 1].gripper_open
        )
        current_state = float(
            demo[frame_id].gripper_open
        )

        print(
            f"  gripper frame {frame_id}: "
            f"{previous_state:.1f} "
            f"-> {current_state:.1f}"
        )
        print_gripper_window(
            demo=demo,
            change_frame=frame_id,
            radius=3,
        )
    final_pose = np.asarray(
        demo[-1].gripper_pose,
        dtype=np.float64,
    )

    previous_frame = 0

    for keypoint_id, frame_id in enumerate(
        keypoints
    ):
        obs = demo[frame_id]
        previous_obs = demo[previous_frame]

        pose = np.asarray(
            obs.gripper_pose,
            dtype=np.float64,
        )

        previous_pose = np.asarray(
            previous_obs.gripper_pose,
            dtype=np.float64,
        )

        joint_velocities = np.asarray(
            obs.joint_velocities,
            dtype=np.float64,
        )

        position_delta = np.linalg.norm(
            pose[:3] - previous_pose[:3]
        )

        rotation_delta = quaternion_angle_deg(
            pose[3:7],
            previous_pose[3:7],
        )

        distance_to_final = np.linalg.norm(
            pose[:3] - final_pose[:3]
        )

        rotation_to_final = quaternion_angle_deg(
            pose[3:7],
            final_pose[3:7],
        )

        reasons = get_frame_reasons(
            demo=demo,
            frame_id=frame_id,
            gripper_change_frames=(
                gripper_change_frames
            ),
            stopping_delta=stopping_delta,
        )

        print(
            f"\n  keypoint[{keypoint_id}]"
            f" frame={frame_id},"
            f" gap={frame_id - previous_frame}"
        )

        print(
            f"    reasons={reasons}"
        )

        print(
            f"    terminal="
            f"{frame_id == len(demo) - 1}, "
            f"gripper="
            f"{float(obs.gripper_open):.1f}"
        )

        print(
            f"    position="
            f"{np.round(pose[:3], 4).tolist()}"
        )

        print(
            f"    delta_from_previous:"
            f" position={position_delta:.4f} m,"
            f" rotation={rotation_delta:.2f} deg"
        )

        print(
            f"    distance_to_final:"
            f" position={distance_to_final:.4f} m,"
            f" rotation={rotation_to_final:.2f} deg"
        )

        print(
            f"    joint_velocity:"
            f" max_abs="
            f"{np.max(np.abs(joint_velocities)):.8f},"
            f" mean_abs="
            f"{np.mean(np.abs(joint_velocities)):.8f}"
        )

        previous_frame = frame_id


def analyze_split(
    split_name: str,
    data_dir: Path,
) -> None:
    env = RLBenchEnv(
        data_path=data_dir,
        image_size=IMAGE_SIZE,
        apply_rgb=False,
        apply_depth=False,
        apply_pc=False,
        apply_cameras=(),
        headless=True,
        robot_setup="ur5",
    )

    episode_count = count_episodes(
        data_dir
    )

    histograms: Dict[
        float,
        Counter,
    ] = {
        delta: Counter()
        for delta in STOPPING_DELTAS
    }

    four_keypoint_episodes = {
        delta: []
        for delta in STOPPING_DELTAS
    }

    # 缓存 delta=0.1 下的 Demo 与关键帧，
    # 用于后续详细分析，避免重复读取。
    demos = {}
    reference_keypoints = {}

    for episode in range(episode_count):
        demo = env.get_demo(
            TASK,
            VARIATION,
            episode,
        )[0]

        demos[episode] = demo

        for delta in STOPPING_DELTAS:
            keypoints = keypoint_discovery(
                demo,
                stopping_delta=delta,
            )

            number = len(keypoints)
            histograms[delta][number] += 1

            if delta == STOPPING_DELTAS[0]:
                reference_keypoints[
                    episode
                ] = keypoints

            if number >= 4:
                four_keypoint_episodes[
                    delta
                ].append(episode)

    print("\n" + "#" * 80)
    print(f"Split: {split_name}")
    print(f"Episodes: {episode_count}")

    for delta in STOPPING_DELTAS:
        print(
            f"stopping_delta={delta:.3f}: "
            f"{dict(sorted(histograms[delta].items()))}"
        )

        print(
            "  episodes with >=4 keypoints:",
            four_keypoint_episodes[delta],
        )

    # 详细分析 delta=0.1 时的关键帧。
    for episode in range(episode_count):
        keypoints = reference_keypoints[
            episode
        ]

        should_print = (
            len(keypoints)
            in DETAIL_KEYPOINT_COUNTS
        )

        if (
            ANALYZE_TWO_KEYPOINT_EPISODES
            and len(keypoints) == 2
        ):
            should_print = True

        if not should_print:
            continue

        print_episode_detail(
            split_name=split_name,
            episode=episode,
            demo=demos[episode],
            keypoints=keypoints,
            stopping_delta=0.1,
        )

def get_commanded_gripper(obs):
    misc = getattr(obs, "misc", None)

    if not isinstance(misc, dict):
        return None

    joint_action = misc.get(
        "joint_position_action"
    )

    if joint_action is None:
        return None

    joint_action = np.asarray(
        joint_action,
        dtype=np.float64,
    ).reshape(-1)

    if len(joint_action) == 0:
        return None

    return float(joint_action[-1])

def print_gripper_window(
    demo,
    change_frame: int,
    radius: int = 3,
) -> None:
    start = max(0, change_frame - radius)
    end = min(
        len(demo),
        change_frame + radius + 1,
    )

    print(
        f"\n  Gripper window around "
        f"frame {change_frame}:"
    )

    for frame_id in range(start, end):
        obs = demo[frame_id]

        measured_gripper = float(
            obs.gripper_open
        )

        commanded_gripper = (
            get_commanded_gripper(obs)
        )

        gripper_joints = np.asarray(
            obs.gripper_joint_positions,
            dtype=np.float64,
        )

        print(
            f"    frame={frame_id:4d}, "
            f"measured={measured_gripper:.1f}, "
            f"commanded={commanded_gripper}, "
            f"gripper_joints="
            f"{np.round(gripper_joints, 6).tolist()}"
        )

if __name__ == "__main__":
    for name, directory in DATASETS.items():
        analyze_split(
            name,
            directory,
        )