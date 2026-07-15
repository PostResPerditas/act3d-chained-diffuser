from typing import Dict, List, Tuple

import numpy as np
from rlbench.demo import Demo


def discover_keypoints_with_reason(
    demo: Demo,
    stopping_delta: float = 0.1,
    stopped_buffer_size: int = 4,
) -> Tuple[List[int], Dict[int, List[str]]]:
    keypoints: List[int] = []
    reasons: Dict[int, List[str]] = {}

    previous_gripper_open = demo[0].gripper_open
    stopped_buffer = 0

    for frame_id, obs in enumerate(demo):
        is_first = frame_id == 0
        is_terminal = frame_id == len(demo) - 1

        gripper_changed = (
            obs.gripper_open != previous_gripper_open
        )

        joint_velocities = np.asarray(
            obs.joint_velocities,
            dtype=np.float64,
        )

        velocity_stopped = np.allclose(
            joint_velocities,
            0.0,
            atol=stopping_delta,
        )

        # 避免把夹爪变化附近重复算成停止关键点
        next_gripper_same = True
        if frame_id < len(demo) - 1:
            next_gripper_same = (
                demo[frame_id + 1].gripper_open
                == obs.gripper_open
            )

        stopped = (
            not is_first
            and not is_terminal
            and stopped_buffer <= 0
            and velocity_stopped
            and not gripper_changed
            and next_gripper_same
        )

        frame_reasons: List[str] = []

        if gripper_changed:
            frame_reasons.append("gripper_change")

        if stopped:
            frame_reasons.append("stopped")

        if is_terminal:
            frame_reasons.append("terminal")

        if not is_first and frame_reasons:
            keypoints.append(frame_id)
            reasons[frame_id] = frame_reasons

        if stopped:
            stopped_buffer = stopped_buffer_size
        else:
            stopped_buffer -= 1

        previous_gripper_open = obs.gripper_open

    # 与原关键帧检测中常见的终点去重逻辑保持一致
    if (
        len(keypoints) > 1
        and keypoints[-1] - 1 == keypoints[-2]
    ):
        removed = keypoints.pop(-2)
        reasons.pop(removed, None)

    return keypoints, reasons

keypoints, reasons = discover_keypoints_with_reason(
    demo,
    stopping_delta=0.1,
)

print(
    f"episode={episode}, "
    f"keypoints={keypoints}"
)

for frame_id in keypoints:
    obs = demo[frame_id]

    print(
        f"  frame={frame_id}, "
        f"reasons={reasons[frame_id]}, "
        f"gripper={obs.gripper_open}, "
        f"max_abs_joint_velocity="
        f"{np.max(np.abs(obs.joint_velocities)):.8f}"
    )