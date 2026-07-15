import random
import itertools
from typing import Tuple, Dict, List
import blosc
import pickle
from pathlib import Path
import json
from tqdm import tqdm
import tap
import torch
import numpy as np
import einops
from rlbench.demo import Demo
from online_evaluation.utils_with_rlbench import RLBenchEnv
from utils.utils_with_rlbench import (
    keypoint_discovery,
    obs_to_attn,
    transform,
)


class Arguments(tap.Tap):
    data_dir: Path = Path(__file__).parent / "c2farm"
    seed: int = 2
    tasks: Tuple[str, ...] = ("stack_wine",)
    cameras: Tuple[str, ...] = ("left_shoulder", "right_shoulder", "wrist", "front")
    image_size: str = "256,256"
    output: Path = Path(__file__).parent / "datasets"
    max_variations: int = 199
    offset: int = 0
    num_workers: int = 0
    store_intermediate_actions: int = 1

    debug_keypoints: int = 0

def get_attn_indices_from_demo(
    task_str: str, demo: Demo, cameras: Tuple[str, ...]
) -> List[Dict[str, Tuple[int, int]]]:
    frames = keypoint_discovery(demo)

    frames.insert(0, 0)
    return [{cam: obs_to_attn(demo[f], cam) for cam in cameras} for f in frames]

def quaternion_angle_deg(q1, q2) -> float:
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)

    q1 /= np.linalg.norm(q1) + 1e-12
    q2 /= np.linalg.norm(q2) + 1e-12

    dot = np.clip(
        np.abs(np.dot(q1, q2)),
        0.0,
        1.0,
    )

    return float(
        np.degrees(2.0 * np.arccos(dot))
    )


def print_keypoint_debug(
    task_str: str,
    variation: int,
    episode: int,
    demo: Demo,
    keypoints: List[int],
) -> None:
    final_pose = np.asarray(
        demo._observations[-1].gripper_pose,
        dtype=np.float64,
    )

    print(
        f"\n[Keypoint Debug] "
        f"task={task_str}, "
        f"variation={variation}, "
        f"episode={episode}"
    )

    print(
        f"  demo_length={len(demo)}, "
        f"keypoints={keypoints}, "
        f"num_keypoints={len(keypoints)}"
    )

    previous_frame = 0

    for keypoint_id, frame_id in enumerate(keypoints):
        obs = demo._observations[frame_id]
        previous_obs = demo._observations[previous_frame]

        pose = np.asarray(
            obs.gripper_pose,
            dtype=np.float64,
        )

        previous_pose = np.asarray(
            previous_obs.gripper_pose,
            dtype=np.float64,
        )

        delta_position = np.linalg.norm(
            pose[:3] - previous_pose[:3]
        )

        delta_rotation = quaternion_angle_deg(
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

        print(
            f"  keypoint[{keypoint_id}]"
            f" frame={frame_id},"
            f" gap={frame_id - previous_frame},"
            f" terminal={frame_id == len(demo) - 1},"
            f" gripper={float(obs.gripper_open):.1f}"
        )

        print(
            f"    position="
            f"{np.round(pose[:3], 4).tolist()}"
        )

        print(
            f"    delta_from_previous:"
            f" position={delta_position:.4f} m,"
            f" rotation={delta_rotation:.2f} deg"
        )

        print(
            f"    distance_to_final:"
            f" position={distance_to_final:.4f} m,"
            f" rotation={rotation_to_final:.2f} deg"
        )

        previous_frame = frame_id

def get_observation(task_str: str, variation: int,
                    episode: int, env: RLBenchEnv,
                    store_intermediate_actions: bool,debug_keypoints: bool = False):
    demos = env.get_demo(task_str, variation, episode)
    demo = demos[0]

    # key_frame = keypoint_discovery(demo)
    # key_frame.insert(0, 0)

    raw_keyframes = keypoint_discovery(demo)
    if debug_keypoints:
        print_keypoint_debug(
            task_str=task_str,
            variation=variation,
            episode=episode,
            demo=demo,
            keypoints=raw_keyframes,
        )

    # 使用副本，避免修改 raw_keyframes
    key_frame = list(raw_keyframes)
    key_frame.insert(0, 0)

    keyframe_state_ls = []
    keyframe_action_ls = []
    intermediate_action_ls = []

    for i in range(len(key_frame)):
        state, action = env.get_obs_action(demo._observations[key_frame[i]]);
        state = transform(state)
        keyframe_state_ls.append(state.unsqueeze(0))
        keyframe_action_ls.append(action.unsqueeze(0))

        if store_intermediate_actions and i < len(key_frame) - 1:
            intermediate_actions = []
            for j in range(key_frame[i], key_frame[i + 1] + 1):
                _, action = env.get_obs_action(demo._observations[j])
                intermediate_actions.append(action.unsqueeze(0))
            intermediate_action_ls.append(torch.cat(intermediate_actions))

    return demo, keyframe_state_ls, keyframe_action_ls, intermediate_action_ls


class Dataset(torch.utils.data.Dataset):

    def __init__(self, args: Arguments):
        # load RLBench environment
        self.env = RLBenchEnv(
            data_path=args.data_dir,
            image_size=[int(x) for x in args.image_size.split(",")],
            apply_rgb=True,
            apply_pc=True,
            apply_cameras=args.cameras,
        )

        tasks = args.tasks
        variations = range(args.offset, args.max_variations)
        self.items = []
        for task_str, variation in itertools.product(tasks, variations):
            episodes_dir = args.data_dir / task_str / f"variation{variation}" / "episodes"
            episodes = [
                (task_str, variation, int(ep.stem[7:]))
                for ep in episodes_dir.glob("episode*")
            ]
            self.items += episodes

        self.num_items = len(self.items)

    def __len__(self) -> int:
        return self.num_items

    def __getitem__(self, index: int) -> None:
        task, variation, episode = self.items[index]
        taskvar_dir = args.output / f"{task}+{variation}"
        taskvar_dir.mkdir(parents=True, exist_ok=True)

        (demo,
         keyframe_state_ls,
         keyframe_action_ls,
         intermediate_action_ls) = get_observation(
            task, variation, episode, self.env,
            bool(args.store_intermediate_actions), bool(args.debug_keypoints)
        )

        state_ls = einops.rearrange(
            keyframe_state_ls,
            "t 1 (m n ch) h w -> t n m ch h w",
            ch=3,
            n=len(args.cameras),
            m=2,
        )

        frame_ids = list(range(len(state_ls) - 1))
        num_frames = len(frame_ids)
        attn_indices = get_attn_indices_from_demo(task, demo, args.cameras)

        state_dict: List = [[] for _ in range(6)]
        print("Demo {}".format(episode))
        state_dict[0].extend(frame_ids)
        state_dict[1] = state_ls[:-1].numpy()
        state_dict[2].extend(keyframe_action_ls[1:])
        state_dict[3].extend(attn_indices)
        state_dict[4].extend(keyframe_action_ls[:-1])  # gripper pos
        state_dict[5].extend(intermediate_action_ls)   # traj from gripper pos to keyframe action

        with open(taskvar_dir / f"ep{episode}.dat", "wb") as f:
            f.write(blosc.compress(pickle.dumps(state_dict)))


if __name__ == "__main__":
    args = Arguments().parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    dataset = Dataset(args)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.num_workers,
        collate_fn=lambda x: x,
    )

    for _ in tqdm(dataloader):
        continue
