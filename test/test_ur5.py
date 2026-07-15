from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.tasks import ReachTarget

obs_config = ObservationConfig()
obs_config.set_all(True)
obs_config.gripper_touch_forces = False

action_mode = MoveArmThenGripper(
    arm_action_mode=EndEffectorPoseViaPlanning(
        collision_checking=False
    ),
    gripper_action_mode=Discrete(),
)

env = Environment(
    action_mode=action_mode,
    obs_config=obs_config,
    headless=False,
    robot_setup="ur5",
)

try:
    env.launch()
    task = env.get_task(ReachTarget)

    descriptions, obs = task.reset()

    print("Descriptions:", descriptions)
    print("UR5 joints:", obs.joint_positions)
    print("TCP pose:", obs.gripper_pose)
    print("Gripper open:", obs.gripper_open)

    demo = task.get_demos(
        amount=1,
        live_demos=True,
    )[0]

    print("Demo length:", len(demo))
    print("UR5 demo generated successfully.")

finally:
    env.shutdown()