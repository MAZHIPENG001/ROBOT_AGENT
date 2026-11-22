# dexos-lite robot config --robot-id cobot_magic --path ./config/policy_cfg/maas/cobot_magic_config.py
# dexos-lite robot config --robot-id mirror --path ./config/policy_cfg/maas/mirror_config.py
# dexos-lite robot config --robot-id arx5 --path ./config/policy_cfg/maas/arx5_config.py
# dexos-lite robot config --robot-id franka --path ./config/policy_cfg/maas/franka_config.py
from dexos_lite.robot import CobotMagicRobot


robot = CobotMagicRobot(
    sensors={
        "cam_high": {"serial_number": ""},
        "cam_left_wrist": {"serial_number": ""},
        "cam_right_wrist": {"serial_number": ""},
    },
)
