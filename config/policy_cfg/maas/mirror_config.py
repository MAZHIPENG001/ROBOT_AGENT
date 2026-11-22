from dexos_lite.robot import MirrorRobot


robot = MirrorRobot(
    # Start the Mirror arm service before running inference.
    sensors={
        "cam_high": {"serial_number": ""},
        "cam_left_wrist": {"serial_number": ""},
        "cam_right_wrist": {"serial_number": ""},
    },
)
