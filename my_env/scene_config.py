from pathlib import Path


USD_DIR = Path("/home/huatec/isaac_lab/my_env")

ENV_USD = USD_DIR / "environment.usd"
ROBOT_USD = USD_DIR / "openarmx.usd"

CUBE_PATH = "/World/Green_Physics_Cube"
TRAY_PATH = "/World/Detection_Tray"

ROBOT_PRIM_CANDIDATES = [
    "/openarmx",
    "/Root/openarmx",
]

CAMERA_PATH_CANDIDATES = [
    "/openarmx/openarmx_body_link0/hand_Camera",
    "/openarmx/openarmx_left_hand/left_Camera",
    "/openarmx/openarmx_right_hand/right_Camera",
    "/Root/openarmx/openarmx_body_link0/hand_Camera",
    "/Root/openarmx/openarmx_left_hand/left_Camera",
    "/Root/openarmx/openarmx_right_hand/right_Camera",
]

EE_PRIM_CANDIDATES = [
    "/openarmx/openarmx_right_hand",
    "/openarmx/openarmx_left_hand",
    "/Root/openarmx/openarmx_right_hand",
    "/Root/openarmx/openarmx_left_hand",
]

CUBE_SIZE = 0.04
CUBE_HEIGHT = 0.08
CUBE_MASS = 0.05

TRAY_SIZE_X = 0.14
TRAY_SIZE_Y = 0.21
TRAY_BASE_THICKNESS = 0.004
TRAY_WALL_THICKNESS = 0.003
TRAY_WALL_HEIGHT = 0.04
TRAY_CORNER_RADIUS = 0.0
TRAY_CENTER = [-1.71578, -0.0800, 0.1797]

TRAY_PRIM_CANDIDATES = [
    "/Root/Environment/Detection_Tray",
    "/World/Detection_Tray",
]

RESET_COOLDOWN_STEPS = 60

CAMERA_EYE = [-2.6, -1.07, 0.85]
CAMERA_TARGET = [-1.21, 0.18, 0.18]

# 初始关节角度（每臂 9 关节：joint1~7 + finger_joint1 + finger_joint2）
LEFT_INIT_JOINTS = {
    "openarmx_left_joint1": 0.36884275,
    "openarmx_left_joint2": 0.00019181,
    "openarmx_left_joint3": -0.00019181,
    "openarmx_left_joint4": 1.04745209,
    "openarmx_left_joint5": -0.00287709,
    "openarmx_left_joint6": 0.03394964,
    "openarmx_left_joint7": -0.79350102,
    "openarmx_left_finger_joint1": 0.0,
    "openarmx_left_finger_joint2": 0.0,
}

RIGHT_INIT_JOINTS = {
    "openarmx_right_joint1": -0.31782240,
    "openarmx_right_joint2": -0.02627741,
    "openarmx_right_joint3": -0.02781186,
    "openarmx_right_joint4": 0.97955275,
    "openarmx_right_joint5": 0.02013962,
    "openarmx_right_joint6": -0.00748043,
    "openarmx_right_joint7": 0.80385858,
    "openarmx_right_finger_joint1": 0.0,
    "openarmx_right_finger_joint2": 0.0,
}
