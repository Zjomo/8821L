from pylablib.devices import Newport
import time


# =========================
# 用户可修改参数
# =========================

LASER_AXIS = 2          # 控制激光开关的轴
LASER_ON_STEPS = 400    # 打开激光运动步数
LASER_OFF_STEPS = 400   # 关闭激光运动步数

DEFAULT_SPEED = 5000
DEFAULT_ACCEL = 5000


def connect_controller():
    """
    连接 Newport 8743-CL / Picomotor 控制器。

    返回：
        stage: Newport.Picomotor8742 对象
    """
    stage = Newport.Picomotor8742()

    print("ID:", stage.get_id())
    print("Axes:", stage.get_all_axes())

    return stage


def set_velocity_accel(stage, axis, speed=5000, accel=5000):
    """
    设置 Newport 8743-CL 指定轴的速度和加速度。

    注意：
    这里不使用 stage.setup_velocity()，
    因为你的 8743-CL 在 pyLabLib 里回读 VA?/AC? 时可能读错。
    因此直接发送 Newport 原生命令。
    """

    axis = int(axis)
    speed = int(speed)
    accel = int(accel)

    cmd_speed = f"VA{speed}"
    print(f"[SET] axis={axis}, cmd={cmd_speed}")
    stage.query(cmd_speed, axis=axis)

    time.sleep(0.05)

    cmd_accel = f"AC{accel}"
    print(f"[SET] axis={axis}, cmd={cmd_accel}")
    stage.query(cmd_accel, axis=axis)

    time.sleep(0.1)

    print(f"[OK] axis={axis} speed={speed}, accel={accel}")


def move_and_wait(stage, axis, steps, wait=True):
    """
    指定轴运动指定步数。

    参数：
        stage: Newport 控制器对象
        axis: 轴号
        steps: 运动步数
        wait: 是否等待运动完成
    """

    axis = int(axis)
    steps = int(steps)

    print(f"[MOVE] axis={axis}, steps={steps}")
    stage.move_by(axis=axis, steps=steps)

    if wait:
        try:
            stage.wait_move(axis=axis)
        except Exception as e:
            print(f"[WARN] wait_move failed: {e}")
            time.sleep(0.2)


def laser_on(stage, axis=LASER_AXIS, steps=LASER_ON_STEPS):
    """
    打开激光。

    默认：
        axis=2
        steps=400
    """

    print("[LASER] ON")
    move_and_wait(stage, axis=axis, steps=steps, wait=True)
    print("[LASER] ON done")


def laser_off(stage, axis=LASER_AXIS, steps=LASER_OFF_STEPS):
    """
    关闭激光。

    默认：
        axis=2
        steps=400
    """

    print("[LASER] OFF")
    move_and_wait(stage, axis=axis, steps=steps, wait=True)
    print("[LASER] OFF done")


def close_controller(stage):
    """
    关闭 Newport 控制器连接。
    """

    if stage is not None:
        stage.close()
        print("[CLOSE] controller closed")


def main():
    """
    单独运行本文件时的测试程序。
    """

    stage = None

    try:
        stage = connect_controller()

        set_velocity_accel(
            stage=stage,
            axis=LASER_AXIS,
            speed=DEFAULT_SPEED,
            accel=DEFAULT_ACCEL,
        )

        # 测试：打开激光
        laser_on(stage)
        time.sleep(1)

        # 测试：关闭激光
        laser_off(stage)
        time.sleep(1)

    finally:
        close_controller(stage)


if __name__ == "__main__":
    main()