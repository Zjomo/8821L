"""Alg2 交互逐球搬运：目标驻点环形紧贴（需求"情况1 多球"）。

覆盖 `WorkerThread._alg2_ring_seat`：
  ① 第 0 个球正压目标点中心（未就位球时返回中心）；
  ② 已就位 1 个球时，新球落在环绕半径 2r 的环上（球心距 ≈ 2r，相切紧贴）；
  ③ 已就位多个球时各驻点按 6 方位环错开，球心距仍 ≈ 2r，互不重叠；
  ④ 只对给定 ground 的球组生效（不误用其它 ground 球的中心计数）。
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session")
def qapp():
    from obstacle_avoidance.qt_compat import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def _wt(qapp) -> "object":
    from obstacle_avoidance.app import WorkerThread
    return WorkerThread("sim01", {}, parent=None)  # 不 start()，仅作方法宿主


# 球半径 10（rect 20x20 -> r=(20+20)/4=10），中心距应为 2r=20
# ---------------------------------------------------------------- ① 未就位
def test_ring_seat_first_ball_presses_center(qapp):
    wt = _wt(qapp)
    goal = (500.0, 300.0)
    balls = [[0, 0, 20, 20], [1000, 1000, 20, 20]]
    seat = wt._alg2_ring_seat([0, 1], balls, goal)
    assert seat == (500.0, 300.0), "第 0 个球应正压目标点中心"


# ---------------------------------------------------------------- ② 一个已就位
def test_ring_seat_single_existing_touches_center(qapp):
    wt = _wt(qapp)
    goal = (500.0, 300.0)
    # 球0 已压中心，球1 还远
    balls = [[490, 290, 20, 20], [1000, 1000, 20, 20]]
    seat = wt._alg2_ring_seat([0, 1], balls, goal)
    d = ((seat[0] - goal[0]) ** 2 + (seat[1] - goal[1]) ** 2) ** 0.5
    assert abs(d - 20.0) < 1e-6, f"新球应绕中心紧贴(距2r), got d={d:.2f}"


# ---------------------------------------------------------------- ③ 多个已就位互不重叠
def test_ring_seat_multi_existing_spreads_no_overlap(qapp):
    wt = _wt(qapp)
    goal = (500.0, 300.0)
    seats = [goal]
    # 依次加入第2、第3个球，验证与已就位球中心距 >= 2r（不重叠）
    for n in range(1, 8):   # 第2..第8 球
        balls = [[490, 290, 20, 20]]   # 球0 压中心
        # 模拟后续球已就位（第1..第n-1 个已放到驻点）
        for k in range(1, n):
            slot = (k - 1) % 6
            ring = (k - 1) // 6
            ang = 2 * 3.141592653589793 * slot / 6.0
            rr = 20.0 * (1 + ring)
            balls.append([500 + rr * __import__("math").cos(ang) - 10,
                          300 + rr * __import__("math").sin(ang) - 10,
                          20, 20])
        balls.append([2000, 2000, 20, 20])
        group = list(range(len(balls)))
        seat = wt._alg2_ring_seat(group, balls, goal)
        # 新驻点与所有已就位球中心距都应 >= 2r（不重叠）
        for i in range(1, len(balls) - 1):
            cx = balls[i][0] + balls[i][2] / 2
            cy = balls[i][1] + balls[i][3] / 2
            d = ((seat[0] - cx) ** 2 + (seat[1] - cy) ** 2) ** 0.5
            assert d >= 20.0 - 1e-3, \
                f"第{n+1}球驻点与球{i}中心距{d:.2f} < 2r(20)，重叠"


# ---------------------------------------------------------------- ④ 只对给定球组生效
def test_ring_seat_uses_given_group_only(qapp):
    wt = _wt(qapp)
    goal = (500.0, 300.0)
    # 球0 压中心但不在给定组内；组内只有球1（远处）-> 应视为无已就位
    balls = [[490, 290, 20, 20], [1000, 1000, 20, 20]]
    seat = wt._alg2_ring_seat([1], balls, goal)
    assert seat == (500.0, 300.0), "不属本组的球不应参与已就位计数"