"""CLI 入口：内置场景 dryrun 仿真（OA-01..OA-05, AG-01..AG-02）。

安全策略（PLAN 第 4 节）：
- 默认 dryrun；任何真实设备参数会被拒绝（本项目仅含 dryrun 驱动）。
- ``--real-hardware-confirm`` 仅作为错误提示门槛，缺省即拒绝真实硬件。

用法示例：
    python -m obstacle_avoidance --scenario oa01 --report-prefix artifacts/oa01
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from typing import List, Optional, Tuple

from .aggregation import AggregationConfig, AggregationPlanner
from .controller import ControllerConfig, ObstacleAvoidController
from .models import GoalRegion, Obstacle, Point, RunState, SubstrateRegion
from .planner import CollisionModel, GridPlanner
from .reporter import RunReporter, replay, summarize
from .simulator import SimWorld
from .vision import VisionPipeline

FRAME = (640, 480)


# ---------------------------------------------------------------- 场景定义
def _substrate_rect(x0=40.0, y0=40.0, x1=600.0, y1=440.0) -> SubstrateRegion:
    return SubstrateRegion(polygon=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
                           safety_margin_px=6.0)


def _make_world(particles: List[Tuple[Point, float]],
                obstacles: Optional[List[Obstacle]] = None) -> SimWorld:
    sub = _substrate_rect()
    return SimWorld(substrate_polygon=sub.polygon,
                    obstacles=obstacles or [], particles=particles,
                    frame_size=FRAME)


def build_scenario(name: str, sample_spec: Optional[dict] = None,
                   sim_layout: Optional[dict] = None,
                   origin_um: Optional[tuple] = None):
    """返回 (world, run_fn, report_path_hint)。

    sample_spec / sim_layout / origin_um 仅 sim01 有效：前两者为仿真显微镜
    图层配置（掩码/衬底/障碍物）与 UI ROI 划分布局（球/衬底/障碍/目标点），
    origin_um 为布局窗口坐标对应的视窗原点（台位 µm，None=样本中心）。
    """
    r_eff = CollisionModel().ball_radius_px
    if name == "oa01":   # 单球无障碍直线路径
        world = _make_world([( (150, 240), r_eff )])
        goal = GoalRegion(center=(480, 240), radius_px=20)

        def run(world=world, goal=goal, rep=None, stage_sink=None):
            snap = world.snapshot()
            stage = world.make_stage(1)
            ctl = ObstacleAvoidController(stage, VisionPipeline(expected_radius_px=r_eff),
                                          GridPlanner(), ControllerConfig(), rep)
            return ctl.run(snap, 1, goal, task_id=name,
                           get_frame=world.render, get_snapshot=world.snapshot)
        return world, run

    if name == "oa02":   # 单球静态障碍绕行
        obs = [Obstacle(kind="circle", center=(315, 240), radius=45,
                        obstacle_id="obs-static")]
        world = _make_world([( (150, 240), r_eff )], obs)
        goal = GoalRegion(center=(480, 240), radius_px=20)

        def run(world=world, goal=goal, rep=None, stage_sink=None):
            snap = world.snapshot()
            stage = world.make_stage(1)
            ctl = ObstacleAvoidController(stage, VisionPipeline(expected_radius_px=r_eff),
                                          GridPlanner(), ControllerConfig(), rep)
            return ctl.run(snap, 1, goal, task_id=name,
                           get_frame=world.render, get_snapshot=world.snapshot)
        return world, run

    if name == "oa03":   # 障碍墙封闭目标 -> NO_SAFE_PATH
        obs = [Obstacle(kind="circle", center=(440, 240), radius=60,
                        obstacle_id="obs-cap-A"),
               Obstacle(kind="circle", center=(430, 185), radius=60,
                        obstacle_id="obs-cap-B"),
               Obstacle(kind="circle", center=(430, 295), radius=60,
                        obstacle_id="obs-cap-C"),
               Obstacle(kind="circle", center=(430, 130), radius=60,
                        obstacle_id="obs-cap-D"),
               Obstacle(kind="circle", center=(430, 350), radius=60,
                        obstacle_id="obs-cap-E")]
        world = _make_world([( (150, 240), r_eff )], obs)
        goal = GoalRegion(center=(540, 240), radius_px=15)

        def run(world=world, goal=goal, rep=None, stage_sink=None):
            snap = world.snapshot()
            stage = world.make_stage(1)
            ctl = ObstacleAvoidController(stage, VisionPipeline(expected_radius_px=r_eff),
                                          GridPlanner(), ControllerConfig(), rep)
            return ctl.run(snap, 1, goal, task_id=name,
                           get_frame=world.render, get_snapshot=world.snapshot)
        return world, run

    if name == "oa04":   # 终点在衬底外 -> 拒绝
        world = _make_world([( (150, 240), r_eff )])
        goal = GoalRegion(center=(650, 240), radius_px=20)  # 越出画面/衬底

        def run(world=world, goal=goal, rep=None, stage_sink=None):
            snap = world.snapshot()
            stage = world.make_stage(1)
            ctl = ObstacleAvoidController(stage, VisionPipeline(expected_radius_px=r_eff),
                                          GridPlanner(), ControllerConfig(), rep)
            return ctl.run(snap, 1, goal, task_id=name,
                           get_frame=world.render, get_snapshot=world.snapshot)
        return world, run

    if name == "oa05":   # 障碍突现 -> 安全停止并重规划
        world = _make_world([( (150, 240), r_eff )])
        goal = GoalRegion(center=(480, 240), radius_px=20)
        state = {"armed": False}

        def hook(frame_id: int) -> None:
            if frame_id == 8 and not state["armed"]:
                state["armed"] = True
                world.add_obstacle(Obstacle(kind="circle", center=(410, 240),
                                            radius=45, obstacle_id="obs-dyn"))

        world.on_frame(hook)

        def run(world=world, goal=goal, rep=None, stage_sink=None):
            snap = world.snapshot()
            stage = world.make_stage(1)
            ctl = ObstacleAvoidController(stage, VisionPipeline(expected_radius_px=r_eff),
                                          GridPlanner(), ControllerConfig(), rep)
            return ctl.run(snap, 1, goal, task_id=name,
                           get_frame=world.render, get_snapshot=world.snapshot)
        return world, run

    if name == "ag01":   # 单球聚拢
        world = _make_world([( (150, 240), r_eff )])
        region = GoalRegion(center=(470, 240), radius_px=60)

        def run(world=world, region=region, rep=None, stage_sink=None):
            snap = world.snapshot()
            ag = AggregationPlanner(GridPlanner(),
                                    VisionPipeline(expected_radius_px=r_eff),
                                    AggregationConfig(required_count=1), rep)
            return ag.run(world, snap, region, task_id=name)
        return world, run

    if name == "ag02":   # 多球聚拢（含交叉路径）
        world = _make_world([((120, 120), r_eff), ((520, 360), r_eff),
                             ((120, 360), r_eff), ((520, 120), r_eff)])
        region = GoalRegion(center=(320, 240), radius_px=90)

        def run(world=world, region=region, rep=None, stage_sink=None):
            snap = world.snapshot()
            ag = AggregationPlanner(GridPlanner(),
                                    VisionPipeline(expected_radius_px=r_eff),
                                    AggregationConfig(required_count=4), rep)
            return ag.run(world, snap, region, task_id=name)
        return world, run

    if name in ("video01", "video02", "video03"):  # 真实视频 + YOLO 后端闭环
        from . import video_sim
        return video_sim.build_video_scenario(name)
    if name == "sim01":   # 显微镜仿真：microscope-master 相机+位移台闭环
        from . import sim_microscope
        return sim_microscope.build_sim_scenario(sample_spec=sample_spec,
                                                 layout=sim_layout,
                                                 origin_um=origin_um)
    raise SystemExit(f"unknown scenario: {name}")


SCENARIOS = ["oa01", "oa02", "oa03", "oa04", "oa05", "ag01", "ag02",
             "video01", "video02", "video03", "sim01"]


def _load_sample_spec(text: Optional[str]) -> Optional[dict]:
    """解析 --sample-spec：内联 JSON 字符串或 .json 文件路径。"""
    if not text:
        return None
    import json
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        if os.path.isfile(text):
            with open(text, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            raise SystemExit(f"[FAIL] --sample-spec 不是合法 JSON，"
                             f"也不是存在的文件: {text}")
    if not isinstance(data, dict):
        raise SystemExit("[FAIL] --sample-spec 必须是 JSON 对象")
    return data


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="obstacle_avoidance",
                                 description="避障/聚拢 CLI（虚拟/电机双模式）")
    ap.add_argument("--scenario", required=True, choices=SCENARIOS)
    ap.add_argument("--report-prefix", default=None,
                    help="JSONL 报告前缀，如 artifacts/oa01")
    ap.add_argument("--real-hardware-confirm", action="store_true",
                    help="电机模式硬件确认门控；虚拟模式忽略")
    ap.add_argument("--xy-driver", default="dryrun")
    ap.add_argument("--z-driver", default="dryrun")
    # ---- 电机模式参数
    ap.add_argument("--mode", choices=["virtual", "motor"], default="virtual",
                    help="virtual=仿真闭环；motor=真实相机+真实串口位移台")
    ap.add_argument("--camera-index", type=int, default=0,
                    help="电机模式真实相机索引")
    ap.add_argument("--serial-port", default=None, help="电机模式串口，如 COM3")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--cmd-template", default="G91 G1 X{dx:.4f} Y{dy:.4f}\n",
                    help="串口指令模板，{dx}/{dy} 为毫米增量")
    ap.add_argument("--roi-config", default=None,
                    help="video03 区域配置路径（默认 artifacts/roi_config.json）")
    ap.add_argument("--sample-spec", default=None,
                    help=("sim01 仿真显微镜图层配置：JSON 字符串或 .json 文件路径。"
                          "键 ground/mask/obstacle（衬底/掩码/障碍物），每类字段: "
                          "count 数量, shape 形状(ellipse/blob/rect/triangle/star/"
                          "custom), label 类别标签, size 基准尺寸px, "
                          "custom 自定义多边形顶点(\"x,y x,y ...\")。"
                          "未指定的类别/字段沿用当前配置"))
    args = ap.parse_args(argv)

    if args.xy_driver != "dryrun" or args.z_driver != "dryrun":
        print("[SAFETY] z 轴驱动仅支持 dryrun。", flush=True)
        return 2
    if args.mode == "motor" and args.scenario != "video03":
        print("[SAFETY] motor 模式仅支持 video03（需 ROI/区域配置）。", flush=True)
        return 2

    prefix = args.report_prefix
    os.makedirs(os.path.dirname(prefix), exist_ok=True) if prefix and \
        os.path.dirname(prefix) else None
    report_path = f"{prefix}.jsonl" if prefix else None
    if report_path and os.path.exists(report_path):
        os.remove(report_path)  # 每次CLI运行生成独立报告（replay可复现单次任务）
    rep = RunReporter(report_path)
    rep.log("run_start", scenario=args.scenario, dryrun=(args.mode == "virtual"),
            mode=args.mode, algorithm="Alg1", task_mode=args.scenario,
            controller="8742/8743-sim" if args.mode == "virtual" else args.xy_driver)
    sample_spec = _load_sample_spec(args.sample_spec)
    if sample_spec is not None:
        rep.log("sample_spec", spec=sample_spec)

    if args.mode == "motor" and args.scenario == "video03":
        from . import video_sim
        from .roi_zones import RoiConfig
        if not args.serial_port:
            print("[SAFETY] motor 模式需要 --serial-port。", flush=True)
            return 2
        config = (RoiConfig.load(args.roi_config) if args.roi_config else None)
        motor = {"port": args.serial_port, "baudrate": args.baud,
                 "cmd_template": args.cmd_template,
                 "camera_index": args.camera_index,
                 "confirmed": bool(args.real_hardware_confirm)}
        world, run_fn = video_sim.build_video_scenario(
            "video03", config=config, motor=motor)
    else:
        world, run_fn = build_scenario(args.scenario, sample_spec=sample_spec)
    try:
        result = run_fn(rep=rep)
    except Exception as exc:  # CLI 层兜底，报告后向上传递退出码
        rep.log("error", reason=f"{type(exc).__name__}: {exc}")
        rep.log("run_end", final_state="FAULT")
        rep.finalize_experiment()
        rep.close()
        print(f"[FAIL] {type(exc).__name__}: {exc}", flush=True)
        return 1
    rep.finalize_experiment()
    rep.close()

    ok = result.final_state in (RunState.COMPLETE,) or getattr(
        result, "completed", False)
    print(f"scenario={args.scenario} state={result.final_state.value} "
          f"reason={getattr(result.failure_reason, 'value', None)} "
          f"detail={getattr(result, 'detail', '')}")
    if report_path:
        s = summarize(replay(report_path))
        print(f"report={report_path} events={s['event_count']} "
              f"commands={s['stage_command_count']}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
