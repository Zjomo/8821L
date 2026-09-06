"""临时诊断：复现 sim01 第二次规划失败。用完即删。"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from obstacle_avoidance import sim_microscope as sm
from obstacle_avoidance.vision import ClassicDetector, VisionPipeline, ParticleTracker
from obstacle_avoidance.video_sim import _initial_detect
from obstacle_avoidance.planner import GridPlanner
from obstacle_avoidance.models import GoalRegion

world = sm.SimMicroscopeWorld()
det = ClassicDetector(min_particle_radius_px=8.0)
pipeline = VisionPipeline(det, tracker=ParticleTracker(max_jump_px=220.0))
snap, tid = _initial_detect(world, det, sm.TARGET_HINT, pipeline=pipeline)
planner = GridPlanner()
goal = GoalRegion(center=(620.0, 450.0), radius_px=25)

plan1 = planner.plan(snap, snap.particles[0].position_px, goal)
print("plan1:", plan1.success, getattr(plan1, 'failure_reason', None))
wp1 = plan1.waypoints if hasattr(plan1, 'waypoints') else plan1.waypoints_px
print("plan1 waypoints:", [(round(x),round(y)) for x,y in (wp1 or [])])

# 执行第一个 waypoint 的第一步（controller 逻辑：朝 wp1[1] 走 max_step）
target = wp1[1]
seg = math.dist(snap.particles[0].position_px, target)
step_mm = 0.05
dx_px = (target[0]-snap.particles[0].position_px[0])/seg*100  # 100px=0.05mm
dy_px = (target[1]-snap.particles[0].position_px[1])/seg*100
stage = world.make_stage()
stage.move_by(dx_px/2000.0, dy_px/2000.0)
print("moved px:", round(dx_px,1), round(dy_px,1))

# 重新感知
frame = world.render()
pipeline.process(frame, 2)
snap2 = world.snapshot()
print("particles2:", [(p.track_id, tuple(round(v,1) for v in p.position_px)) for p in snap2.particles])
print("obstacles2:", [(o.obstacle_id, tuple(round(v,1) for v in o.center), round(o.radius,1)) for o in snap2.obstacles])
print("check goal2:", planner.check_point(snap2, (620.0, 450.0)))
plan2 = planner.plan(snap2, snap2.particles[0].position_px, goal)
print("plan2:", plan2.success, getattr(plan2, 'failure_reason', None),
      getattr(plan2, 'detail', None))
world.close()
