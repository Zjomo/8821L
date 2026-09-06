"""临时诊断：sim01 规划端点检查。用完即删。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from obstacle_avoidance import sim_microscope as sm
from obstacle_avoidance.vision import ClassicDetector, VisionPipeline, ParticleTracker
from obstacle_avoidance.video_sim import _initial_detect
from obstacle_avoidance.planner import GridPlanner, CollisionModel
from obstacle_avoidance.models import GoalRegion

world = sm.SimMicroscopeWorld()
det = ClassicDetector(min_particle_radius_px=8.0)
pipeline = VisionPipeline(det, tracker=ParticleTracker(max_jump_px=220.0))
snap, tid = _initial_detect(world, det, sm.TARGET_HINT, pipeline=pipeline)
print("target:", tid, "particles:", [(p.track_id, tuple(round(v,1) for v in p.position_px)) for p in snap.particles])
print("obstacles:", [(o.obstacle_id, tuple(round(v,1) for v in o.center), o.radius) for o in snap.obstacles])
planner = GridPlanner()
print("inflation:", planner.config.model.inflation_px,
      "edge:", planner.config.edge_clearance)
print("check start:", planner.check_point(snap, (320.0, 450.0)))
print("check goal:", planner.check_point(snap, (620.0, 450.0)))
plan = planner.plan_path(snap, (320.0, 450.0),
                         GoalRegion(center=(620.0, 450.0), radius_px=25))
print("plan:", plan.success, plan.failure_reason, plan.detail if hasattr(plan, 'detail') else '')
world.close()
