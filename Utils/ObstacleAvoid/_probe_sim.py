"""临时诊断：sim01 首帧检测。用完即删。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2
import numpy as np
from obstacle_avoidance import sim_microscope as sm
from obstacle_avoidance.vision import ClassicDetector

world = sm.SimMicroscopeWorld()
frame = world.render()
cv2.imwrite("artifacts/_sim_frame.png", frame)
print("frame:", frame.shape, "min/max:", frame.min(), frame.max())
print("ball windows:", [world._to_window((x, y)) for x, y, r in world._balls])

det = ClassicDetector(min_particle_radius_px=8.0)
sub = det.detect_substrate(frame)
print("substrate:", sub.polygon if sub else None)
dets, dbg = det.detect_particles(frame)
print("particles:", dets)
print("dbg keys:", list(dbg.keys()) if isinstance(dbg, dict) else type(dbg))
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
bx, by = world._to_window((1420.0, 1150.0))
patch = gray[int(by)-30:int(by)+30, int(bx)-30:int(bx)+30]
print("ball patch min/max/mean:", patch.min(), patch.max(), round(float(patch.mean()),1))
world.close()
