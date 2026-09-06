"""临时诊断：台位移动后球画面位移方向/幅度。用完即删。"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from obstacle_avoidance import sim_microscope as sm
from obstacle_avoidance.vision import ClassicDetector

world = sm.SimMicroscopeWorld()
det = ClassicDetector(min_particle_radius_px=8.0)

def dets():
    f = world.render()
    return det.detect_particles(f)[0], f

d0, f0 = dets()
print("t0 dets:", [(tuple(round(v,1) for v in p[0]), round(p[1],1)) for p in d0])
print("t0 stage:", dict(world.micro_stage.position))

stage = world.make_stage()
stage.move_by(0.05, 0.0)          # 命令 +x 0.05mm
print("t1 stage:", dict(world.micro_stage.position))
d1, f1 = dets()
print("t1 dets:", [(tuple(round(v,1) for v in p[0]), round(p[1],1)) for p in d1])
if d0 and d1:
    print("ball0 dx:", round(d1[0][0][0]-d0[0][0][0],1), "dy:",
          round(d1[0][0][1]-d0[0][0][1],1))
world.close()
