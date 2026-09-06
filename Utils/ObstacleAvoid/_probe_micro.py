"""临时探测：microscope-master 依赖链 + SampleAwareCamera 出帧。用完即删。"""
import os
import sys

ROOT = os.path.join("API", "microscope-master", "microscope-master")
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), ROOT)
APP = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "API", "microscope-master", "simulator_app")
sys.path.insert(0, REPO)
sys.path.insert(0, APP)

import numpy as np
import microscope  # noqa: E402
import microscope.simulators  # noqa: E402
from db import Database  # noqa: E402
from devices import SampleAwareCamera, SQLiteStage  # noqa: E402

print("microscope import OK:", microscope.__file__)

db = Database(os.path.join("artifacts", "_probe_sim.db"))
stage = SQLiteStage(db, {"x": (-1000.0, 1000.0), "y": (-1000.0, 1000.0),
                         "z": (-50.0, 50.0)})
cam = SampleAwareCamera(stage, db)
cam.enable()
print("acquiring:", cam._acquiring, "sensor:", cam._sensor_shape)

cam.trigger()
frame = cam._fetch_data()
print("frame:", None if frame is None else (frame.shape, frame.dtype))

# 台位移动 -> 视野平移
p0 = dict(stage.position)
stage.move_by({"x": 100.0})          # +100µm
cam.trigger()
frame2 = cam._fetch_data()
diff = np.mean(np.abs(frame2.astype(np.int16) - frame.astype(np.int16)))
print("pos x:", p0["x"], "->", stage.position["x"], "mean|diff|:", round(float(diff), 2))
cam.disable()
print("OK")
