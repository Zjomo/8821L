# Object-Angle-Deviation
Python script to measure deviation from window center to object center and angle

# Outputs
![](/Outputs/output.png)
![](/Outputs/output2.png)
![](/Outputs/output3.png)
![](/Outputs/output4.png)
![](/Outputs/output5.png)
![](/Outputs/output6.png)

# Requirements 
OpenCV4\
Python3\
Numpy\
Scipy

# Usage
1. Measure angle and deviation of image from center : 
**python center_align.py --image path_to_image**

2. Measure angle and deviation in video and returned aligned object frame based on allowed threshold :  
**python realtime_align.py --video path_to_video --angle angle_threshold --deviation deviation_threshold**
