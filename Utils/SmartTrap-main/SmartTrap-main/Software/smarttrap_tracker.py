"""
Here the tracking classes used in the SmartTrap are implemented.
-------------------------------------------------------
Classes

- ParticleCNN: The convolutional network used to track the focal (z) position of particles.
- ObjectTrackerYOLO: An implementation of the ObjectTracker based on YOLO. Uses also the ParticleCNN
for tracking 3d positions-
"""

import cv2
import torch
import torch.nn as nn
from yolov5 import YOLOv5
from ultralytics import YOLO
import numpy as np
from real_time_tracking import ObjectTracker

class ParticleCNN(nn.Module):
    def __init__(self):
        super(ParticleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1), # Assuming grayscale images
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Size now 64x64
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # Size now 32x32
        )
        # Calculate the size after convolutions and pooling
        # For 128x128 input, after two pooling layers, the size is 32x32
        # And if you have 64 output channels from the last conv layer, then:
        self.size_after_convs = 64 * 32 * 32
        self.fc_layers = nn.Sequential(
            nn.Linear(self.size_after_convs, 512),
            nn.ReLU(),
            nn.Linear(512, 1)  # Predicting a single value
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1) # Flatten the output for the fully connected layer
        x = self.fc_layers(x)
        return x


def load_yolov5_model(model_path):
    import types
    orig_load = torch.load

    def _load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return orig_load(*args, **kwargs)

    torch.load = _load  # monkey-patch
    try:
        return torch.hub.load(".", "custom", path=model_path, source="local", force_reload=True)
    finally:
        torch.load = orig_load
    return model


class ObjectTrackerYOLO(ObjectTracker):

    def __init__(self, YOLO_model_path, z_model_path, particle_size_limits=[20,150]):

        # Load the models
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = load_yolov5_model(YOLO_model_path)
        self.z_model = self.load_z_model(z_model_path)
        #self.z_model = torch.load(z_model_path)
        print("Successfully loaded models")
        print(f"Using device: {self.device}")        
        self.z_model.to(self.device, non_blocking=True)

        self.pipett_location = [0, 0, 0, 0]  # x, y, width, height
        self.pipette_tip_location = [0, 0]  # x, y
        self.confidence_threshold_particle = 0.5 # Deteciton limit for the YOLO model
        self.confidence_threshold_pipette = 0.5 # Detection limit for the pipette(YOLO model)
        self.results = None

        # Particle size limits used to reduce risk of trapping very small or large particles
        # (usually noise)
        self.particle_lower_size_limit = particle_size_limits[0]
        self.particle_upper_size_limit = particle_size_limits[1]
        self.crop_width = int(64)  # Width of the crops for z-detection

    def analyze_frame(self, frame):
        self.results = self.model(frame).xywh[0].cpu().numpy()

    def predict_particle_positions(self):
        particle_positions = []
        radii = []
        for prediction in self.results:
            # Check if particle
            if prediction [-1] == 0 and prediction[4] > self.confidence_threshold_particle:
                if (prediction[2] > self.particle_lower_size_limit and
                    prediction[3] > self.particle_lower_size_limit and
                    prediction[2] < self.particle_upper_size_limit and
                    prediction[3] < self.particle_upper_size_limit):
                    particle_positions.append(prediction[:2])
                    radii.append((prediction[2]+prediction[3])/4)
        return np.array(particle_positions), np.array(radii)

    def predict_pipette_position(self):
        # Check if pipette is present
        pipette_located = False
        for prediction in self.results:
            if prediction[-1] == 1 and prediction[4] > self.confidence_threshold_pipette:
                pipette_located = True
                self.pipett_location[0] = prediction[0]
                self.pipett_location[1] = prediction[1] - prediction[3]/2
                self.pipett_location[2] = prediction[2]
                self.pipett_location[3] = prediction[3]
                self.pipette_view_counter = 0
                self.pipette_tip_location[0] = prediction[0]
                self.pipette_tip_location[1] = prediction[1] - prediction[3]/2

        return self.pipett_location, self.pipette_tip_location, pipette_located

    def pipette_tilt_detection(self, c_p, crop_width=600):
        """
        A second call to the YOLO model to detect the tilt of the pipette.
        Used to get a more accurate location of the pipette tip.
        """
        s = np.shape(c_p['image'])
        # If there is no pipette in the image or we are zoomed in then we should return.
        if not c_p['pipette_located'] or s[0]<crop_width or s[1]<crop_width:
            return
        # exctract a 500x500 pixel area around the pipette tip
        left = int(c_p['pipette_location'][1]-crop_width/2)
        right = int(c_p['pipette_location'][1]+crop_width/2)
        top = int(c_p['pipette_location'][0]-crop_width/2)
        bottom = int(c_p['pipette_location'][0]+crop_width/2)
        if left < 0 or top < 0 or right > s[0] or bottom > s[1]:
            return
        crop = c_p['image'][left:right, top:bottom]
        results = self.model(crop).xywh[0].cpu().numpy()
        for prediction in results:
            if prediction[-1] == 1 and prediction[4] > self.confidence_threshold_pipette:
                new_x = prediction[0] + top
                dx = c_p['pipette_location'][0] - new_x
                dy = c_p['pipette_location'][3]
                c_p['pipette_tilt'] = dx/dy
                c_p['accurate_tip_detection_needed'] = False
                return

    def predict_z_positions(self, frame, positions):
        """
        Function which makes a prediction of the z-positions of the particles located with
        the deep learning model. The z-positions are then stored in the control parameters.
        """

        # Pre-compute constants
        image_shape = np.shape(frame)
        try:
            # Had problem with Index error here when the camera failed to return an image
            image_width, image_height = image_shape[1], image_shape[0]
        except:
            return []
        if image_width < 128 or image_height < 128:            
            return []


        # List to collect crops
        crops = []
        # Loop through predicted positions to collect crops

        for pos in positions:
            x, y = int(pos[0]), int(pos[1])        
            # Resizes each crop to the same size
            x0 = x - self.crop_width
            x1 = x + self.crop_width
            y0 = y - self.crop_width
            y1 = y + self.crop_width

            if 0 >= x0:
                x0 = 0
                x1 = 2*self.crop_width
            if x1 >= image_width:
                x1 = image_width
                x0 = image_width - 2*self.crop_width
            if 0 >= y0:
                y0 = 0
                y1 = 2*self.crop_width
            if y1 >= image_height:
                y1 = image_height
                y0 = image_height - 2*self.crop_width
            try:
                crop = frame[y0:y1,x0:x1].astype(np.float32)
                crop /= 20
                crop = cv2.resize(crop, (128, 128))
                crop = np.reshape(crop, (128, 128, 1)).astype(np.float32)
            except Exception as E:
                # print("Zero prediction added",E)
                crop = np.zeros((128, 128, 1)).astype(np.float32)
            crops.append(np.copy(crop))


        # Convert list of crops to a tensor and prepare for the model
        if crops:  # Check if there are any crops to process
            crops_tensor = torch.tensor(
                crops, dtype=torch.float32).permute(0, 3, 1, 2).to(self.device)

            with torch.no_grad():  # No gradients needed for inference
                predictions = self.z_model(crops_tensor)
            z_vals = predictions.squeeze().tolist()  # Convert predictions to a list            
        else:
            z_vals = []
    
        # Convert to list of z-values
        if isinstance(z_vals, float):
            z_vals = np.array([z_vals])
        if len(z_vals) > 0:
            z_vals = np.array(z_vals)
        return z_vals # Added a factor 2 here to decrease it a bit
    
    def load_2D_model(self, network_name):
        try:
            self.model = load_yolov5_model(network_name)
        except Exception as e:
            print(e)
            print(f"Could not load model {network_name}")
    
    def load_z_model(self, network_name):

        # import types
        orig_load = torch.load

        if self.device.type == 'cpu':
            model = torch.load(network_name, map_location=torch.device("cpu"),weights_only=False)
            return model

        def _load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return orig_load(*args, **kwargs)

        torch.load = _load
        try:
            return torch.load(network_name)
        except Exception as E:
            print("Could not loadz z-model")
            print(E)
            return None
