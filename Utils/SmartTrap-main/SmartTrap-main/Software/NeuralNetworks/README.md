# Contains the neural networks used in the SmartTrap system.
Place this folder in the same folder as the main.py for the system to automatically find the networks. The folder should be called NeuralNetworks.

## Retraining the YOLO network
If your system, or particles, are very different from the original SmartTrap you may need to re-train the object detection network to achieve accurate tracking.

### Collecting training data
To re-train the system you need training data for the network. Depending on the complexity of the task at hand you may need more or less data. Our network was trained on ca 1000 manually annotated images and this can be seen as a good starting point.
You have two major options for obtaining the data.
- Collecting real-world data. Save images from operating your system and manually label (mark position of your objectes). You can for instance use the website [roboflow](<https://roboflow.com/annotate>) for annotating the data. From roboflow you can also easily obtain the correct format of the training images and their respective targets.
- Simulating the data. By simulating the data you can easily obtain large amounts of data but ensuring that it is sufficiently similar to imaging of your system. If you choose to simulate the training data we recommend using the [DeepTrack package](<https://github.com/DeepTrackAI/DeepTrack2>)

### Training the network
The SmartTrap uses [YOLOV5s](<https://docs.ultralytics.com/models/yolov5/>). There are newer YOLO models which are slightly more accurate and which can make do with less training data but these are also somewhat slower. Still if you choose to use a newer version the changes needed in the code to make them work are minimal.
To install yolov5 follow their instructions on the [GitHub](<https://github.com/ultralytics/yolov5>)

With the training data collected and 
Open a command prompt and navigate to the installation folder. Run the command
 - yolo train model=yolov5n.pt data=TrainingYOLO.yaml epochs=100 imgsz=640

Where the TrainingYOLO.yaml is the file which specifies where the training data is placed and which classes to use. An example of this file is uploaded to this GitHub.
## Focal position prediction network
The file with weights for the focal position network is too large (130 Mb) to host in github. Instead you can download them from our OneDrive
[here](<https://gunet-my.sharepoint.com/:u:/g/personal/martin_selin_gu_se/EdVVgLlzXXJIrPBgRyKFyUkBOrFAUpYH9q3BJUx0UXnAvg?e=Lmdhxm>)
After downloading the weights file put it in the NeuralNetworks folder.
