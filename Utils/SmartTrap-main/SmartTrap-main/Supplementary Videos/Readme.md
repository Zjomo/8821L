# Here are the supplementary videos:
The videos showcases the operation of the autonomous system.

## 1 - 3D tracking of the pipette

This video is a screen recording of the camera view in the user interface with real-time tracking visualization turned on. The video shows where the program detects the pipette and the particle. This is done while simultaneously moving the pipette with the particle in and out of focus, estimating the particle focal position.


https://github.com/user-attachments/assets/bb30b676-af74-496c-bb91-06eaeb164970

## 2 - Particle characterization 

The video shows the autonomous particle characterization being performed. The video starts with the instrument moving to a capillary to trap a particle. The first particle trapped is one of the target particles, which is determined by the autonomous algorithm based on it being larger than a certain threshold. Since it is one of the target particles the hydrodynamic radius is measured by moving it between two fixed positions in the sample while recording the forces and motor movement. Once the measurement is finished the microfluidic pump connected to the central chamber is turned on creating a flow which removes the trapped particle. Next, another particle is trapped. By chance this is a small particle, which should not be characterized, therefore it is immediately removed, again using the pump connected to the central chamber. Next, a situation in which multiple particles are trapped is shown. In this case the software detects that the focal position is offset compared to what is expected from a single particle and therefore the particles are removed even though the profile size match those of the target particles. 

Lastly, several experiments are displayed at a high speed with a timer added to illustrate how the system can run for extended times characterizing dozens of particles.

The recording is performed using the interface itself rather than recording the screen itself. This gives slightly higher video quality than a screen recording.

https://github.com/user-attachments/assets/05ffb2d9-0af1-4bef-83d9-a1c86171ede3

## 3 - Autonomous DNA pulling

Video illustrating the autonomous DNA pulling. First, a recording of a single autonomous pulling is shown starting with an empty chamber, without particles in either trap or pipette. This part of the video is commented and is a recorded camera feed showing all the different steps of the autonomous pulling process. Starting with checking the pipette, followed by trapping of the streptadavin particle and positioning of this in the pipette. This is followed by the trapping of a particle with DNA which is then attached to the particle in the pipette where after the experiment measurement is perfromed. After this first pulling, the video shows the full graphical user interface while the program performs a large number of pullings autonomously. This part of the video is shown at increased speed (to limit the duration of the video) and also the real-time plotting is shown. At the end a recording from a 10 hour continuous experiment is shown. This is shown at 600 times normal speed (10 minutes shown in 1 second) to showcase long term operation.

https://github.com/user-attachments/assets/ae2c6801-45a3-41bb-a317-445c46f49120

## 4 - Red blood cells
The videos shows how the red blood cell experiments are performed by the instrument. The video is a screen recording of the GUI and also includes a timer. After a cell has been trapped, initially at low power, cells profile is recorded in a video. Then the trapping power is briefly increased and the cell profile at higher power is recorded in another video. Thereafter the trapping power is reduced and the process repeated four more times at increasingly high powers. Thereafter the cell is removed by a flow. Since there are cells dispersed in the medium the flow is also used to bring new cells into view which are trapped and measured. If by chance there are no cells in view after flowing new medium then the flow is briefly turned on again.

https://github.com/user-attachments/assets/55d9182e-a0de-4d83-9847-974e01d260b4

## 5 - Electrostatic repulsion
Showcase of how electrostatic repulsions can be measured autonomously. When the video starts there is nothing in either trap or pipette. The program first focuses the pipette and checks its content. Next it moves to the capillary, turns on the microfluidic pump and traps a particle. The trapped particle is then placed in the pipette. After confirming that the particle has successfully transferred to the pipette, a second particle is trapped and brought to the pipette. The particle in the trap is aligned to the particle in the pipette. The trapped particle is pushed towards the particle in the pipette by moving the trap to find appropriate limits for the measurement protocol. Next the protocol and recording of data (both force and video) are started. At this stage the program zooms in on the two particles to limit the size of the videos. Once the measurement is completed, a strong flow removes both particles resetting the experiment and preparing the system for another measurement.

https://github.com/user-attachments/assets/04064912-fe21-4e13-9837-17cc39bfbc88

