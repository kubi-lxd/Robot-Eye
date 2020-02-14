# Robot-Eye
Monocular visual information analysis with a movable camera installed in the eyeball of humanoid robot

## Code Folder

This folder contains programs to do image processing, image annotation, and to control the robot's head and eye movements.

### ImageProcessing

To be added...

### MarkPoints_matlab

The MarkPoints_matlab folder contains matlab programs for marking real-world coordinates of key points in pictures. Some example figures have been put in the path ***\figures\examples\***, run **\code\MarkPoints_matlab\mark.m** and the image annotation program will start. The figure to be marked will show on the screen. Click the point you want to mark, then enter the real-world coordinates, and then click OK to complete the marking of a picture. After all figures are marked, a txt format file which stores the image annotation data will be created.

### RobotControl

This folder contains ROS programs to control humanoid robot's head and eyeball movements. After each movement of the robot, this program will also control the fisheye camera in the eyeball to take pictures to obtain the bitmap information in the scene.

## File structure

```
*---\  <== Root path
|------code  <=== all program source codes
|------|-----ImageProcessing
|------|-----MarkPoints_matlab
|------|-----RobotControl
|------data  <=== Annotation data file obtained after all images are annotated
|------|-----mark.txt <=== A small annotation data file which only contains example figures data.
|------|-----total  <=== Complete annotation data file. *** Important ***
|------figures  <===This folder stores figures to be processed
|------|-----examples <===This folder stores example figures to show algorithmic effects or quickly check programs correctness
|------|-----figures <===This folder stores all figures, hundreds of robot vision pictures should be placed here. *** Important ***
```

