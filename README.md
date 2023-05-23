
# Structure from Motion (SfM) and Neural Radiance Fields (NeRF) Project

<!-- ![Project Output](./NeRF/test_gif.gif) -->
<!-- <img src="./NeRF/test_gif.gif" width="500" height="600"> -->
<img align="left" src="./NeRF/test_gif.gif" width="50%">
<img align="right" src="./sfm_p/Phase1/outputs/Registered camera poses with nonlinear PnP2.png" width="50%">
<!-- <p float="left"> -->
  
<!--   <img src="./NeRF/test_gif.gif" width="400" />
  <img src="./sfm_p/Phase1/outputs/Registered camera poses with nonlinear PnP2.png" width="400" /> 
</p> -->


This repository hosts the academic project exploring computer graphics for 3D rendering with Neural Radiance Fields (NeRF) and Structure from Motion (SfM) techniques.

#Table of Contents
About The Project
Repository Structure
Technologies
Installation & Usage
Contributing


## About The Project
The project was conducted from February to March 2023 and focused on applying SfM and NeRF techniques on a dataset of five images of a glass building. It involves implementing feature matching, epipolar geometry, RANSAC, visibility matrix, and bundle adjustment techniques for SfM. Moreover, a data loader, parser, network, and loss function for NeRF were developed. This repository contains both the final report detailing the findings of the project and the code used for implementation.

### Repository Structure
/src: This folder contains all the source code for the project.
/report: This folder contains the academic report for this project.
/data: This folder contains the image dataset used for the project.


#### Technologies
NeRF: For volumetric scene representation.
SfM: For estimating the 3D structure of the scene.
Epipolar Geometry & RANSAC: For accurate estimation of 3D points.
Visibility Matrix: For handling occlusions.
Bundle Adjustment: For refining the 3D point estimates and camera parameters.
