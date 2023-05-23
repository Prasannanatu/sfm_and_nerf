
# Structure from Motion (SfM) and Neural Radiance Fields (NeRF) Project

<!-- ![Project Output](./NeRF/test_gif.gif) -->
<!-- <img src="./NeRF/test_gif.gif" width="500" height="600"> -->
<!-- <img align="left" src="./NeRF/test_gif.gif" width="49%">
<img align="right" src="./sfm_p/Phase1/outputs/Registered camera poses with nonlinear PnP2.png" width="50%"> -->
<!-- <p float="left"> -->
  
<!--   <img src="./NeRF/test_gif.gif" width="400" />
  <img src="./sfm_p/Phase1/outputs/Registered camera poses with nonlinear PnP2.png" width="400" /> 
</p> -->

| ![image1](./NeRF/test_gif.gif) | ![image2](./sfm_p/Phase1/outputs/Registered_camera_poses_with_nonlinear_PnP2.png) |
|:--:|:---:|
| NeRF | SFM |




This repository contains the academic project exploring computer graphics for 3D rendering with Neural Radiance Fields (NeRF) and Structure from Motion (SfM) techniques. The project was conducted as part of the RBE-549 course during the spring semester of 2023. The official university course project page can be found [here](https://rbe549.github.io/spring2023/proj/p2/).


## Table of Contents
- [About The Project](#about-the-project)
- [Repository Structure](#repository-structure)
- [Technologies](#technologies)
- [Installation & Usage](#installation--usage)
- [Contributing](#contributing)


## About The Project
The project was conducted from February to March 2023 and focused on applying SfM and NeRF techniques on a dataset of five images of a glass building. The primary goals of the project were:

Implement feature matching, epipolar geometry, RANSAC, visibility matrix, and bundle adjustment techniques for SfM.
Develop a data loader, parser, network, and loss function for NeRF.
Generate a 3D reconstruction of the scene using the combined SfM and NeRF techniques.

## Repository Structure
The repository is structured as follows:

- `/NeRF`: This folder contains all the source code for the Neural Radiance field, including implementations of NeRF algorithm.
- `/sfm_p`: This folder contains all the source code for the Structure from Motion, including implementations of SFM algorithm.
- `/report`: This folder contains the academic report documenting the project, including the methodology, experimental results, and conclusions.
- `/sfm_p/Data`: This folder contains the data provided on the course project page [here](https://drive.google.com/file/d/1DLdCpX5ojtSN4RjYZ2UwpV2fAJn3sX_k/view)
- For NeRF Please download dataset from the project page[here](https://rbe549.github.io/spring2023/proj/p2/) or tinynerf dataset from [here](https://people.eecs.berkeley.edu/~bmild/nerf/tiny_nerf_data.npz)

## Technologies
The project utilizes the following technologies:

- NeRF: Neural Radiance Fields for volumetric scene representation.
- SfM: Structure from Motion for estimating the 3D structure of the scene.
- Epipolar Geometry & RANSAC: Techniques for accurate estimation of 3D points from multiple views.
- Visibility Matrix: Method for handling occlusions in the scene reconstruction.
- Bundle Adjustment: Technique for refining the estimates of 3D points and camera parameters.

## Installation & Usage
To run the project locally, follow these steps:

1. Clone the repository:

```shell
git clone https://github.com/Prasannanatu/sfm_and_nerf.git
 ```
 
Install the required dependencies. You can use the provided requirements.txt file to install the necessary packages. Run the following command:


```shell
pip install -r requirements.txt
 ```
 
 
Run the project using the provided scripts or commands. Refer to the documentation or project files for specific instructions on running the SfM and NeRF algorithms.

## References

1. Wikipedia. "Eight-point Algorithm." [Link](https://en.wikipedia.org/wiki/Eight-point_algorithm)

2. Hartley, R. and Zisserman, A. "Multiple View Geometry in Computer Vision." Second Edition, Cambridge University Press, 2003. [Link](http://users.cecs.anu.edu.au/~hongdong/new5pt_cameraREady_ver_1.pdf)

3. Mildenhall, B., Srinivasan, P. P., Tancik, M., Barron, J. T., Ramamoorthi, R., Ng, R. "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis." ACM Transactions on Graphics (TOG), Vol. 39, No. 2, Article No. 24, 2020. [Link](https://arxiv.org/abs/2003.08934)

4. RBE-549 Course Project Page. [Link](https://rbe549.github.io/spring2023/proj/p2/)



