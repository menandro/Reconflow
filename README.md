# Reconflow

Implementation of our paper presented in WACV2018.

M. Roxas and T. Oishi, , "Real-Time Simultaneous 3D Reconstruction and Optical Flow Estimation," in Proc. IEEE Winter Conf. on Applications of Computer Vision (WACV 2018), Mar. 2018.
http://www.cvl.iis.u-tokyo.ac.jp/index.php?id=mr#Opticalflow

## Requirements

1. OpenCV, OpenCV Contrib (optflow, viz) (tested with v3.2.0)
2. CUDA 8.0 (Including Samples for headers)
3. Visual Studio 2015

## Building Instructions
The solution consists of two projects - reconflow and test_reconflow. reconflow generates a static library from which test_reconflow links. test_reconflow generates a Win32 .exe file. 

There is a lib_link.h header (for both project) that links the necessary libraries. Modify the directories:

```
#define LIB_PATH "D:/dev/lib64/"
#define CV_LIB_PATH "D:/dev/lib64/"
#define CUDA_LIB_PATH "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v8.0/lib/x64/"
```

to point to the location of OpenCV (CV_LIB_PATH) and CUDA (CUDA_LIB_PATH) .lib files.

At the same time, modify the Project Properties -> VC++ Directories -> (Executables, Includes, and Libraries) to point to the location of the OpenCV and CUDA builds, too.

```
Executable: D:/dev/bin
Includes: D:/dev/include
Libraries: D:/dev/lib64
```

### To do
*CMake

## License
This project is licensed under the MIT license

## Author
Menandro Roxas, 3D Vision Laboratory (Oishi Laboratory), Institute of Industrial Science, The University of Tokyo


