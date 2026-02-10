# C++ Video Parallel Processing (CPU & GPU)

## 🔧 Tech Stack

<p align="left">
  <a href="https://isocpp.org/" target="_blank"><img src="https://img.shields.io/badge/C++-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white" alt="C++"/></a>
  <a href="https://www.khronos.org/opencl/" target="_blank"><img src="https://img.shields.io/badge/OpenCL-F0DB4F?style=for-the-badge&logo=khronos&logoColor=black" alt="OpenCL"/></a>
  <a href="https://opencv.org/" target="_blank"><img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV"/></a>
  <a href="https://cmake.org/" target="_blank"><img src="https://img.shields.io/badge/CMake-064F8C?style=for-the-badge&logo=cmake&logoColor=white" alt="CMake"/></a>
  <a href="https://ffmpeg.org/" target="_blank"><img src="https://img.shields.io/badge/FFmpeg-FFFFFF?style=for-the-badge&logo=ffmpeg&logoColor=black" alt="FFmpeg"/></a>
</p>

## Description

This project demonstrates video processing using Gaussian Blur implemented in three different ways:
- Serial CPU processing
- Parallel CPU processing using std::thread and std::future
- GPU processing using OpenCL

Face detection is applied using Haar Cascade, and detected faces are excluded from blurring.

 ---

### 📝 Test video files

For testing the application, I generated two video files using **FFmpeg**:

- **input5s.mp4** – 5 seconds, 640x480, 30 FPS  
- **input30s.mp4** – 30 seconds, 640x480, 30 FPS

  ---
  
## Features

- Separable Gaussian Blur (horizontal + vertical pass)
- CPU parallelization using std::thread
- GPU acceleration using OpenCL
- Performance comparison between serial, CPU-parallel, and GPU-parallel approaches
- Face detection with selective blur exclusion
- Execution time measurement

  ---

## Technologies

- C++17
- OpenCV
- std::thread / std::future
- OpenCL
- CMake

  ---

## 🧠 Face Detection  

For face recognition, the Haar Cascade Classifier (`haarcascade_frontalface_default.xml`) from OpenCV was used.  
Detection is performed every 5th frame (interval detection) for performance optimization.

---

## ⚙️ System & Hardware Requirements  
To properly run the application with GPU parallel processing using OpenCL, the following requirements must be met:

**Graphics Card with OpenCL Support**  
- GPU must support OpenCL.  
- Latest GPU drivers including OpenCL runtime must be installed.  
- Availability of OpenCL devices is checked by the application at startup (it will display the number and type of OpenCL devices).  
- OpenCL must also be included in the CMake project, with proper include and lib directories.

**OpenCV Library**  
- OpenCV must be installed and compiled with OpenCL support (`WITH_OPENCL=ON`) for GPU function execution.  
- Recommended OpenCV version: >=4.5.

**C++17 Compatible Compiler**  
- GCC >= 9, MSVC 2019, or similar, with support for `std::filesystem`, `std::thread`, `std::future`.

**Optional: FFmpeg**  
- If you want to generate test videos (`input5s.mp4`, `input30s.mp4`) or convert video formats, FFmpeg must be installed and available from the command line.

  ---

## 🖥️ Test Device  

| Component           | Specification                     |
|--------------------|----------------------------------|
| Operating System    | Microsoft Windows 11 Pro         |
| CPU                 | AMD Ryzen 5 5500, 6 cores / 12 threads |
| RAM                 | 16 GB                            |
| GPU                 | AMD Radeon RX 6650 XT (OpenCL support) |

---

## 📹 Test Video  
- File: `inputFACE.mp4`  
- Duration: 26 seconds  
- Resolution: 852x480  
- FPS: 30  
- Note: For parallel GPU face detection, the CPU parallel detection method was used.  
- Gaussian Blur Kernel: K-15

  ---

## ⏱️ Processing Results  

| Method                        | Face Detection [ms] | Gaussian Blur [ms] | Total Time [ms] |
|--------------------------------|-------------------|------------------|----------------|
| Sequential (CPU) (K-15)        | 4,133             | 34,822           | 38,955         |
| Parallel (CPU) (K-15)          | 1,743             | 7,145            | 8,888          |
| Parallel (GPU – OpenCL) (K-15) | ---- (1,743)      | 2,775            | 4,518          |

## 💡 Note:  
The values in parentheses for the GPU method indicate that CPU parallel detection was still used for face detection, while Gaussian Blur was applied on the GPU.  

---

## 🖼️ Application Screenshots  

Here is how the application looks during execution:

<div style="display: flex; gap: 20px; flex-wrap: wrap; justify-content: center;">
  <img width="681" height="967" alt="ParallelCPUGPU" src="https://github.com/user-attachments/assets/c1af6c54-e48a-4bb1-94a8-5d93f402f409" />
  <img width="644" height="126" alt="ParallelCPUGPU2" src="https://github.com/user-attachments/assets/f0ce9e78-dcbe-46f7-a865-366498df7d3e" />
</div>

> 🔹 Note: The video processing results are available in the repository under the following names:  
> - `output_serialCPU.mp4` – sequential CPU processing  
> - `output_parallelCPU.mp4` – parallel CPU processing  
> - `output_parallelGPU.mp4` – parallel GPU processing

## Build Instructions
```bash
mkdir build
cmake -B .\build\
cmake --build .\build\
cmake --build .\build\ --config Release
.\build\Release\OpenCVExample.exe .
