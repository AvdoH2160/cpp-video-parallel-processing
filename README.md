# C++ Video Parallel Processing (CPU & GPU)

## 🔧 Tech Stack

<p align="left">
  <a href="https://isocpp.org/" target="_blank"><img src="https://img.shields.io/badge/C++-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white" alt="C++"/></a>
  <a href="https://www.khronos.org/opencl/" target="_blank"><img src="https://img.shields.io/badge/OpenCL-F0DB4F?style=for-the-badge&logo=khronos&logoColor=black" alt="OpenCL"/></a>
  <a href="https://opencv.org/" target="_blank"><img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV"/></a>
  <a href="https://cmake.org/" target="_blank"><img src="https://img.shields.io/badge/CMake-064F8C?style=for-the-badge&logo=cmake&logoColor=white" alt="CMake"/></a>
</p>

## Description
This project demonstrates video processing using Gaussian Blur implemented in three different ways:
- Serial CPU processing
- Parallel CPU processing using std::thread and std::future
- GPU processing using OpenCL

Face detection is applied using Haar Cascade, and detected faces are excluded from blurring.

## Features
- Separable Gaussian Blur (horizontal + vertical pass)
- CPU parallelization using std::thread
- GPU acceleration using OpenCL
- Performance comparison between serial, CPU-parallel, and GPU-parallel approaches
- Face detection with selective blur exclusion
- Execution time measurement

## Technologies
- C++17
- OpenCV
- std::thread / std::future
- OpenCL
- CMake

## Project Structure
- src/ – main C++ source file
- opencl/ – OpenCL kernel files
- data/ – input video and Haar cascade
- results/ – execution time results

## Build Instructions
```bash
mkdir build
cmake -B .\build\
cmake --build .\build\
cmake --build .\build\ --config Release
.\build\Release\OpenCVExample.exe .
