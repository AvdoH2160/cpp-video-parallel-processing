# C++ Video Parallel Processing (CPU & GPU)

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
cd build
cmake ..
cmake --build .
