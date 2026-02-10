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

### 📝 Testni video fajlovi

Za testiranje aplikacije, generisao sam dva video fajla koristeći **FFmpeg**:

- **input5s.mp4** – 5 sekundi, 640x480, 30 FPS  
- **input30s.mp4** – 30 sekundi, 640x480, 30 FPS

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

## 🧠 Detekcija lica

- Za prepoznavanje lica korišćen je **Haar Cascade Classifier** (`haarcascade_frontalface_default.xml`) iz OpenCV-a.  
- Detekcija je izvršena svaka 5. frejm (detekcija u intervalima) radi optimizacije performansi.  

---

## ⚙️ Sistem i hardverski zahtevi

Da bi aplikacija ispravno radila sa paralelnom GPU obradom koristeći **OpenCL**, neophodno je ispuniti sledeće zahteve:

1. **Grafička kartica sa podrškom za OpenCL**  
   - GPU mora imati **OpenCL podršku**.  
   - Potrebno je instalirati **najnovije drajvere za GPU** koji uključuju OpenCL runtime.  
   - Testiranje dostupnosti OpenCL uređaja se vrši preko aplikacije pri pokretanju (aplikacija će ispisati broj i tip OpenCL uređaja).
   - OpenCL takođe mora biti **dostupan u CMake projektu**, uključujući `include` i `lib` direktorijume.

2. **OpenCV biblioteka**  
   - OpenCV mora biti **instaliran i kompajliran sa podrškom za OpenCL** (`WITH_OPENCL=ON`) da bi GPU verzija funkcija radila.  
   - Verzija OpenCV-a: **>=4.5** preporučena.  

3. **C++17 kompatibilan kompajler**  
   - GCC >= 9, MSVC 2019 ili sličan, sa podrškom za **std::filesystem, std::thread, std::future**.

4. **Opcionalno: FFmpeg**  
   - Ako želite generisati testne video fajlove (`input5s.mp4`, `input30s.mp4`) ili konvertovati video formate, FFmpeg treba biti instaliran i dostupan iz komandne linije.

---

## 🖥️ Testni uređaj

| Komponenta            | Specifikacija                               |
|-----------------------|---------------------------------------------|
| Operativni sistem      | Microsoft Windows 11 Pro                    |
| Procesor              | AMD Ryzen 5 5500, 6 jezgara / 12 niti      |
| RAM                   | 16 GB                                       |
| Grafička kartica      | AMD Radeon RX 6650 XT (OpenCL podrška)     |

---

## 📹 Testni video

- Fajl: `inputFACE.mp4`  
- Trajanje: 26 sekundi  
- Rezolucija: 852x480  
- FPS: 30  
- Napomena: Za paralelno GPU **detekciju lica** korišćena je vrijednost **Paralelno (CPU)**.  
- Kernel za Gaussian Blur: **K-15**  

---

## ⏱️ Rezultati obrade

| Metoda                        | Detekcija lica [ms] | Gaussian Blur [ms] | Ukupno vrijeme [ms] |
|--------------------------------|-------------------|------------------|--------------------|
| Sekvencijalno (CPU) (K-15)     | 4,133             | 34,822           | 38,955             |
| Paralelno (CPU) (K-15)         | 1,743             | 7,145            | 8,888              |
| Paralelno (GPU – OpenCL) (K-15)| ---- (1,743)      | 2,775            | 4,518              |

---

💡 **Napomena:**  
Vrijednosti u zagradama kod GPU metode označavaju da se za detekciju lica i dalje koristi **CPU paralelna detekcija**, dok se Gaussian Blur primjenjuje na GPU-u.

## Build Instructions
```bash
mkdir build
cmake -B .\build\
cmake --build .\build\
cmake --build .\build\ --config Release
.\build\Release\OpenCVExample.exe .
