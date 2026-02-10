#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <CL/opencl.hpp>
#include <iostream>
#include <vector>
#include <future>
#include <numeric>
#include <fileSystem>
#include <chrono>
#include <thread>
#include <atomic>
#include <algorithm>
namespace fs = std::filesystem;

std::atomic<bool> isProcessing(true);

void showProgressDots() {
    int dots = 0;
    while (isProcessing) {
        std::cout << "\r" << std::string(dots % 4, '.') << std::string(3 - (dots % 4), ' ') << std::flush;
        std::this_thread::sleep_for(std::chrono::milliseconds(400));
        dots++;
    }
    std::cout << "\r" << std::flush;
}

std::vector<float> generateGaussianKernel(int radius, float sigma)
{
    int kernelSize = 2 * radius + 1;
    std::vector<float> kernel(kernelSize);

    const float sigma2 = 2.0f * sigma * sigma;
    float sum = 0.0f;

    for (int i = -radius; i <= radius; ++i) {
        float val = std::exp(-(i * i) / sigma2);
        kernel[i + radius] = val;
        sum += val;
    }

    // Normalizacija kernela (da suma bude 1)
    for (auto& k : kernel) {
        k /= sum;
    }

    return kernel;
}

void gaussianBlurHorizontalVerticalSerial(const cv::Mat& input, cv::Mat& output, std::vector<float> kernel1D, int radius) {
    cv::Mat paddedInput;
    cv::copyMakeBorder(input, paddedInput, radius, radius, radius, radius, cv::BORDER_REFLECT);

    // Privremena i izlazna slika
    cv::Mat temp = cv::Mat::zeros(input.size(), input.type());
    output = cv::Mat::zeros(input.size(), input.type());

    // Horizontalni prolaz
    for (int y = 0; y < input.rows; ++y) {
        for (int x = 0; x < input.cols; ++x) {
            for (int c = 0; c < input.channels(); ++c) {
                float sum = 0.0f;
                for (int k = -radius; k <= radius; ++k) {
                    int px = paddedInput.at<cv::Vec3b>(y + radius, x + radius + k)[c];
                    sum += px * kernel1D[k + radius];
                }
                sum = std::clamp(sum, 0.0f, 255.0f);
                temp.at<cv::Vec3b>(y, x)[c] = static_cast<uchar>(sum);
            }
        }
    }

    // Vertikalni prolaz
    for (int y = 0; y < input.rows; ++y) {
        for (int x = 0; x < input.cols; ++x) {
            for (int c = 0; c < input.channels(); ++c) {
                float sum = 0.0f;
                for (int k = -radius; k <= radius; ++k) {
                    int py = temp.at<cv::Vec3b>(std::clamp(y + k, 0, input.rows - 1), x)[c];
                    sum += py * kernel1D[k + radius];
                }
                sum = std::clamp(sum, 0.0f, 255.0f);
                output.at<cv::Vec3b>(y, x)[c] = static_cast<uchar>(sum);
            }
        }
    }
}

void blurRange(std::vector<cv::Mat>& frames, std::vector<float>& kernel1D, int start, int end, int radius) {
    for (int i = start; i < end; ++i) {
        cv::Mat output;
        gaussianBlurHorizontalVerticalSerial(frames[i], output, kernel1D, radius);
        frames[i] = output;
    }
}

void applyGaussianBlurParallel(std::vector<cv::Mat>& frames, int numThreads, std::vector<float> kernel1D, int radius) {
    int totalFrames = frames.size();
    int chunkSize = totalFrames / numThreads;

    std::vector<std::future<void>> futures;
    for (int i = 0; i < numThreads; ++i) {
        int start = i * chunkSize;
        int end = (i == numThreads - 1) ? totalFrames : (start + chunkSize);
        futures.push_back(std::async(std::launch::async, blurRange, std::ref(frames),std::ref(kernel1D), start, end, radius));
    }
    for (auto& f : futures) {
        f.get();
    }
}

void detectFacesParallel(std::vector<cv::Mat>& frames,
                        std::vector<std::vector<cv::Rect>>& facesPerFrame,
                        int start, int end, int detectionInterval) {
    cv::CascadeClassifier faceCascade;
    faceCascade.load("haarcascade_frontalface_default.xml");
    std::vector<cv::Rect> lastDetected;

    for (int i = start; i < end; ++i) {
        if (frames[i].empty()) continue;

        if (i % detectionInterval == 0) {
            cv::Mat gray;
            cv::cvtColor(frames[i], gray, cv::COLOR_BGR2GRAY);
            std::vector<cv::Rect> faces;
            faceCascade.detectMultiScale(gray, faces);
            lastDetected = faces;
        }

        facesPerFrame[i] = lastDetected;
    }
}

const char *kernelSourceHorizontal = R"CLC(
    __kernel void gaussianBlurHorizontal(
    __global const uchar* input,
    __global uchar* output,
    __constant float* gaussKernel,
    int width, int height,
    int channels, int radius
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    for (int c = 0; c < channels; ++c) {
        float sum = 0.0f;
        for (int k = -radius; k <= radius; ++k) {
            int px = clamp(x + k, 0, width - 1);
            int idx = (y * width + px) * channels + c;
            sum += input[idx] * gaussKernel[k + radius];
        }
        int outIdx = (y * width + x) * channels + c;
        output[outIdx] = (uchar)clamp(sum, 0.0f, 255.0f);
    }
}
)CLC";

const char *kernelSourceVertical = R"CLC(
    __kernel void gaussianBlurVertical(
    __global const uchar* input,
    __global uchar* output,
    __constant float* gaussKernel,
    int width, int height,
    int channels, int radius
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    for (int c = 0; c < channels; ++c) {
        float sum = 0.0f;
        for (int k = -radius; k <= radius; ++k) {
            int py = clamp(y + k, 0, height - 1);
            int idx = (py * width + x) * channels + c;
            sum += input[idx] * gaussKernel[k + radius];
        }
        int outIdx = (y * width + x) * channels + c;
        output[outIdx] = (uchar)clamp(sum, 0.0f, 255.0f);
    }
}
)CLC";

int main() {
    std::cout << R"(
    
                    _____  _____   _____  
                    |  __ \|  __ \ / ____|    
                    | |__) | |__) | (___   
                    |  ___/|  _  / \___ \  
                    | |    | | \ \ ____) |          
                    |_|    |_|  \_\_____/              
                                 
                      
                        Avdo Hrnjic                                               
    )" << '\n';
    std::cout << "\033[1;36m";
    std::cout << "\t\tVideo obrada\n\tGaussian Blur efekat na videu uz detekciju lica\n" << std::endl;
    std::cout << "\033[32m";
    if (cv::ocl::haveOpenCL()) {
        std::cout << "OpenCL je dostupan." << std::endl;
        cv::ocl::Context context;
        context.create(cv::ocl::Device::TYPE_ALL);
 
        std::cout << "Broj pronadjenih OpenCL uredjaja: " << context.ndevices() << "\n";

        for (int i = 0; i < context.ndevices(); ++i) {
            cv::ocl::Device device = context.device(i);
            std::cout << "\n=== Uredjaj #" << i << " ===\n";
            std::cout << "Naziv           : " << device.name() << "\n";
            std::cout << "Vendor          : " << device.vendorName() << "\n";
            std::cout << "Tip             : " 
                    << (device.type() == cv::ocl::Device::TYPE_CPU ? "CPU" :
                        device.type() == cv::ocl::Device::TYPE_GPU ? "GPU" : "Other") << "\n";
            std::cout << "OpenCL Verzija  : " << device.version() << "\n";
            std::cout << "Driver Verzija  : " << device.driverVersion() << "\n";
            std::cout << "Globalna memorija : " << device.globalMemSize() / (1024 * 1024) << " MB\n";
            std::cout << "Lokalna memorija  : " << device.localMemSize() / 1024 << " KB\n";
            std::cout << "Compute jedinice  : " << device.maxComputeUnits() << "\n";
            std::cout << "Maks. radna grupa : " << device.maxWorkGroupSize() << "\n";
            std::cout << "Image podrska     : " << (device.imageSupport() ? "Da" : "Ne") << "\n";
            std::cout << "Double podrska    : " << (device.doubleFPConfig() > 0 ? "Da" : "Ne") << "\n\n";
        }
    }
    else
    {
        std::cout << "OpenCL nije dostupan.\n" << std::endl;
    }
    system("wmic cpu get name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed");

    std::cout << "\033[1;36m";

    //Izlistavanje mogucih videa za obradu u direktorijumu
    std::string target = "input";
    fs::path currentPath = fs::current_path();
    std::cout <<"Izaberi jedan od ponudjenih videa: \n";
    for(const auto& entry : fs::directory_iterator(currentPath))
    {
        if(entry.is_regular_file()) 
        {
            std::string fileName = entry.path().filename().string();
            if(fileName.find(target) != std::string::npos)
            {
                std::cout<<"[FILE] "<< fileName <<"\n";
            }
        }
    }
    std::cout << "\n";

    //Unos videa za obradu
    std::string inputVideoPath;
    std::getline(std::cin, inputVideoPath);

    cv::VideoCapture cap(inputVideoPath);
    if (!cap.isOpened()) {
        std::cerr << "Greska pri otvaranju videa!!!!\n" << std::endl;
        return -1;
    }

    //Generisanje kernela za Gaussian Blur
    std::cout << "\nGenerisanje kernela za Gaussian Blur efekat (stepen zamucenja): \n";
    int radius;
    float sigma;
    std::cout << "Unesite poluprecnik kernela (npr. 3): ";
    std::cin >> radius;
    std::cout << "Unesite sigma za Gaussian (npr. 1.0): ";
    std::cin >> sigma;
    if(radius <= 0 || sigma <= 0.0f)
    {
        std::cerr << "\nPoluprecnik i sigma moraju biti pozitivni brojevi!!!\n";
        return 1;
    }
    auto kernel1D = generateGaussianKernel(radius, sigma);
    std::cout << "--- Kernel velicine " << kernel1D.size() << " generisan ---\n\n";

    int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    int totalFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    int channelsRGB = 3;

    // VideoWriter za serijski i paralelni izlaz
    cv::VideoWriter outSerialCPU("output_serialCPU.mp4", fourcc, fps, cv::Size(frameWidth, frameHeight));
    cv::VideoWriter outParallelCPU("output_parallelCPU.mp4", fourcc, fps, cv::Size(frameWidth, frameHeight));
    cv::VideoWriter outParallelGPU("output_parallelGPU.mp4", fourcc, fps, cv::Size(frameWidth, frameHeight));

    // Učitaj sve frejmove i izvrši detekciju lica za svaki 5 frame
    std::cout << "Izvrsavanje detekcije lica na " << inputVideoPath << "\n";
    cv::CascadeClassifier faceCascade;
    faceCascade = cv::CascadeClassifier("haarcascade_frontalface_default.xml");
    if(!faceCascade.load("haarcascade_frontalface_default.xml"))
    {
         std::cerr << "Greska pri ucitavanju Haar cascade fajla!\n";
        return -1;
    }
    std::vector<std::vector<cv::Rect>> facesPerFrameSerialCPU;
    std::vector<cv::Rect> lastDetectedFacesSerialCPU;
    int detectionInterval = 5;
    cap.set(cv::CAP_PROP_POS_FRAMES, 0);

    //Serijsko ucitavanje lica
    std::thread progressThread(showProgressDots);
    std::vector<cv::Mat> originalFrames;
    auto startFacialRecognitionSerial = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < totalFrames; ++i)
    {
        cv::Mat frame;
        cap.read(frame);
        if(frame.empty()) break;

        cv::Mat gray;
        std::vector<cv::Rect> faces;

        if(i % detectionInterval == 0)
        {
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
            faceCascade.detectMultiScale(gray, faces);
            lastDetectedFacesSerialCPU = faces;
        }
        facesPerFrameSerialCPU.push_back(lastDetectedFacesSerialCPU);
        originalFrames.push_back(frame.clone());
    }
    cap.release();
    isProcessing = false;
    progressThread.join();
    auto endFacialRecognitionSerial = std::chrono::high_resolution_clock::now();
    std::cout << "Serijska obrada(CPU) - Detekcija lica: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(endFacialRecognitionSerial - startFacialRecognitionSerial).count() << " ms" <<std::endl;

    //Paralelno ucitavanje lica CPU
    isProcessing = true;
    std::thread progressThreadCPU(showProgressDots);
    auto startFacialRecognition = std::chrono::high_resolution_clock::now();
    int maxThreads = std::thread::hardware_concurrency();
    int chunkSize = totalFrames / maxThreads;
    std::vector<std::vector<cv::Rect>> facesPerFrameParallelCPU(totalFrames);
    std::vector<std::future<void>> futures;

    for (unsigned int t = 0; t < maxThreads; ++t) {
        int start = t * chunkSize;
        int end = (t == maxThreads - 1) ? totalFrames : start + chunkSize;

        futures.push_back(std::async(std::launch::async, detectFacesParallel,
                                    std::ref(originalFrames),
                                    std::ref(facesPerFrameParallelCPU),
                                    start, end, detectionInterval));
    }
    for (auto& f : futures) {
        f.get();
    }
    isProcessing = false;
    progressThreadCPU.join();

    auto endFacialRecognition = std::chrono::high_resolution_clock::now();
    std::cout << "Paralelna obrada(CPU: " << maxThreads << " niti) -" << " Detekcija lica: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(endFacialRecognition - startFacialRecognition).count() << " ms\n" <<std::endl;


    bool hasAnyFace = std::any_of(facesPerFrameSerialCPU.begin(), facesPerFrameSerialCPU.end(),[](const std::vector<cv::Rect>& v) { return !v.empty(); });
    if (hasAnyFace) {
        std::cout << "--- Zavrsena detekcija lica - Pronadjeno je bar jedno lice u videu.\n";
    } else {
        std::cout << "--- Zavrsena detekcija lica - Nijedno lice nije pronadjeno u videu.\n";
    }

    std::cout << "\nOBRADA (Primjenjivanje Gaussian Blur efekta): " << inputVideoPath << "\n"<<std::endl;
    // ----- Serijska obrada CPU -----
    std::vector<cv::Mat> serialFrames = originalFrames;
    isProcessing = true;
    std::thread progressThread2(showProgressDots);
    auto startSerial = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < serialFrames.size(); ++i) {
        cv::Mat blurred;
        gaussianBlurHorizontalVerticalSerial(serialFrames[i], blurred, kernel1D, radius);

        for(const auto& face : facesPerFrameSerialCPU[i])
        {
            int x = std::max(static_cast<int>(face.x - face.width * 0.3), 0);
            int y = std::max(static_cast<int>(face.y - face.height * 0.5), 0);
            int w = std::min(static_cast<int>(face.width * 1.6), serialFrames[i].cols - x);
            int h = std::min(static_cast<int>(face.height * 2.0), serialFrames[i].rows - y);
            
            cv::rectangle(serialFrames[i], cv::Rect(x, y, w, h), cv::Scalar(0, 255, 0), 2);
            cv::Rect expandedFace(x, y, w, h);
            serialFrames[i](expandedFace).copyTo(blurred(expandedFace));
        }
        outSerialCPU.write(blurred);
    }
    auto endSerial = std::chrono::high_resolution_clock::now();
    isProcessing = false;
    progressThread2.join();
    outSerialCPU.release();

    std::cout << "Serijska obrada(CPU): "
              << std::chrono::duration_cast<std::chrono::milliseconds>(endSerial - startSerial).count()
              << " ms - output_serialCPU" << std::endl;

    // ----- Paralelna obrada CPU(future) -----
    std::vector<cv::Mat> parallelFrames = originalFrames;
    int numThreads = std::thread::hardware_concurrency();
    isProcessing = true;
    std::thread progressThread3(showProgressDots);
    auto startParallel = std::chrono::high_resolution_clock::now();
    applyGaussianBlurParallel(parallelFrames, numThreads, kernel1D, radius);
    for(size_t i = 0; i < parallelFrames.size(); ++i)
    {
        cv::Mat blurred;
        blurred = parallelFrames[i].clone();
        for(const auto& face : facesPerFrameParallelCPU[i])
        {
            int x = std::max(static_cast<int>(face.x - face.width * 0.3), 0);
            int y = std::max(static_cast<int>(face.y - face.height * 0.5), 0);
            int w = std::min(static_cast<int>(face.width * 1.6), serialFrames[i].cols - x);
            int h = std::min(static_cast<int>(face.height * 2.0), serialFrames[i].rows - y);
            
            cv::rectangle(serialFrames[i], cv::Rect(x, y, w, h), cv::Scalar(0, 255, 0), 2);
            cv::Rect expandedFace(x, y, w, h);
            originalFrames[i](expandedFace).copyTo(blurred(expandedFace));
        }
        outParallelCPU.write(blurred);
    }
    auto endParallel = std::chrono::high_resolution_clock::now();
    isProcessing = false;
    progressThread3.join();
    outParallelCPU.release();

    std::cout << "Paralelna obrada (CPU: " << numThreads << " niti): "
              << std::chrono::duration_cast<std::chrono::milliseconds>(endParallel - startParallel).count()
              << " ms - output_parallelCPU" << std::endl;

    // ----- Paralelna obrada GPU(OpenCL) -----
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Platform platform = platforms[0];
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    cl::Device device = devices[0];
    cl::Context context({device});
    cl::CommandQueue queue(context,device);

    cl::Program programHorizontal(context, kernelSourceHorizontal, true);
    cl::Program programVertical(context, kernelSourceVertical, true);
    cl::Kernel kernelHorizontal(programHorizontal, "gaussianBlurHorizontal");
    cl::Kernel kernelVertical(programVertical, "gaussianBlurVertical");

    size_t imageSize = frameWidth * frameHeight * channelsRGB;
    cl::Buffer inputBuf(context, CL_MEM_READ_WRITE, imageSize);
    cl::Buffer tempBuf(context, CL_MEM_READ_WRITE, imageSize);
    cl::Buffer outputBuf(context, CL_MEM_READ_WRITE, imageSize);
    cl::Buffer kernelBuf(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * kernel1D.size(), kernel1D.data());

    auto setKernelArgs = [&](cl::Kernel kernel, cl::Buffer inBuf, cl::Buffer outBuf) {
        kernel.setArg(0, inBuf);
        kernel.setArg(1, outBuf);
        kernel.setArg(2, kernelBuf);
        kernel.setArg(3, frameWidth);
        kernel.setArg(4, frameHeight);
        kernel.setArg(5, channelsRGB);
        kernel.setArg(6, radius);
    };
    
    size_t globalSize[2] = { (size_t)frameWidth, (size_t)frameHeight};
    cl::NDRange globalSize2((size_t)frameWidth, (size_t)frameHeight);

    cv::VideoCapture cap2(inputVideoPath);
    if (!cap2.isOpened()) {
        std::cerr << "Greška pri otvaranju videa!" << std::endl;
        return -1;
    }
    cv::Mat frame2;
    std::vector<uchar> output(imageSize);
    int frameIndex = 0;
    isProcessing = true;
    std::thread progressThread4(showProgressDots);
    auto startParallelGPU = std::chrono::high_resolution_clock::now();
    while(cap2.read(frame2))
    {
        if (frame2.channels() != channelsRGB)
            cv::cvtColor(frame2, frame2, cv::COLOR_BGR2RGB);
        queue.enqueueWriteBuffer(inputBuf, CL_TRUE,0, imageSize, frame2.data);
        
        setKernelArgs(kernelHorizontal, inputBuf, tempBuf);
        queue.enqueueNDRangeKernel(kernelHorizontal, cl::NullRange, globalSize2, cl::NullRange);

        setKernelArgs(kernelVertical, tempBuf, outputBuf);
        queue.enqueueNDRangeKernel(kernelVertical, cl::NullRange, globalSize2, cl::NullRange);
        queue.finish();

        queue.enqueueReadBuffer(outputBuf, CL_TRUE, 0, imageSize, output.data());
        cv::Mat outFrame(frameHeight, frameWidth, CV_8UC3, output.data());

        for(const auto& face : facesPerFrameParallelCPU[frameIndex]) {
            int x = std::max(static_cast<int>(face.x - face.width * 0.3), 0);
            int y = std::max(static_cast<int>(face.y - face.height * 0.5), 0);
            int w = std::min(static_cast<int>(face.width * 1.6), outFrame.cols - x);
            int h = std::min(static_cast<int>(face.height * 2.0), outFrame.rows - y);

            cv::Rect expandedFace(x, y, w, h);
            originalFrames[frameIndex](expandedFace).copyTo(outFrame(expandedFace));
        }   
        outParallelGPU.write(outFrame);
        ++frameIndex;
    }
    auto endParallelGPU = std::chrono::high_resolution_clock::now();
    isProcessing = false;
    progressThread4.join();

    std::cout << "Paralelna obrada (GPU): "
              << std::chrono::duration_cast<std::chrono::milliseconds>(endParallelGPU - startParallelGPU).count()
              << " ms - output_parallelGPU.mp4\n" << std::endl;
    cap2.release();
    outParallelGPU.release();

    std::cout << "\nPritisni ENTER za izlaz...";
    std::cin.get();
    return 0;
}