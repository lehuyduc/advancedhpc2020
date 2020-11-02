#include <stdio.h>
#include <include/labwork.h>
#include <cuda_runtime_api.h>
#include <omp.h>
#include <fstream>

#define ACTIVE_THREADS 6

int main(int argc, char **argv) {
    printf("USTH ICT Master 2018, Advanced Programming for HPC.\n");
    if (argc < 2) {
        printf("Usage: labwork <lwNum> <inputImage>\n");
        printf("   lwNum        labwork number\n");
        printf("   inputImage   the input file name, in JPEG format\n");
        return 0;
    }

    int lwNum = atoi(argv[1]);
    std::string inputFilename;

    // pre-initialize CUDA to avoid incorrect profiling
    printf("Warming up...\n");
    char *temp;
    cudaMalloc(&temp, 1024);

    Labwork labwork;
    if (lwNum != 2 ) {
        inputFilename = std::string(argv[2]);
        labwork.loadInputImage(inputFilename);
    }

    printf("Starting labwork %d\n", lwNum);
    Timer timer;
    timer.start();
    switch (lwNum) {
        case 1:
            labwork.labwork1_CPU();
            labwork.saveOutputImage("labwork2-cpu-out.jpg");
            printf("labwork %d CPU ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            timer.start();
            labwork.labwork1_OpenMP();
            labwork.saveOutputImage("labwork2-openmp-out.jpg");
            break;
        case 2:
            labwork.labwork2_GPU();
            break;
        case 3:
            labwork.labwork3_GPU();
            labwork.saveOutputImage("labwork3-gpu-out.jpg");
            break;
        case 4:
            labwork.labwork4_GPU();
            labwork.saveOutputImage("labwork4-gpu-out.jpg");
            break;
        case 5:
            labwork.labwork5_CPU();
            labwork.saveOutputImage("labwork5-cpu-out.jpg");
            labwork.labwork5_GPU(FALSE);
            labwork.saveOutputImage("labwork5-gpu-out.jpg");
            break;
        case 6:
            labwork.labwork6_GPU();
            labwork.saveOutputImage("labwork6-gpu-out.jpg");
            break;
        case 7:
            labwork.labwork7_GPU();
            printf("[ALGO ONLY] labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork7-gpu-out.jpg");
            break;
        case 8:
            labwork.labwork8_GPU();
            printf("[ALGO ONLY] labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork8-gpu-out.jpg");
            break;
        case 9:
            labwork.labwork9_GPU();
            printf("[ALGO ONLY] labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork9-gpu-out.jpg");
            break;
        case 10:
            labwork.labwork10_GPU();
            printf("[ALGO ONLY] labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork10-gpu-out.jpg");
            break;
    }
    printf("labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
    printf("labwork %d ellapsed see in Report%d\n", lwNum, lwNum);
}

void Labwork::loadInputImage(std::string inputFileName) {
    inputImage = jpegLoader.load(inputFileName);
}

void Labwork::saveOutputImage(std::string outputFileName) {
    jpegLoader.save(outputFileName, outputImage, inputImage->width, inputImage->height, 90);
}

void Labwork::labwork1_CPU() {
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));

    for (int j = 0; j < 100; j++) {     // let's do it 100 times, otherwise it's too fast!
        for (int i = 0; i < pixelCount; i++) {
            outputImage[i * 3] = (char) (((int) inputImage->buffer[i * 3] + (int) inputImage->buffer[i * 3 + 1] +
                                        (int) inputImage->buffer[i * 3 + 2]) / 3);
            outputImage[i * 3 + 1] = outputImage[i * 3];
            outputImage[i * 3 + 2] = outputImage[i * 3];
        }
    }        
}

void Labwork::labwork1_OpenMP() {
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));
    
    std::ofstream fo("Report1/bench_labwork1_teamsize.txt");
    for (int nbThread = 1; nbThread <= 7; nbThread++)
    {
        omp_set_num_threads(nbThread);
        Timer timer;
        timer.start();
        for (int j = 0; j < 100; j++) {     // let's do it 100 times, otherwise it's too fast!
            #pragma omp parallel for
            for (int i = 0; i < pixelCount; i++) {
                outputImage[i * 3] = (char) (((int) inputImage->buffer[i * 3] + (int) inputImage->buffer[i * 3 + 1] +
                                            (int) inputImage->buffer[i * 3 + 2]) / 3);
                outputImage[i * 3 + 1] = outputImage[i * 3];
                outputImage[i * 3 + 2] = outputImage[i * 3];
            }
        }
        fo << nbThread << " " << timer.getElapsedTimeInMilliSec() << "\n";
    }
    fo.close();

    omp_set_num_threads(6);
    fo.open("Report1/bench_labwork1_dynamic.txt");
    for (int portion=1; portion <= 20; portion++) {
        Timer timer;
        timer.start();
        for (int j = 0; j < 100; j++) {     // let's do it 100 times, otherwise it's too fast!
            #pragma omp parallel for schedule(dynamic, pixelCount / portion)
            for (int i = 0; i < pixelCount; i++) {
                outputImage[i * 3] = (char) (((int) inputImage->buffer[i * 3] + (int) inputImage->buffer[i * 3 + 1] +
                                            (int) inputImage->buffer[i * 3 + 2]) / 3);
                outputImage[i * 3 + 1] = outputImage[i * 3];
                outputImage[i * 3 + 2] = outputImage[i * 3];
            }
        }
        fo << portion << " " << timer.getElapsedTimeInMilliSec() << "\n";
    }
    fo.close();
}

int getSPcores(cudaDeviceProp devProp) {
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major) {
        case 2: // Fermi
            if (devProp.minor == 1) cores = mp * 48;
            else cores = mp * 32;
            break;
        case 3: // Kepler
            cores = mp * 192;
            break;
        case 5: // Maxwell
            cores = mp * 128;
            break;
        case 6: // Pascal
            if (devProp.minor == 1) cores = mp * 128;
            else if (devProp.minor == 0) cores = mp * 64;
            else printf("Unknown device type\n");
            break;
        default:
            printf("Unknown device type\n");
            break;
    }
    return cores;
}

//***********************************
void printDevProp(cudaDeviceProp devProp)
{
    printf("Name:                          %s\n",  devProp.name);
    printf("Total global memory:           %lu\n",  devProp.totalGlobalMem);
    printf("Warp size:                     %d\n",  devProp.warpSize);
    printf("Memory Clock Rate (KHz):       %d\n", devProp.memoryClockRate);
    printf("Memory Bus Width (bits):       %d\n", devProp.memoryBusWidth);
    printf("Peak Memory Bandwidth (GB/s):  %f\n\n", 2.0*devProp.memoryClockRate*(devProp.memoryBusWidth/8)/1.0e6);
    printf("Clock rate:                    %d\n",  devProp.clockRate);
    printf("Number of multiprocessors:     %d\n",  devProp.multiProcessorCount);
    return;
}

void Labwork::labwork2_GPU() {
    int nDevices = 0;
    // get all devices
    cudaGetDeviceCount(&nDevices);
    printf("Number total of GPU : %d\n\n", nDevices);
    for (int i = 0; i < nDevices; i++){
        // get informations from individual device
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        // something more here
        printDevProp(prop);
    }
}

//**********************
__global__
void rgb2gray(char* goutput, char* ginput, int pixelCount)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int i=index; i<pixelCount; i += stride) {
        goutput[i * 3] = (char)((int(ginput[i * 3]) + int(ginput[i * 3 + 1]) + int(ginput[i * 3 + 2])) / 3);
        goutput[i * 3 + 1] = goutput[i * 3];
        goutput[i * 3 + 2] = goutput[i * 3];        
    }
}

void Labwork::labwork3_GPU() {
    // Calculate number of pixels
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));

    // Allocate CUDA memory    
    char* ginput = nullptr, *goutput = nullptr;
    cudaMalloc(&ginput, pixelCount * 3);
    cudaMalloc(&goutput, pixelCount * 3);

    // Copy CUDA Memory from CPU to GPU
    cudaMemcpy(ginput, inputImage->buffer, pixelCount * 3, cudaMemcpyHostToDevice);

    // Processing
    rgb2gray<<<80, 64>>>(goutput, ginput, pixelCount);

    // Copy CUDA Memory from GPU to CPU
    cudaMemcpy(outputImage, goutput, pixelCount * 3, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Cleaning
    cudaFree(ginput);
    cudaFree(goutput);
}

void Labwork::labwork4_GPU() {
}

void Labwork::labwork5_CPU() {
}

void Labwork::labwork5_GPU(bool shared) {
}

void Labwork::labwork6_GPU() {
}

void Labwork::labwork7_GPU() {
}

void Labwork::labwork8_GPU() {
}

void Labwork::labwork9_GPU() {

}

void Labwork::labwork10_GPU(){
}


























