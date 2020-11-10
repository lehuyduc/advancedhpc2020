#include <stdio.h>
#include <include/labwork.h>
#include <cuda_runtime_api.h>
#include <omp.h>
#include <fstream>
#include <iostream>
#include <include/jpegloader.h>
using std::cin;
using std::cout;
using std::string;

#define ACTIVE_THREADS 6

JpegInfo *inputImage2;

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

    bool option = true;
    if (lwNum == 5) {
        if (argc < 4) option = true;
        else option = atoi(argv[3]);
    }

    char subTask = 'a';
    float parameter = 0.0f;
    if (lwNum == 6) {
        if (argc < 4) subTask = 'a';
        else subTask = argv[3][0];

        if (argc < 5) parameter = 0;
        else parameter = std::stof(argv[4]);

        if (subTask == 'c') {
            if (argc < 6) {
                cout << "Labwork 6 subtask C needs a second image\n";
                exit(0);
            }
            JpegLoader loader;
            string inputFileName2 = std::string(argv[5]);
            inputImage2 = loader.load(inputFileName2);
        }
    }

    printf("Starting labwork %d\n", lwNum);
    Timer timer;
    timer.start();
    switch (lwNum) {
        case 1:
            labwork.labwork1_CPU();
            labwork.saveOutputImage("labwork1-cpu-out.jpg");
            printf("labwork %d CPU ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            timer.start();
            labwork.labwork1_OpenMP();
            labwork.saveOutputImage("labwork1-openmp-out.jpg");
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
            labwork.labwork5_GPU(option);
            labwork.saveOutputImage("labwork5-gpu-out.jpg");
            break;
        case 6:
            labwork.labwork6_GPU(subTask, parameter);
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
    outputImage = static_cast<byte *>(malloc(pixelCount * 3));

    for (int j = 0; j < 100; j++) {     // let's do it 100 times, otherwise it's too fast!
        for (int i = 0; i < pixelCount; i++) {
            outputImage[i * 3] = (byte) (((int) inputImage->buffer[i * 3] + (int) inputImage->buffer[i * 3 + 1] +
                                        (int) inputImage->buffer[i * 3 + 2]) / 3);
            outputImage[i * 3 + 1] = outputImage[i * 3];
            outputImage[i * 3 + 2] = outputImage[i * 3];
        }
    }        
}

void Labwork::labwork1_OpenMP() {
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<byte *>(malloc(pixelCount * 3));
    
    std::ofstream fo("Report1/bench_labwork1_teamsize.txt");
    for (int nbThread = 1; nbThread <= 7; nbThread++)
    {
        omp_set_num_threads(nbThread);
        Timer timer;
        timer.start();
        for (int j = 0; j < 100; j++) {     // let's do it 100 times, otherwise it's too fast!
            #pragma omp parallel for
            for (int i = 0; i < pixelCount; i++) {
                outputImage[i * 3] = (byte) (((int) inputImage->buffer[i * 3] + (int) inputImage->buffer[i * 3 + 1] +
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
                outputImage[i * 3] = (byte) (((int) inputImage->buffer[i * 3] + (int) inputImage->buffer[i * 3 + 1] +
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
    printf("Number of CUDA cores:          %d\n", getSPcores(devProp));
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
void rgb2gray_labwork3(byte* goutput, byte* ginput, int pixelCount)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < pixelCount) {
		goutput[i * 3] = (byte)((int(ginput[i * 3]) + int(ginput[i * 3 + 1]) + int(ginput[i * 3 + 2])) / 3);
        goutput[i * 3 + 1] = goutput[i * 3];
        goutput[i * 3 + 2] = goutput[i * 3];        
	}
}

void Labwork::labwork3_GPU() {
	Timer timer;
	double tmp, kernelTime;
	
	int pixelCount = inputImage->width * inputImage->height;
	outputImage = static_cast<byte *>(malloc(pixelCount * 3));
	// Allocate CUDA memory    
   	byte* ginput = nullptr, *goutput = nullptr;
	cudaMalloc(&ginput, pixelCount * 3);
	cudaMalloc(&goutput, pixelCount * 3);

	std::ofstream fo("Report3/bench_labwork3_totaltime.txt");
    for (int blockSize=1; blockSize<=256; blockSize++)
    {
    	// Calculate number of pixels
    	timer.start();
    	for (int t=1; t<=100; t++)
    	{	
			// Copy CUDA Memory from CPU to GPU
			cudaMemcpy(ginput, inputImage->buffer, pixelCount * 3, cudaMemcpyHostToDevice);

			// Processing
			int numBlock = (pixelCount + blockSize - 1) / blockSize;
			rgb2gray_labwork3<<<numBlock, blockSize>>>(goutput, ginput, pixelCount);
			
			// Copy CUDA Memory from GPU to CPU
			cudaMemcpy(outputImage, goutput, pixelCount * 3, cudaMemcpyDeviceToHost);		
    	}
    	tmp = timer.getElapsedTimeInMilliSec();
    	fo << blockSize << " " << tmp << "\n";
		
    }
    fo.close();
    
    fo.open("Report3/bench_labwork3_kerneltime.txt");
    for (int blockSize=1; blockSize<=256; blockSize++)
    {
    	kernelTime = 0;
    	for (int t=1; t<=100; t++)
    	{	
			// Copy CUDA Memory from CPU to GPU
			cudaMemcpy(ginput, inputImage->buffer, pixelCount * 3, cudaMemcpyHostToDevice);

			// Processing
			timer.start();
			int numBlock = pixelCount / blockSize + 1;
			rgb2gray_labwork3<<<numBlock, blockSize>>>(goutput, ginput, pixelCount);
			cudaDeviceSynchronize();
			kernelTime += timer.getElapsedTimeInMilliSec(); 
			
			// Copy CUDA Memory from GPU to CPU
			cudaMemcpy(outputImage, goutput, pixelCount * 3, cudaMemcpyDeviceToHost);		
    	}
    	fo << blockSize << " " << kernelTime << "\n";		
    }
    fo.close();
    
    cudaFree(ginput);
	cudaFree(goutput);
}

//**********************
__global__
void rgb2gray_labwork4(byte* goutput, byte* ginput, int height, int width, int pixelCount)
{
	const int row = blockIdx.x * blockDim.x + threadIdx.x,
			  col = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (row < height && col < width) {
		const int i = row * width + col;
		goutput[i * 3] = (byte)((int(ginput[i * 3]) + int(ginput[i * 3 + 1]) + int(ginput[i * 3 + 2])) / 3);
        goutput[i * 3 + 1] = goutput[i * 3];
        goutput[i * 3 + 2] = goutput[i * 3];
	}			  
}

void Labwork::labwork4_GPU() {
	Timer timer;
	double tmp, kernelTime;
	
	int pixelCount = inputImage->width * inputImage->height;
	int width = inputImage->width, height = inputImage->height;
	outputImage = static_cast<byte *>(malloc(pixelCount * 3));
	// Allocate CUDA memory    
	byte* ginput = nullptr, *goutput = nullptr;
	cudaMalloc(&ginput, pixelCount * 3);
	cudaMalloc(&goutput, pixelCount * 3);
	
	//****
	std::ofstream fo("Report4/bench_labwork4_totaltime.txt");
    for (int blockSize=8; blockSize<=32; blockSize+=8)
    {
    	// Calculate number of pixels
    	timer.start();
    	dim3 gridDim = dim3((height + blockSize - 1) / blockSize, (width + blockSize - 1) / blockSize, 1);
    	dim3 blockDim = dim3(blockSize, blockSize, 1);
    	
    	for (int t=1; t<=100; t++)
    	{	
			// Copy CUDA Memory from CPU to GPU
			cudaMemcpy(ginput, inputImage->buffer, pixelCount * 3, cudaMemcpyHostToDevice);

			// Processing
			rgb2gray_labwork4<<<gridDim, blockDim>>>(goutput, ginput, height, width, pixelCount);
			
			// Copy CUDA Memory from GPU to CPU
			cudaMemcpy(outputImage, goutput, pixelCount * 3, cudaMemcpyDeviceToHost);		
    	}
    	tmp = timer.getElapsedTimeInMilliSec();
    	fo << blockSize << " " << tmp << "\n";
    }
    fo.close();
    
    fo.open("Report4/bench_labwork4_kerneltime.txt");
    for (int blockSize=8; blockSize<=32; blockSize+=8)
    {
    	// Calculate number of pixels
    	timer.start();
    	dim3 gridDim = dim3(height / blockSize + 1, width / blockSize + 1, 1);
    	dim3 blockDim = dim3(blockSize, blockSize, 1);
    	kernelTime = 0;
    	
    	for (int t=1; t<=100; t++)
    	{	
			// Copy CUDA Memory from CPU to GPU
			cudaMemcpy(ginput, inputImage->buffer, pixelCount * 3, cudaMemcpyHostToDevice);

			// Processing
			timer.start();
			rgb2gray_labwork4<<<gridDim, blockDim>>>(goutput, ginput, height, width, pixelCount);
			cudaDeviceSynchronize();
			kernelTime += timer.getElapsedTimeInMilliSec();
			
			// Copy CUDA Memory from GPU to CPU
			cudaMemcpy(outputImage, goutput, pixelCount * 3, cudaMemcpyDeviceToHost);		
    	}
    	tmp = timer.getElapsedTimeInMilliSec();
    	fo << blockSize << " " << kernelTime << "\n";
    }
    fo.close();
          
    cudaFree(ginput);
	cudaFree(goutput);
}

//*********************
#define cell(i,j,w) ((i)*(w) + (j))

const
float filt[] = {0,   0,   1,   2,   1,   0,   0,
					 0,   3,   13,  22,  13,  3,   0,
					 1,   13,  59,  97,  59,  13,  1,
					 2,   22,  97,  159, 97,  22,  2,
					 1,   13,  59,  97,  59,  13,  1,
					 0,   3,   13,  22,  13,  3,   0,
					 0,   0,   1,   2,   1,   0,   0};
const float filtSum = 1003, filtSumInv = double(1) / filtSum;
const int filtH = 7, filtW = 7, midRow = 3, midCol = 3;

__constant__ float gfilt[49];
__constant__ float gfiltSum, gfiltSumInv;
__constant__ int gfiltH, gfiltW, gmidRow, gmidCol;

void Labwork::labwork5_CPU() {
	//Timer timer;
	//double tmp, kernelTime;

	int pixelCount = inputImage->width * inputImage->height;
	int width = inputImage->width, height = inputImage->height;
	labwork1_CPU();
	byte* grayImage = (byte*)malloc(pixelCount);
	for (int i=0; i<pixelCount; i++) grayImage[i] = outputImage[3*i];

	 //****
	for (int i=-midRow; i<height; i++)
	for (int j=-midCol; j<width; j++)
	{
		float sum = 0;
		int outputX = i + midRow, outputY = j + midCol;
		if (outputX < 0 || outputY < 0  || outputX >= height || outputY >= width) continue; // output outside image
		
		for (int u=0; u<filtH; u++)
		for (int v=0; v<filtW; v++) 
			{
				int pixelRow = i + u, pixelCol = j + v;
				if (pixelRow < 0 || pixelCol < 0 || pixelRow >= height || pixelCol >= width) 
					sum += 0;
				else 
					sum += filt[cell(u,v,filtW)] * grayImage[cell(pixelRow, pixelCol, width)];
			}
			
		int outputPixel = cell(outputX, outputY, width);
		outputImage[3 * outputPixel] = sum * filtSumInv;
		outputImage[3 * outputPixel + 1] = outputImage[3 * outputPixel];
		outputImage[3 * outputPixel + 2] = outputImage[3 * outputPixel];
    }
    
    free(grayImage);
}

//****
const int TILE_DIM = 32;
#define tidx (threadIdx.x)
#define tidy (threadIdx.y)

// assume that blocks cover all columns: blockDim.y * gridDim.y >= width.
// block size = TILE_DIM x TILE_DIM
__global__
void convo2dNoShare(byte* goutput, byte* ginput, int height, int width)
{
    // place top-left corner of filter on each thread (y,x) -> output at pixel (y + midRow, x + midCol)
    const int outputCol = blockIdx.x * blockDim.x + threadIdx.x;
    if (outputCol >= width) return; // these threads will always output to pixel not in image
    const int col = outputCol - gmidCol; 
    
    for (int row = threadIdx.y - gmidRow; row < height; row += blockDim.y)
    {
        const int outputRow = row + gmidRow;
        if (outputRow < height)
        {
            float sum = 0;

            for (int u=0; u<gfiltH; u++)
            for (int v=0; v<gfiltW; v++)
            {
                int pixelRow = row + u, pixelCol = col + v;
				if (pixelRow < 0 || pixelCol < 0 || pixelRow >= height || pixelCol >= width) 
					sum += 0;
				else 
					sum += gfilt[cell(u,v,gfiltW)] * ginput[cell(pixelRow, pixelCol, width)];
            }
            
            int outputPixel = cell(outputRow, outputCol, width);
            float outputValue = sum * gfiltSumInv;
            goutput[3 * outputPixel] = outputValue;
            goutput[3 * outputPixel + 1] = outputValue;
            goutput[3 * outputPixel + 2] = outputValue;
        }
    } 
}

__global__
void convo2dShare(byte* goutput, byte* ginput, int height, int width)
{
    __shared__ float smem[TILE_DIM][TILE_DIM+1];    // padding to prevent mem conflict

    // each block process (TILE_DIM - filtH + 1) rows and (TILE_DIM - filtW + 1) columns
    const int stride = TILE_DIM - gfiltH + 1,
              loop = (height + stride - 1) / stride;
                
    const int outputCol = blockIdx.x * (TILE_DIM - gfiltW + 1) + threadIdx.x, 
              col = outputCol - gmidCol;
    if (outputCol >= width) return;

    int outputRow = blockIdx.y * blockDim.y + threadIdx.y,
        row = outputRow - gmidRow;

    //****
    for (int t = 0; t < loop; t++)
    {
        __syncthreads();
        
        // load data into shared memory
        if (row < 0 || col < 0 || row>=height || col>=width)
            smem[tidy][tidx] = 0;
        else 
            smem[tidy][tidx] = ginput[cell(row, col, width)];
        __syncthreads();

        
        if (outputRow < height 
            && tidy < TILE_DIM - gfiltH + 1 && tidx < TILE_DIM - gfiltW + 1) // top-left of the kernel is placed here, and it must fit shared mem 
        {
            float sum = 0;

            for (int u=0; u<gfiltH; u++)
            for (int v=0; v<gfiltW; v++)
            {
                int pixelRow = row + u, pixelCol = col + v;
				if (pixelRow < 0 || pixelCol < 0 || pixelRow >= height || pixelCol >= width) 
					sum += 0;
				else 
					sum += gfilt[cell(u,v,gfiltW)] * smem[tidy + u][tidx + v];
            }
            
            int outputPixel = cell(outputRow, outputCol, width);
            float outputValue = sum * gfiltSumInv;
            goutput[3 * outputPixel] = outputValue;
            goutput[3 * outputPixel + 1] = outputValue;
            goutput[3 * outputPixel + 2] = outputValue;
        }

        outputRow += stride;
        row += stride;
    } 
}

void Labwork::labwork5_GPU(bool shared) {
    //Timer timer;
    //double tmp, kernelTime;

    int pixelCount = inputImage->width * inputImage->height;
    int width = inputImage->width, height = inputImage->height;
    labwork1_CPU();
    byte* grayImage = (byte*)malloc(pixelCount);
    for (int i=0; i<pixelCount; i++) grayImage[i] = outputImage[3*i];

    // Allocate CUDA memory    
    byte* ginput = nullptr, *goutput = nullptr;
    cudaMalloc(&ginput, pixelCount);
    cudaMalloc(&goutput, pixelCount * 3);	

    // Copy to constant memory because all thread access the same place in the filter at the same time
    // Constant scalar variable can be store in either register or constant mem. 
    cudaMemcpyToSymbol(gfilt, filt, 49 * sizeof(float));
    cudaMemcpyToSymbol(gfiltSum, &filtSum, sizeof(float));
    cudaMemcpyToSymbol(gfiltSumInv, &filtSumInv, sizeof(float));
    cudaMemcpyToSymbol(gfiltH, &filtH, sizeof(int));
    cudaMemcpyToSymbol(gfiltW, &filtW, sizeof(int));
    cudaMemcpyToSymbol(gmidRow, &midRow, sizeof(int));
    cudaMemcpyToSymbol(gmidCol, &midCol, sizeof(int));

    //*****
    cudaMemcpy(ginput, grayImage, pixelCount, cudaMemcpyHostToDevice);

    // Processing
    if (!shared) {
        dim3 blockDim = dim3(TILE_DIM, TILE_DIM, 1);
        dim3 gridDim = dim3((width + TILE_DIM - 1) / TILE_DIM, 1, 1);
        convo2dNoShare<<<gridDim, blockDim>>>(goutput, ginput, height, width);
    }
    else {
        int columnsPerBlock = TILE_DIM - filtW + 1;
        dim3 blockDim = dim3(TILE_DIM, TILE_DIM, 1);
        dim3 gridDim = dim3((width + columnsPerBlock - 1) / columnsPerBlock, 1, 1);
        convo2dShare<<<gridDim, blockDim>>>(goutput, ginput, height, width);
    }
    // Copy CUDA Memory from GPU to CPU
    cudaMemcpy(outputImage, goutput, pixelCount * 3, cudaMemcpyDeviceToHost);	

    free(grayImage);
    cudaFree(ginput);
    cudaFree(goutput);
}

//********

// input/output are array of 3*pixelCount char
__device__
void binarizePixel(int i, byte* ginput, byte* goutput, float threshold)
{
    const byte gray = byte(int(ginput[3*i]) + int(ginput[3*i + 1]) + int(ginput[3*i+2])) / 3;
    const byte binary = (gray >= threshold) ? 255 : 0;
    
    goutput[3 * i] = binary;
    goutput[3 * i + 1] = binary;
    goutput[3 * i + 2] = binary;
}   

// coeff is the ratio of new brightness. FOr example, 50% more brightness -> coeff = 1.5, 20% less -> coeff = 0.8
__device__
void changePixelBrightness(int i, byte* ginput, byte* goutput, float coeff)
{
    goutput[3 * i] = min(coeff * ginput[3 * i], 255.0f);
    goutput[3 * i + 1] = min(coeff * ginput[3 * i + 1], 255.0f);
    goutput[3 * i + 2] = min(coeff * ginput[3 * i + 2], 255.0f);
}

template<char subTask>
__global__
void mappingAB(int n, byte* ginput, byte* goutput, float parameter)
{
    const int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
    {
        if (subTask == 'a') binarizePixel(i, ginput, goutput, parameter);
        else if (subTask == 'b') changePixelBrightness(i, ginput, goutput, parameter);
    }
}

__global__
void mappingC(int n, byte* ginput1, byte* ginput2, byte* goutput, const float c)
{
    const int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
    {
        const int index = 3 * i;
        goutput[index] = c * ginput1[index] + (1 - c) * ginput2[index];
        goutput[index + 1] = c * ginput1[index + 1] + (1 - c) * ginput2[index + 1];
        goutput[index + 2] = c * ginput1[index + 2] + (1 - c) * ginput2[index + 2];
    }
}

void Labwork::labwork6_GPU(char subTask, float parameter) {
    Timer timer;
	double tmp, kernelTime;
	
	int pixelCount = inputImage->width * inputImage->height;
	outputImage = static_cast<byte *>(malloc(pixelCount * 3));
	// Allocate CUDA memory    
	byte* ginput = nullptr, *ginput2 = nullptr, *goutput = nullptr;
	cudaMalloc(&ginput, pixelCount * 3);
    cudaMalloc(&goutput, pixelCount * 3);
    
    //
    kernelTime = 0;
    for (int t=1; t<=100; t++)
    {
        cudaMemcpy(ginput, inputImage->buffer, pixelCount * 3, cudaMemcpyHostToDevice);
        if (subTask == 'c') {
            cudaMalloc(&ginput2, pixelCount * 3);
            cudaMemcpy(ginput2, inputImage2->buffer, pixelCount * 3, cudaMemcpyHostToDevice);
        }

        timer.start();
        if (subTask == 'a') mappingAB<'a'><<<80, 128>>>(pixelCount, ginput, goutput, parameter);
        else if (subTask == 'b') mappingAB<'b'><<<80, 128>>>(pixelCount, ginput, goutput, parameter);
        else if (subTask == 'c') mappingC<<<80, 128>>>(pixelCount, ginput, ginput2, goutput, parameter);
        else {
            std::cout << "Labwork 6 wrong subtask name (a,b,c only). Exiting\n";
            exit(0);
        }
        tmp = timer.getElapsedTimeInMilliSec();
        kernelTime += tmp;

        cudaMemcpy(outputImage, goutput, pixelCount * 3, cudaMemcpyDeviceToHost);
    }
    cout << "Time 100 task a = " << kernelTime << "ms\n";

    cudaFree(ginput);
    if (subTask == 'c') cudaFree(ginput2);
    cudaFree(goutput);
}


//*****************

__device__
inline void getMinmax(uchar2* sdata, int stride) {
    sdata[tidx].x = min(sdata[tidx].x, sdata[tidx + stride].x);
    sdata[tidx].y = max(sdata[tidx].y, sdata[tidx + stride].y);    
}

__device__
inline void getMinmaxVolatile(volatile uchar2* sdata, int stride) {
    sdata[tidx].x = min(sdata[tidx].x, sdata[tidx + stride].x);
    sdata[tidx].y = max(sdata[tidx].y, sdata[tidx + stride].y);    
}

template<unsigned int blockSize>
__device__
void warpReduceMinmax(volatile uchar2* sdata)
{
    if (blockSize >= 64) getMinmaxVolatile(sdata, 32);
    if (blockSize >= 32) getMinmaxVolatile(sdata, 16);
    if (blockSize >= 16) getMinmaxVolatile(sdata, 8);
    if (blockSize >= 8) getMinmaxVolatile(sdata, 4);
    if (blockSize >= 4) getMinmaxVolatile(sdata, 2);
    if (blockSize >= 2) getMinmaxVolatile(sdata, 1);
}

// blockSize must be power of 2, <= 512.
template<unsigned int blockSize>
__global__
void reduceMinmaxStage1(const int n, byte* input, uchar2* output)
{
    __shared__ uchar2 sdata[blockSize]; // uchar[2*i] = min, uchar[2*i+1] = max
    // * 2 because each element loads and find min/max of 2 input index at each step
    int i = blockIdx.x * (blockSize * 2) + threadIdx.x; 
    const int gridSize = blockSize * 2 * gridDim.x;    

    // strided loop to cover the entire array. We do this instead of creating 
    // more blocks because too many blocks = inefficient.
    // NOTE 1: IF N IS NOT POWER OF 2 THEN THERE NEEDS TO BE AN IF STATEMENT FOR THE SECOND OPERATION
    // Note 2: use register perform strided-reduction instead of shared memory for more speed, 
    //         since each thread process completely independent data. Data is only written to smem at the end
    
    byte minval = 255, maxval = 0;
    while (i < n) {        
        minval = min(minval, input[i]);
        maxval = max(maxval, input[i]);
        if (i + blockSize < n) {
            minval = min(minval, input[i + blockSize]);
            maxval = max(maxval, input[i + blockSize]);
        }
        i += gridSize;
    }        
    sdata[tidx].x = minval;
    sdata[tidx].y = maxval;    
    __syncthreads();

    // Manually unrolling the loop to reduce loop-overhead. Probably #pragma unroll is enough.
    // these if statements are done at compile time, thansk to template.
    if (blockSize >= 512) {if (tidx < 256) getMinmax(sdata, 256); __syncthreads();}
    if (blockSize >= 256) {if (tidx < 128) getMinmax(sdata, 128); __syncthreads();}
    if (blockSize >= 128) {if (tidx < 64)  getMinmax(sdata, 64);  __syncthreads();}

    if (tidx < 32) warpReduceMinmax<blockSize>(sdata);
    if (tidx == 0) {
        output[blockIdx.x].x = sdata[0].x;
        output[blockIdx.x].y = sdata[0].y;
    }
}


// in this function numBlock is around 64-512.
// So we launch 1 block with enough threads
template<unsigned int blockSize>
__global__
void reduceMinmaxStage2(int numBlock, uchar2* stage1Output, uchar2* minmax)
{
    __shared__ uchar2 sdata[blockSize];    
    if (tidx < numBlock) sdata[tidx] = stage1Output[tidx];
    else {
        sdata[tidx].x = 255;
        sdata[tidx].y = 0;
    }
    __syncthreads();

    if (blockSize >= 512) {if (tidx < 256) getMinmax(sdata, 256); __syncthreads();}
    if (blockSize >= 256) {if (tidx < 128) getMinmax(sdata, 128); __syncthreads();}
    if (blockSize >= 128) {if (tidx < 64)  getMinmax(sdata, 64);  __syncthreads();}

    if (tidx < 32) warpReduceMinmax<blockSize>(sdata);
    if (tidx == 0) *minmax = sdata[0];
}

// input is single-channel grayscale, but output is 3-channel
__global__
void grayscaleStretch(const int n, byte* input, byte* output, const float minval, const float maxval)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    const float coeff = float(255) / (maxval - minval); // multiply much faster than division

    for (int i=index; i<n; i+=stride) {
        output[3 * i] = (float(input[i]) - minval) * coeff;
        output[3 * i + 1] = output[3 * i];
        output[3 * i + 2] = output[3 * i];
    }
}

void Labwork::labwork7_GPU() {
    //Timer timer;
    //double tmp, kernelTime;
    
    int pixelCount = inputImage->width * inputImage->height;    
    labwork1_CPU();
    byte* grayImage = (byte*)malloc(pixelCount);
    for (int i=0; i<pixelCount; i++) grayImage[i] = outputImage[3*i];

    // Allocate CUDA memory    
    const int numBlock = 64, blockSize = 128;
    byte* ginput = nullptr, *goutput = nullptr;
    uchar2* gstage1Output, *gminmax; 
    uchar2 minmax;
	cudaMalloc(&ginput, pixelCount);
    cudaMalloc(&gstage1Output, numBlock * sizeof(uchar2));
    cudaMalloc(&goutput, pixelCount * 3);    
    cudaMalloc(&gminmax, sizeof(uchar2));

    for (int t=1; t<=20; t++)
    {
        //cout << "pixel count = " << pixelCount << "\n";
        cudaMemcpy(ginput, grayImage, pixelCount, cudaMemcpyHostToDevice);    
        reduceMinmaxStage1<blockSize><<<numBlock, blockSize>>>(pixelCount, ginput, gstage1Output);
        reduceMinmaxStage2<blockSize><<<1, blockSize>>>(numBlock, gstage1Output, gminmax);
        cudaMemcpy(&minmax, gminmax, sizeof(uchar2), cudaMemcpyDeviceToHost);
        grayscaleStretch<<<numBlock,blockSize>>>(pixelCount, ginput, goutput, minmax.x, minmax.y);
        cudaMemcpy(outputImage, goutput, pixelCount * 3, cudaMemcpyDeviceToHost);
    }

    cudaFree(ginput);
    cudaFree(gstage1Output);
    cudaFree(goutput);    
    cudaFree(gminmax);
}

//*****
struct HSV {
    float h, s, v;
};

__device__
inline void rgb2hsvConvert(float r, float g, float b, HSV* outputPtr)
{    
    float h, s, v;
    
    r /= 255.0f;
    g /= 255.0f;
    b /= 255.0f;	
	
	float max = fmax(r, fmax(g, b));
	float min = fmin(r, fmin(g, b));
	float diff = max - min;
	
	v = max;
	
	if(v == 0.0f) { // black
		h = s = 0.0f;
	} else {
		s = diff / v;
		if(diff < 0.001f) { // grey
			h = 0.0f;
		} else { // color
			if(max == r) {
				h = 60.0f * (g - b)/diff;
				if(h < 0.0f) { h += 360.0f; }
			} else if(max == g) {
				h = 60.0f * (2 + (b - r)/diff);
			} else {
				h = 60.0f * (4 + (r - g)/diff);
			}
		}		
	}
    
    outputPtr->h = h;
    outputPtr->s = s;
    outputPtr->v = v;	
}

__global__
void rgb2hsvCuda(int n, byte* input, HSV* output)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int gridSize = blockDim.x * gridDim.x;

    while (i < n) {
        rgb2hsvConvert(input[3*i], input[3*i+1], input[3*i+2], &output[i]);
        i += gridSize;
    }
}

//****
__device__
inline void hsv2rgbConvert(int i, HSV* input, byte* output)
{
    float h = input[i].h, s = input[i].s, v = input[i].v;
	float r, g, b;		
	
	float f = h/60.0f;
	float hi = floorf(f);
	f = f - hi;
	float p = v * (1 - s);
	float q = v * (1 - s * f);
	float t = v * (1 - s * (1 - f));
	
	if(hi == 0.0f || hi == 6.0f) {
		r = v;
		g = t;
		b = p;
	} else if(hi == 1.0f) {
		r = q;
		g = v;
		b = p;
	} else if(hi == 2.0f) {
		r = p;
		g = v;
		b = t;
	} else if(hi == 3.0f) {
		r = p;
		g = q;
		b = v;
	} else if(hi == 4.0f) {
		r = t;
		g = p;
		b = v;
	} else {
		r = v;
		g = p;
		b = q;
    }
    
    output[3*i] = 255.0f * r;
    output[3*i+1] = 255.0f * g;
    output[3*i+2] = 255.0f * b;		
}

__global__
void hsv2rgbCuda(int n, HSV* input, byte* output)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int gridSize = blockDim.x * gridDim.x;

    while (i < n) {
        hsv2rgbConvert(i, input, output);
        i += gridSize;
    }

}

void Labwork::labwork8_GPU() {
    int pixelCount = inputImage->width * inputImage->height;
    byte* input = inputImage->buffer;
	outputImage = static_cast<byte *>(malloc(pixelCount * 3));
	// Allocate CUDA memory    
    byte* ginput = nullptr, *goutput = nullptr;
    HSV* hsvImage = nullptr, *ghsvImage = nullptr;   
    cudaHostAlloc((void**)&hsvImage, pixelCount * sizeof(HSV), cudaHostAllocDefault);
    cudaMalloc(&ginput, pixelCount * 3);
    cudaMalloc(&ghsvImage, pixelCount * sizeof(HSV));
    cudaMalloc(&goutput, pixelCount * 3);
    
    cudaMemcpy(ginput, input, pixelCount * 3, cudaMemcpyHostToDevice);
    rgb2hsvCuda<<<64, 128>>>(pixelCount, ginput, ghsvImage);
    hsv2rgbCuda<<<64, 128>>>(pixelCount, ghsvImage, goutput);
    cudaMemcpy(outputImage, goutput, pixelCount * 3, cudaMemcpyDeviceToHost);
    cudaMemcpy(hsvImage, ghsvImage, pixelCount * sizeof(HSV), cudaMemcpyDeviceToHost);    
}

void Labwork::labwork9_GPU() {

}

void Labwork::labwork10_GPU(){
}


























