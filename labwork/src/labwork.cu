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

    bool option = true;
    if (lwNum == 5) {
        if (argc < 4) option = true;
        else option = atoi(argv[3]);
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

#include <iostream>
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
}

//*****
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


























