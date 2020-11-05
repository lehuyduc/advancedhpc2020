#pragma once

#include <include/jpegloader.h>
#include <include/timer.h>

class Labwork {

private:
    JpegLoader jpegLoader;
    JpegInfo *inputImage;
    byte *outputImage;

public:
    void loadInputImage(std::string inputFileName);
    void saveOutputImage(std::string outputFileName);

    void labwork1_CPU();
    void labwork1_OpenMP();

    void labwork2_GPU();

    void labwork3_GPU();

    void labwork4_GPU();

    void labwork5_CPU();

    void labwork5_GPU(bool shared);

    void labwork6_GPU(char subtask='a', float parameter=0.0f);

    void labwork7_GPU();

    void labwork8_GPU();

    void labwork9_GPU();

    void labwork10_GPU();
};
