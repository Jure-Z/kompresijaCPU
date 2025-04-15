#include <iostream>
#include <cassert>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <array>
#include <cmath>
#include <numeric>
#include <bitset>
#include <limits>

#define _USE_MATH_DEFINES
#include <math.h>

#include <ctime>
#include <chrono>

#include <Eigen/Dense>

#include "fileHandlers.h"
#include "pcaFunctions.h"
#include "colorDistances.h"
#include "colorSpaceTransformations.h"
#include "colorSearchFunctions.h"


//pack RGB value into a single uint16_t
uint16_t packColor(double R, double G, double B) {

    R = std::round(std::clamp(R, 0.0, 255.0));
    G = std::round(std::clamp(G, 0.0, 255.0));
    B = std::round(std::clamp(B, 0.0, 255.0));

    uint16_t R16 = static_cast<uint16_t>(R * 31.0 / 255.0); // 5 bits
    uint16_t G16 = static_cast<uint16_t>(G * 63.0 / 255.0); // 6 bits
    uint16_t B16 = static_cast<uint16_t>(B * 31.0 / 255.0); // 5 bits

    uint16_t color = (R16 << 11) + (G16 << 5) + B16;

    return color;
}


int main(int argc, char* argv[]) {

    //char* inputFileArg = argv[1];
    //char* outputFile = argv[2];
    //char* colorSpaceArg = argv[3];
    //char* distanceMetricArg = argv[4];

    char* inputFileArg = "C:/faks/diplomska/image1.jpg";
    char* outputFileArg = "C:/faks/diplomska/output1.dds";
    char* colorSpaceArg = "HSV";
    char* distanceMetricArg = "L2";

    //set color space for color optimisation
    ColorSpace colorSpace;
    Eigen::Vector3d(*colorTransformFunction)(const Eigen::Vector3d&);
    if (strcmp(colorSpaceArg, "RGB") == 0) {
        std::cout << "Using RGB color space for optimisation" << std::endl;
        colorSpace = RGB;
        colorTransformFunction = RGB2RGB;
    }
    else if (strcmp(colorSpaceArg, "HSV") == 0) {
        std::cout << "Using HSV color space for optimisation" << std::endl;
        colorSpace = HSV;
        colorTransformFunction = RGB2HSV;
    }
    else if (strcmp(colorSpaceArg, "CIELAB") == 0) {
        std::cout << "Using CIELAB color space for optimisation" << std::endl;
        colorSpace = CIELAB;
        colorTransformFunction = RGB2CIELAB;
    }
    else {
        std::cout << "Color space argument invalid, color space is set to RGB." << std::endl;
        colorTransformFunction = RGB2RGB;
    }

    //set distance functions for color optimisation
    double (*distanceFunction)(const Eigen::Vector3d&, const Eigen::Vector3d&);
    double (*distanceFunctionBlock)(const Eigen::MatrixXd&, const Eigen::MatrixXd&);
    if (strcmp(distanceMetricArg, "L1") == 0) {
        if (colorSpace == HSV) { //HSV is a special case, since Hue is circular
            distanceFunction = distanceL1HSV;
            distanceFunctionBlock = distanceL1HSVBlock;
        }
        else {
            distanceFunction = distanceL1;
            distanceFunctionBlock = distanceL1Block;
        }
        std::cout << "Using L1 metric for optimisation" << std::endl;
    }
    else if(strcmp(distanceMetricArg, "L2") == 0) {
        if (colorSpace == HSV) { //HSV is a special case, since Hue is circular
            distanceFunction = distanceL2HSV;
            distanceFunctionBlock = distanceL2HSVBlock;
        }
        else {
            distanceFunction = distanceL2;
            distanceFunctionBlock = distanceL2Block;
        }
        std::cout << "Using L2 metric for optimisation" << std::endl;
    }
    else if (strcmp(distanceMetricArg, "weightedL1") == 0) {
        if (colorSpace == HSV) { //HSV is a special case, since Hue is circular
            distanceFunction = weightedDistanceL1HSV;
            distanceFunctionBlock = weightedDistanceL1HSVBlock;
            std::cout << "Using weighted L1 metric for optimisation" << std::endl;
        }
        else if(colorSpace == RGB){
            distanceFunction = weightedDistanceL1RGB;
            distanceFunctionBlock = weightedDistanceL1RGBBlock;
            std::cout << "Using weighted L1 metric for optimisation" << std::endl;
        }
        else {
            distanceFunction = distanceL1;
            distanceFunctionBlock = distanceL1Block;
            std::cout << "Weighted metric only applies to RGB and HSV, using unweighted L1" << std::endl;
        }

    }
    else if (strcmp(distanceMetricArg, "weightedL2") == 0) {
        if (colorSpace == HSV) { //HSV is a special case, since Hue is circular
            distanceFunction = weightedDistanceL2HSV;
            distanceFunctionBlock = weightedDistanceL2HSVBlock;
            std::cout << "Using weighted L2 metric for optimisation" << std::endl;
        }
        else if (colorSpace == RGB) {
            distanceFunction = weightedDistanceL2RGB;
            distanceFunctionBlock = weightedDistanceL2RGBBlock;
            std::cout << "Using weighted L2 metric for optimisation" << std::endl;
        }
        else {
            distanceFunction = distanceL2;
            distanceFunctionBlock = distanceL2Block;
            std::cout << "Weighted metric only applies to RGB and HSV, using unweighted L2" << std::endl;
        }
    }
    else if (strcmp(distanceMetricArg, "default") == 0) {
        distanceFunction = nullptr;
        distanceFunctionBlock = nullptr;
        std::cout << "Using default metric for optimisation" << std::endl;
    }
    else {
        distanceFunction = nullptr;
        distanceFunctionBlock = nullptr;
        std::cout << "Distance metric argument invald" << std::endl;
    }

	int width, height, channels;

    uint8_t* data = openImage(inputFileArg, width, height, channels);

	std::cout << "width: " << width << std::endl;
	std::cout << "height: " << height << std::endl;
	std::cout << "channels: " << channels << std::endl;
	std::cout << "size: " << width * height * channels << std::endl;

    int blockCountHorizontal = (width + 3) / 4;
    int blockCountVertical = (height + 3) / 4;


    std::vector<uint8_t> compressedData(blockCountHorizontal * blockCountVertical * 8); // DXT1 compressed data size

    //measure compression time
    auto start = std::chrono::high_resolution_clock::now();

    double costSum = 0;
    double costImprovementSum = 0; //testing

    for (int blockI = 0; blockI < blockCountVertical; blockI++) {

        std::cout << "block row: " << blockI << std::endl;

        for (int blockJ = 0; blockJ < blockCountHorizontal; blockJ++) {

            int pixelCounter = 0;

            //pack pixels of a 4x4 block into a matrix
            Eigen::MatrixXd PCAdata(16, channels);
            for (int offsetI = 0; offsetI < 4; offsetI++) {
                for (int offsetJ = 0; offsetJ < 4; offsetJ++) {

                    int pixelIndex = (blockI * 4 + offsetI) * width + (blockJ * 4 + offsetJ);

                    if (!((blockI * 4 + offsetI) >= height || (blockJ * 4 + offsetJ) >= width)) {
                        PCAdata(pixelCounter, 0) = data[pixelIndex * channels + 0];
                        PCAdata(pixelCounter, 1) = data[pixelIndex * channels + 1];
                        PCAdata(pixelCounter, 2) = data[pixelIndex * channels + 2];

                        pixelCounter += 1;
                    }
                }
            }

            //find optimal colors for block type 1
            double cost_block1;
            double costImprovement_block1; //testing
            Eigen::Vector3d mean_block1;
            
            auto [color0_block1, color1_block1] = findOptimalColorsLinear(PCAdata, distanceFunction, distanceFunctionBlock, colorSpace, 1, &mean_block1, &cost_block1, &costImprovement_block1);


            //find optimal colors for block type 2
            double cost_block2;
            double costImprovement_block2; //testing
            Eigen::Vector3d mean_block2;

            auto [color0_block2, color1_block2] = findOptimalColorsLinear(PCAdata, distanceFunction, distanceFunctionBlock, colorSpace, 2, &mean_block2, &cost_block2, &costImprovement_block2);
            
            //choose best block type
            int bestType;
            Eigen::Vector3d mean, color0, color1;
            if (cost_block1 <= cost_block2) {
                bestType = 1;
                mean = mean_block1;
                color0 = color0_block1;
                color1 = color1_block1;
                costSum += cost_block1;
                costImprovementSum += costImprovement_block1; //testing
            }
            else {
                bestType = 2;
                mean = mean_block2;
                color0 = color0_block2;
                color1 = color1_block2;
                costSum += cost_block2;
                costImprovementSum += costImprovement_block2; //testing
            }

            std::cout << "optimalType: " << bestType << std::endl;

            uint16_t color0Packed = packColor(color0(0), color0(1), color0(2));
            uint16_t color1Packed = packColor(color1(0), color1(1), color1(2));

            //color0 should be > color1 for block type 1 and color0 < color1 for block type 2
            if ((color0Packed <= color1Packed && bestType == 1) || (color0Packed > color1Packed && bestType == 2)) {
                Eigen::Vector3d temp = color0;
                color0 = color1;
                color1 = temp;

                uint16_t temp1 = color0Packed;
                color0Packed = color1Packed;
                color1Packed = temp1;
            }


            //interpolate to get color2 and color3 (different depending on block type)
            Eigen::Vector3d color2;
            Eigen::Vector3d color3;
            if (bestType == 1) {
                color2 = (2 * color0 + color1) / 3;
                color3 = (color0 + 2 * color1) / 3;
            }
            else {
                color2 = (color0 + color1) / 2;
                color3 << 0, 0, 0;
            }

            color0 = colorTransformFunction(color0);
            color1 = colorTransformFunction(color1);
            color2 = colorTransformFunction(color2);
            color3 = colorTransformFunction(color3);

            //for each pixel in the block, find the closest color and encode it as a 2-bit value
            std::vector<uint8_t> pixelEncodings(16, 0);

            for (int offsetI = 0; offsetI < 4; offsetI++) {
                for (int offsetJ = 0; offsetJ < 4; offsetJ++) {

                    int pixelIndex = (blockI * 4 + offsetI) * width + (blockJ * 4 + offsetJ);

                    Eigen::Vector3d pixelColor;
                    if ((blockI * 4 + offsetI) >= height || (blockJ * 4 + offsetJ) >= width) {
                        pixelColor << mean;
                    }
                    else {
                        pixelColor << data[pixelIndex * channels + 0], data[pixelIndex * channels + 1], data[pixelIndex * channels + 2];
                    }

                    pixelColor = colorTransformFunction(pixelColor);

                    if (color0Packed == color1Packed) {
                        pixelEncodings[offsetI * 4 + offsetJ] = 0x00;
                        continue;
                    }

                    double minDist = distanceFunction(pixelColor, color0);
                    pixelEncodings[offsetI * 4 + offsetJ] = 0x00;

                    double dist;

                    dist = distanceFunction(pixelColor, color1);

                    if (dist <= minDist) {
                        minDist = dist;
                        pixelEncodings[offsetI * 4 + offsetJ] = 0x01;
                    }

                    dist = distanceFunction(pixelColor, color2);
                    if (dist <= minDist) {
                        minDist = dist;
                        pixelEncodings[offsetI * 4 + offsetJ] = 0x02;
                    }

                    dist = distanceFunction(pixelColor, color3);
                    if (dist <= minDist) {
                        minDist = dist;
                        pixelEncodings[offsetI * 4 + offsetJ] = 0x03;
                    }
                }

            }


            int blockIndex = (blockI * blockCountHorizontal + blockJ) * 8;

            //c0_lo, c0_hi, c1_lo, c1_hi, bits_0, bits_1, bits_2, bits_3

            compressedData[blockIndex + 0] = color0Packed & 0xFF;; //lower 8 bits of color0
            compressedData[blockIndex + 1] = (color0Packed >> 8) & 0xFF; //higher 8 bits of color0
            compressedData[blockIndex + 2] = color1Packed & 0xFF;; //lower 8 bits of color1
            compressedData[blockIndex + 3] = (color1Packed >> 8) & 0xFF; //higher 8 bits of color1

            //pixel encodings
            compressedData[blockIndex + 4] = (pixelEncodings[3] << 6) | (pixelEncodings[2] << 4) | (pixelEncodings[1] << 2) | pixelEncodings[0];
            compressedData[blockIndex + 5] = (pixelEncodings[7] << 6) | (pixelEncodings[6] << 4) | (pixelEncodings[5] << 2) | pixelEncodings[4];
            compressedData[blockIndex + 6] = (pixelEncodings[11] << 6) | (pixelEncodings[10] << 4) | (pixelEncodings[9] << 2) | pixelEncodings[8];
            compressedData[blockIndex + 7] = (pixelEncodings[15] << 6) | (pixelEncodings[14] << 4) | (pixelEncodings[13] << 2) | pixelEncodings[12];

        }
    }

    std::cout << "costAvg: " << costSum / (blockCountHorizontal * blockCountVertical) << std::endl;
    std::cout << "costImprovementAvg: " << costImprovementSum / (blockCountHorizontal * blockCountVertical) << std::endl;

    //measure encoding time
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    std::cout << "duration: " << duration.count() << std::endl;

    //frite compressed image into a file 
    if (WriteDDS_DXT1(outputFileArg, compressedData, width, height)) {
        std::cout << "DDS file with DXT1 compression written successfully.\n";
    }
    else {
        std::cout << "Failed to write DDS file.\n";
    }

	return 0;
}