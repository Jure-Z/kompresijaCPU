#include <iostream>

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
#include <random>
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


enum ColorSpace {
    RGB,
    CIELAB,
    HSV
};


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


double computeCumulativeDistance(Eigen::VectorXd data, double val0, double val1) {
    double val2 = (2 * val0 + val1) / 3;
    double val3 = (val0 + 2 * val1) / 3;

    double totalDistance = 0.0;

    for (int i = 0; i < 16; i++) {
        double dist = std::min({ std::abs(data[i] - val0), std::abs(data[i] - val1), std::abs(data[i] - val2), std::abs(data[i] - val3)});
        totalDistance += dist;
    }

    return totalDistance;
}


Eigen::MatrixXd transformToClosestColors(const Eigen::MatrixXd& colorData, Eigen::Vector3d color0, Eigen::Vector3d color1, double (*distanceFunction)(const Eigen::Vector3d&, const Eigen::Vector3d&)) {
    Eigen::MatrixXd transformedData(colorData.rows(), colorData.cols());

    // Precompute intermediate colors
    Eigen::Vector3d color2 = (2 * color0 + color1) / 3;
    Eigen::Vector3d color3 = (color0 + 2 * color1) / 3;

    // Store all colors in a vector for easy comparison
    std::vector<Eigen::Vector3d> palette = { color0, color1, color2, color3 };

    for (int i = 0; i < colorData.rows(); i++) {
        Eigen::Vector3d pixelColor = colorData.row(i);

        // Find the closest color
        auto closestColor = std::min_element(
            palette.begin(), palette.end(),
            [&](const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
                return distanceFunction(pixelColor, a) < distanceFunction(pixelColor, b);
            });

        transformedData.row(i) = *closestColor;
    }

    return transformedData;
}


//any color must be inside the defined color space. If we are searching along the line defined as y = x*v + m, where v is eigenvector and m is mean,
//the function returns the lower and upper limit for x. lowerBoundary an upperBoundary reffer to the color space boundaries
std::pair<double, double> findSearchSpaceBounds(const Eigen::Vector3d& mean, const Eigen::Vector3d& eigenvector, double lowerBoundary, double upperBoundary) {

    double candidates[] = {
        (lowerBoundary - mean[0]) / eigenvector[0],
        (upperBoundary - mean[0]) / eigenvector[0],
        (lowerBoundary - mean[1]) / eigenvector[1],
        (upperBoundary - mean[1]) / eigenvector[1],
        (lowerBoundary - mean[2]) / eigenvector[2],
        (upperBoundary - mean[2]) / eigenvector[2]
    };

    double lower = -1e20;
    double upper = 1e20;

    for (double candidate : candidates) {
        if (candidate < 0) {
            if (candidate > lower)
                lower = candidate;
        }
        else {
            if (candidate < upper)
                upper = candidate;
        }
    }

    return { lower, upper };
 }

std::pair<Eigen::Vector3d, Eigen::Vector3d> findOptimalColors(
    const Eigen::MatrixXd& RGBData,
    double (*distanceFunction)(const Eigen::Vector3d&, const Eigen::Vector3d&),
    double (*distanceFunctionBlock)(const Eigen::MatrixXd&, const Eigen::MatrixXd&),
    ColorSpace colorSpace,
    Eigen::Vector3d* meanReturn,
    double* cost
) {

    //transform color data into a different color space
    int matrixRows = RGBData.rows();
    Eigen::MatrixXd searchSpaceData;
    Eigen::Vector3d (*colorTransformFunction)(const Eigen::Vector3d&);
    if (colorSpace == HSV) {
        searchSpaceData = RGB2HSVBlock(RGBData);
        colorTransformFunction = RGB2HSV;
    }
    else if (colorSpace == CIELAB) {
        searchSpaceData = RGB2CIELABBlock(RGBData);
        colorTransformFunction = RGB2CIELAB;
    }
    else {
        searchSpaceData = RGBData;
        colorTransformFunction = RGB2RGB;
    }

    //auto [color0, color1] = getColorProjections_fullPCA(PCAdata);
    //auto [color0, color1] = getColorProjections_LargestEigenvector(PCAdata);

    Eigen::Vector3d mean;
    Eigen::MatrixXd centeredData;


    Eigen::Vector3d eigenvector = getEigenvector_PowerIteration(RGBData, &mean, &centeredData);

    *meanReturn = mean;

    if (eigenvector.norm() == 0) {//entire block has the same color
        return { RGBData.row(0), RGBData.row(0) };
    }

    Eigen::MatrixXd projectionLenghts = centeredData * eigenvector / eigenvector.norm();

    auto [lowerBound, upperBound] = findSearchSpaceBounds(mean, eigenvector, 0, 255); //find boundaries for search space

    // Define parameters for simulated annealing
    double minRange = std::max(projectionLenghts.minCoeff(), lowerBound);
    double maxRange = std::min(projectionLenghts.maxCoeff(), upperBound);
    double initialTemperature = 5.0;
    double coolingRate = 0.70;
    int maxIterations = 60;
    double randomStep = 0.5;

    //random number generation
    std::mt19937 rng(static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count()));
    std::uniform_real_distribution<double> dist(minRange, maxRange);
    std::uniform_real_distribution<double> probDist(0.0, 1.0);


    // Initial solution
    double val0 = minRange;
    double val1 = maxRange;
    double bestVal0 = val0, bestVal1 = val1;

    double currentCost;
    if (distanceFunction == nullptr || distanceFunctionBlock == nullptr) { //use default distance metric
        currentCost = computeCumulativeDistance(projectionLenghts, val0, val1);
    }
    else {
        Eigen::Vector3d col0 = colorTransformFunction((bestVal0 * eigenvector + mean));
        Eigen::Vector3d col1 = colorTransformFunction((bestVal1 * eigenvector + mean));
        Eigen::MatrixXd closestColorData = transformToClosestColors(searchSpaceData, col0, col1, distanceFunction);

        currentCost = distanceFunctionBlock(searchSpaceData, closestColorData);
    }

    double bestCost = currentCost;

    //std::cout << "initial cost: " << currentCost <<"\n";
    double initialCost = currentCost;

    // Simulated annealing loop
    double temperature = initialTemperature;
    for (int i = 0; i < maxIterations; i++) {

        double newVal0 = val0 + dist(rng) * randomStep - randomStep/2; // Small random adjustment
        double newVal1 = val1 + dist(rng) * randomStep - randomStep/2;

        if (newVal0 > newVal1) {
            double temp = newVal0;
            newVal0 = newVal1;
            newVal1 = temp;
        }

        // Clamp the values to the search range
        newVal0 = std::clamp(newVal0, minRange, maxRange);
        newVal1 = std::clamp(newVal1, minRange, maxRange);

        double newCost;
        if (distanceFunction == nullptr || distanceFunctionBlock == nullptr) { //use default distance metric
            newCost = computeCumulativeDistance(projectionLenghts, newVal0, newVal1);
        }
        else {
            Eigen::Vector3d col0 = colorTransformFunction((newVal0 * eigenvector + mean));
            Eigen::Vector3d col1 = colorTransformFunction((newVal1 * eigenvector + mean));
            Eigen::MatrixXd closestColorData = transformToClosestColors(searchSpaceData, col0, col1, distanceFunction);

            newCost = distanceFunctionBlock(searchSpaceData, closestColorData);
        }

        double acceptanceProbability = std::exp((currentCost - newCost) / temperature);

        // Accept or reject the new solution
        if (/*newCost < currentCost || */ probDist(rng) < acceptanceProbability) {
            val0 = newVal0;
            val1 = newVal1;
            currentCost = newCost;

            // Update the best solution
            if (newCost < bestCost) {
                bestVal0 = newVal0;
                bestVal1 = newVal1;
                bestCost = newCost;
            }
        }

        // Cool down the temperature
        temperature *= coolingRate;
    }

    if (distanceFunction == nullptr || distanceFunctionBlock == nullptr) { //use default distance metric
        *cost = initialCost - computeCumulativeDistance(projectionLenghts, bestVal0, bestVal1);
    }
    else {
        Eigen::Vector3d col0 = colorTransformFunction((bestVal0 * eigenvector + mean));
        Eigen::Vector3d col1 = colorTransformFunction((bestVal1 * eigenvector + mean));
        Eigen::MatrixXd closestColorData = transformToClosestColors(searchSpaceData, col0, col1, distanceFunction);

        *cost = initialCost - distanceFunctionBlock(searchSpaceData, closestColorData);
    }

    Eigen::Vector3d color0 = (bestVal0 * eigenvector + mean);
    Eigen::Vector3d color1 = (bestVal1 * eigenvector + mean);


    //transform color back into RGB space
    /*if (colorSpace == HSV) {
        color0 = HSV2RGB(color0);
        color1 = HSV2RGB(color1);
    }
    else if (colorSpace == CIELAB) {
        color0 = CIELAB2RGB(color0);
        color1 = CIELAB2RGB(color1);
    }*/

    return { color0, color1 };
}

std::pair<Eigen::Vector3d, Eigen::Vector3d> findOptimalColors1(
    const Eigen::MatrixXd& RGBData,
    double (*distanceFunction)(const Eigen::Vector3d&, const Eigen::Vector3d&),
    double (*distanceFunctionBlock)(const Eigen::MatrixXd&, const Eigen::MatrixXd&),
    ColorSpace colorSpace,
    Eigen::Vector3d* meanReturn,
    double* cost
) {

    //transform color data into a different color space
    int matrixRows = RGBData.rows();
    Eigen::MatrixXd searchSpaceData;
    Eigen::Vector3d(*colorTransformFunction)(const Eigen::Vector3d&);
    if (colorSpace == HSV) {
        searchSpaceData = RGB2HSVBlock(RGBData);
        colorTransformFunction = RGB2HSV;
    }
    else if (colorSpace == CIELAB) {
        searchSpaceData = RGB2CIELABBlock(RGBData);
        colorTransformFunction = RGB2CIELAB;
    }
    else {
        searchSpaceData = RGBData;
        colorTransformFunction = RGB2RGB;
    }

    //auto [color0, color1] = getColorProjections_fullPCA(PCAdata);
    //auto [color0, color1] = getColorProjections_LargestEigenvector(PCAdata);

    Eigen::Vector3d mean;
    Eigen::MatrixXd centeredData;


    Eigen::Vector3d eigenvector = getEigenvector_PowerIteration(RGBData, &mean, &centeredData);

    *meanReturn = mean;

    if (eigenvector.norm() == 0) {//entire block has the same color
        return { RGBData.row(0), RGBData.row(0) };
    }

    Eigen::MatrixXd projectionLenghts = centeredData * eigenvector / eigenvector.norm();

    auto [lowerBound, upperBound] = findSearchSpaceBounds(mean, eigenvector, 0, 255); //find boundaries for search space

    // Define parameters for simulated annealing
    double minRange = std::max(projectionLenghts.minCoeff(), lowerBound);
    double maxRange = std::min(projectionLenghts.maxCoeff(), upperBound);
    int resolution = 100;


    // Initial solution
    double val0 = minRange;
    double val1 = maxRange;
    double bestVal0 = val0, bestVal1 = val1;

    double currentCost;
    if (distanceFunction == nullptr || distanceFunctionBlock == nullptr) { //use default distance metric
        currentCost = computeCumulativeDistance(projectionLenghts, val0, val1);
    }
    else {
        Eigen::Vector3d col0 = colorTransformFunction((bestVal0 * eigenvector + mean));
        Eigen::Vector3d col1 = colorTransformFunction((bestVal1 * eigenvector + mean));
        Eigen::MatrixXd closestColorData = transformToClosestColors(searchSpaceData, col0, col1, distanceFunction);

        currentCost = distanceFunctionBlock(searchSpaceData, closestColorData);
    }

    double bestCost = currentCost;

    //std::cout << "initial cost: " << currentCost <<"\n";
    double initialCost = currentCost;

    double stepSize = (maxRange - minRange) / resolution;

    //search entire space for the optimal solution
    for (int a = 0; a < resolution; a++) {
        for(int b = a; b < resolution; b++) {
            val0 = minRange + a * stepSize;
            val1 = minRange + b * stepSize;

            double newCost;
            if (distanceFunction == nullptr || distanceFunctionBlock == nullptr) { //use default distance metric
                newCost = computeCumulativeDistance(projectionLenghts, val0, val1);
            }
            else {
                Eigen::Vector3d col0 = colorTransformFunction((val0 * eigenvector + mean));
                Eigen::Vector3d col1 = colorTransformFunction((val1 * eigenvector + mean));
                Eigen::MatrixXd closestColorData = transformToClosestColors(searchSpaceData, col0, col1, distanceFunction);

                newCost = distanceFunctionBlock(searchSpaceData, closestColorData);
            }

            if (newCost < bestCost) {
                bestVal0 = val0;
                bestVal1 = val1;
                bestCost = newCost;
            }
        }
    }

    if (distanceFunction == nullptr || distanceFunctionBlock == nullptr) { //use default distance metric
        *cost = initialCost - computeCumulativeDistance(projectionLenghts, bestVal0, bestVal1);
    }
    else {
        Eigen::Vector3d col0 = colorTransformFunction((bestVal0 * eigenvector + mean));
        Eigen::Vector3d col1 = colorTransformFunction((bestVal1 * eigenvector + mean));
        Eigen::MatrixXd closestColorData = transformToClosestColors(searchSpaceData, col0, col1, distanceFunction);

        *cost = initialCost - distanceFunctionBlock(searchSpaceData, closestColorData);
    }

    Eigen::Vector3d color0 = (bestVal0 * eigenvector + mean);
    Eigen::Vector3d color1 = (bestVal1 * eigenvector + mean);


    return { color0, color1 };
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
    if (strcmp(colorSpaceArg, "RGB") == 0) {
        std::cout << "Using RGB color space for optimisation" << std::endl;
        colorSpace = RGB;
    }
    else if (strcmp(colorSpaceArg, "HSV") == 0) {
        std::cout << "Using HSV color space for optimisation" << std::endl;
        colorSpace = HSV;
    }
    else if (strcmp(colorSpaceArg, "CIELAB") == 0) {
        std::cout << "Using CIELAB color space for optimisation" << std::endl;
        colorSpace = CIELAB;
    }
    else {
        std::cout << "Color space argument invalid, color space is set to RGB." << std::endl;
        colorSpace = RGB;
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
            std::cout << "Weighted metrica only apply to RGB and HSV, using unweighted L1" << std::endl;
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
            std::cout << "Weighted metrica only apply to RGB and HSV, using unweighted L2" << std::endl;
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
        std::cout << "Distance metric argument invald, using default metric for optimisation" << std::endl;
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

    double costSum = 0; //testing

    for (int blockI = 0; blockI < blockCountVertical; blockI++) {

        std::cout << "block row: " << blockI << std::endl;

        for (int blockJ = 0; blockJ < blockCountHorizontal; blockJ++) {

            int pixelCounter = 0;

            //pack pixels of a 4x4 block into a matrix
            Eigen::MatrixXd PCAdata(36, channels);
            for (int offsetI = -1; offsetI < 5; offsetI++) {
                for (int offsetJ = -1; offsetJ < 5; offsetJ++) {

                    int pixelIndex = (blockI * 4 + offsetI) * width + (blockJ * 4 + offsetJ);

                    if (!((blockI * 4 + offsetI) >= height || (blockJ * 4 + offsetJ) >= width)) {
                        PCAdata(pixelCounter, 0) = data[pixelIndex * channels + 0];
                        PCAdata(pixelCounter, 1) = data[pixelIndex * channels + 1];
                        PCAdata(pixelCounter, 2) = data[pixelIndex * channels + 2];

                        pixelCounter += 1;
                    }
                }
            }

            double cost;//testing
            Eigen::Vector3d mean;
            
            auto [color0, color1] = findOptimalColors1(PCAdata, distanceFunction, distanceFunctionBlock, colorSpace, &mean, &cost);
            

            costSum += cost; //testing

            uint16_t color0Packed = packColor(color0(0), color0(1), color0(2));
            uint16_t color1Packed = packColor(color1(0), color1(1), color1(2));

            //color0 should be > color1, to use block type 1 for encoding
            if (color0Packed <= color1Packed) {
                Eigen::Vector3d temp = color0;
                color0 = color1;
                color1 = temp;

                uint16_t temp1 = color0Packed;
                color0Packed = color1Packed;
                color1Packed = temp1;
            }


            //interpolate to get color2 and color3
            Eigen::Vector3d color2 = (2 * color0 + color1) / 3;
            Eigen::Vector3d color3 = (color0 + 2 * color1) / 3;

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

    std::cout << "costImprovementSum: " << costSum / (blockCountHorizontal * blockCountVertical) << std::endl;

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