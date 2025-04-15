#include <iostream>
#include <Eigen/Dense>
#include <random>
#include <ctime>
#include <chrono>

#include "pcaFunctions.h"
#include "colorDistances.h"
#include "colorSpaceTransformations.h"

enum ColorSpace {
    RGB,
    CIELAB,
    HSV
};

/*double computeCumulativeDistance(Eigen::VectorXd data, double val0, double val1) {
    double val2 = (2 * val0 + val1) / 3;
    double val3 = (val0 + 2 * val1) / 3;

    double totalDistance = 0.0;

    for (int i = 0; i < 16; i++) {
        double dist = std::min({ std::abs(data[i] - val0), std::abs(data[i] - val1), std::abs(data[i] - val2), std::abs(data[i] - val3) });
        totalDistance += dist;
    }

    return totalDistance;
}*/

Eigen::MatrixXd transformToClosestColors(const Eigen::MatrixXd& colorData, Eigen::Vector3d color0, Eigen::Vector3d color1, double (*distanceFunction)(const Eigen::Vector3d&, const Eigen::Vector3d&), int blockType) {
    Eigen::MatrixXd transformedData(colorData.rows(), colorData.cols());

    // Precompute intermediate colors
    Eigen::Vector3d color2;
    Eigen::Vector3d color3;
    if (blockType == 1) {
        color2 = (2 * color0 + color1) / 3;
        color3 = (color0 + 2 * color1) / 3;
    }
    else if (blockType == 2){
        color2 = (color0 + color1) / 2;
        color3 << 0, 0, 0;
    }
    else {
        std::cout << "Invalid block type" <<"\n";
        exit(1);
    }

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


std::pair<Eigen::Vector3d, Eigen::Vector3d> findOptimalColorsCooling(
    const Eigen::MatrixXd& RGBData,
    double (*distanceFunction)(const Eigen::Vector3d&, const Eigen::Vector3d&),
    double (*distanceFunctionBlock)(const Eigen::MatrixXd&, const Eigen::MatrixXd&),
    ColorSpace colorSpace,
    int blockType,
    Eigen::Vector3d* meanReturn,
    double* cost, //distance between uncompressed and compressed block
    double* costImprovement //difference between initial and optimised color solutions
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
    Eigen::Vector3d col0 = colorTransformFunction((bestVal0 * eigenvector + mean));
    Eigen::Vector3d col1 = colorTransformFunction((bestVal1 * eigenvector + mean));
    Eigen::MatrixXd closestColorData = transformToClosestColors(searchSpaceData, col0, col1, distanceFunction, blockType);
    currentCost = distanceFunctionBlock(searchSpaceData, closestColorData);

    double bestCost = currentCost;

    double initialCost = currentCost;

    // Simulated annealing loop
    double temperature = initialTemperature;
    for (int i = 0; i < maxIterations; i++) {

        double newVal0 = val0 + dist(rng) * randomStep - randomStep / 2; // Small random adjustment
        double newVal1 = val1 + dist(rng) * randomStep - randomStep / 2;

        if (newVal0 > newVal1) {
            double temp = newVal0;
            newVal0 = newVal1;
            newVal1 = temp;
        }

        // Clamp the values to the search range
        newVal0 = std::clamp(newVal0, minRange, maxRange);
        newVal1 = std::clamp(newVal1, minRange, maxRange);

        double newCost;
        Eigen::Vector3d col0 = colorTransformFunction((newVal0 * eigenvector + mean));
        Eigen::Vector3d col1 = colorTransformFunction((newVal1 * eigenvector + mean));
        Eigen::MatrixXd closestColorData = transformToClosestColors(searchSpaceData, col0, col1, distanceFunction, blockType);
        newCost = distanceFunctionBlock(searchSpaceData, closestColorData);

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

    Eigen::Vector3d color0 = (bestVal0 * eigenvector + mean);
    Eigen::Vector3d color1 = (bestVal1 * eigenvector + mean);

    *cost = bestCost;
    *costImprovement = initialCost - bestCost;


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

std::pair<Eigen::Vector3d, Eigen::Vector3d> findOptimalColorsLinear(
    const Eigen::MatrixXd& RGBData,
    double (*distanceFunction)(const Eigen::Vector3d&, const Eigen::Vector3d&),
    double (*distanceFunctionBlock)(const Eigen::MatrixXd&, const Eigen::MatrixXd&),
    ColorSpace colorSpace,
    int blockType,
    Eigen::Vector3d* meanReturn,
    double* cost, //distance between uncompressed and compressed block
    double* costImprovement //difference between initial and optimised color solutions
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
    int resolution = 400;


    // Initial solution
    double val0 = minRange;
    double val1 = maxRange;
    double bestVal0 = val0, bestVal1 = val1;

    double currentCost;
    Eigen::Vector3d col0 = colorTransformFunction((bestVal0 * eigenvector + mean));
    Eigen::Vector3d col1 = colorTransformFunction((bestVal1 * eigenvector + mean));
    Eigen::MatrixXd closestColorData = transformToClosestColors(searchSpaceData, col0, col1, distanceFunction, blockType);
    currentCost = distanceFunctionBlock(searchSpaceData, closestColorData);

    double bestCost = currentCost;

    //std::cout << "initial cost: " << currentCost <<"\n";
    double initialCost = currentCost;

    double stepSize = (maxRange - minRange) / resolution;

    //search entire space for the optimal solution
    for (int a = 0; a < resolution; a++) {
        for (int b = a; b < resolution; b++) {
            val0 = minRange + a * stepSize;
            val1 = minRange + b * stepSize;

            double newCost;
            Eigen::Vector3d col0 = colorTransformFunction((val0 * eigenvector + mean));
            Eigen::Vector3d col1 = colorTransformFunction((val1 * eigenvector + mean));
            Eigen::MatrixXd closestColorData = transformToClosestColors(searchSpaceData, col0, col1, distanceFunction, blockType);
            newCost = distanceFunctionBlock(searchSpaceData, closestColorData);

            if (newCost < bestCost) {
                bestVal0 = val0;
                bestVal1 = val1;
                bestCost = newCost;
            }
        }
    }

    Eigen::Vector3d color0 = (bestVal0 * eigenvector + mean);
    Eigen::Vector3d color1 = (bestVal1 * eigenvector + mean);

    *cost = bestCost;
    *costImprovement = initialCost - bestCost;


    return { color0, color1 };
}

std::pair<Eigen::Vector3d, Eigen::Vector3d> findOptimalColorsHybrid(
    const Eigen::MatrixXd& RGBData,
    double (*distanceFunction)(const Eigen::Vector3d&, const Eigen::Vector3d&),
    double (*distanceFunctionBlock)(const Eigen::MatrixXd&, const Eigen::MatrixXd&),
    ColorSpace colorSpace,
    int blockType,
    Eigen::Vector3d* meanReturn,
    double* cost, //distance between uncompressed and compressed block
    double* costImprovement //difference between initial and optimised color solutions
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
    int resolution = 50;


    // Initial solution
    double val0 = minRange;
    double val1 = maxRange;
    double bestVal0 = val0, bestVal1 = val1;

    double currentCost;
    Eigen::Vector3d col0 = colorTransformFunction((bestVal0 * eigenvector + mean));
    Eigen::Vector3d col1 = colorTransformFunction((bestVal1 * eigenvector + mean));
    Eigen::MatrixXd closestColorData = transformToClosestColors(searchSpaceData, col0, col1, distanceFunction, blockType);
    currentCost = distanceFunctionBlock(searchSpaceData, closestColorData);

    double bestCost = currentCost;

    //std::cout << "initial cost: " << currentCost <<"\n";
    double initialCost = currentCost;

    double stepSize = (maxRange - minRange) / resolution;

    //search entire space for aproximate solution
    for (int a = 0; a < resolution; a++) {
        for (int b = a; b < resolution; b++) {
            val0 = minRange + a * stepSize;
            val1 = minRange + b * stepSize;

            double newCost;
            Eigen::Vector3d col0 = colorTransformFunction((val0 * eigenvector + mean));
            Eigen::Vector3d col1 = colorTransformFunction((val1 * eigenvector + mean));
            Eigen::MatrixXd closestColorData = transformToClosestColors(searchSpaceData, col0, col1, distanceFunction, blockType);
            newCost = distanceFunctionBlock(searchSpaceData, closestColorData);

            if (newCost < bestCost) {
                bestVal0 = val0;
                bestVal1 = val1;
                bestCost = newCost;
            }
        }
    }

    //use aproximate  solution as a start for simulated annealing
    val0 = bestVal0;
    val1 = bestVal1;
    currentCost = bestCost;

    //random number generation
    std::mt19937 rng(static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count()));
    std::uniform_real_distribution<double> dist(minRange, maxRange);
    std::uniform_real_distribution<double> probDist(0.0, 1.0);

    double initialTemperature = 2.0;
    double coolingRate = 0.85;
    int maxIterations = 40;
    double randomStep = 0.03;

    // Simulated annealing loop
    double temperature = initialTemperature;
    for (int i = 0; i < maxIterations; i++) {

        double newVal0 = val0 + dist(rng) * randomStep - randomStep / 2; // Small random adjustment
        double newVal1 = val1 + dist(rng) * randomStep - randomStep / 2;

        if (newVal0 > newVal1) {
            double temp = newVal0;
            newVal0 = newVal1;
            newVal1 = temp;
        }

        // Clamp the values to the search range
        newVal0 = std::clamp(newVal0, minRange, maxRange);
        newVal1 = std::clamp(newVal1, minRange, maxRange);

        double newCost;
        Eigen::Vector3d col0 = colorTransformFunction((newVal0 * eigenvector + mean));
        Eigen::Vector3d col1 = colorTransformFunction((newVal1 * eigenvector + mean));
        Eigen::MatrixXd closestColorData = transformToClosestColors(searchSpaceData, col0, col1, distanceFunction, blockType);
        newCost = distanceFunctionBlock(searchSpaceData, closestColorData);

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

    Eigen::Vector3d color0 = (bestVal0 * eigenvector + mean);
    Eigen::Vector3d color1 = (bestVal1 * eigenvector + mean);

    *cost = bestCost;
    *costImprovement = initialCost - bestCost;


    return { color0, color1 };
}