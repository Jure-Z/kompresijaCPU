#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <iostream>
#include <cassert>
#include <vector>
#include <filesystem>
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

#include "colorDistances.h"
#include "colorSpaceTransformations.h"

namespace fs = std::filesystem;

enum ColorSpace {
    RGB,
    CIELAB,
    HSV
};

uint8_t* openImage(const char* path, int& width, int& height, int& channels) {

	uint8_t* pixelData = stbi_load(path, &width, &height, &channels, 0);

	if (!pixelData) {
		std::cout << "Could not upen image " << path << std::endl;
        std::cout << stbi_failure_reason() << std::endl;
	}

	return pixelData;
}

#pragma pack(push, 1)
// DDS file header structure
struct DDSHeader {
    uint32_t size;              // Header size (must be 124)
    uint32_t flags;             // Flags to indicate valid fields
    uint32_t height;            // Height of the texture
    uint32_t width;             // Width of the texture
    uint32_t pitchOrLinearSize; // Pitch or linear size
    uint32_t depth;             // Depth of the texture (3D textures)
    uint32_t mipMapCount;       // Number of mipmaps
    uint32_t reserved[11];      // Reserved fields
    struct {
        uint32_t size;          // Size of this structure (32 bytes)
        uint32_t flags;         // Flags to indicate valid fields
        uint32_t fourCC;        // FourCC code for compressed formats
        uint32_t rgbBitCount;   // Number of bits per pixel (unused for DXT1)
        uint32_t rBitMask;      // Red bit mask (unused for DXT1)
        uint32_t gBitMask;      // Green bit mask (unused for DXT1)
        uint32_t bBitMask;      // Blue bit mask (unused for DXT1)
        uint32_t aBitMask;      // Alpha bit mask (unused for DXT1)
    } pixelFormat;
    uint32_t caps[4];           // Capabilities
    uint32_t reserved2;         // Reserved
};
#pragma pack(pop)

//write DXT1-compressed data into a DDS file
bool WriteDDS_DXT1(const fs::path& path, const std::vector<uint8_t>& compressedData, uint32_t width, uint32_t height, uint32_t mipMapCount = 1) {
    // Open file for binary writing
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << path << "\n";
        return false;
    }

    // Write DDS magic number
    const uint32_t DDS_MAGIC = 0x20534444; // "DDS "
    file.write(reinterpret_cast<const char*>(&DDS_MAGIC), sizeof(DDS_MAGIC));

    // Calculate linear size of the top-level mipmap
    uint32_t blockSize = 8; // DXT1 compression uses 8 bytes per 4x4 block
    uint32_t linearSize = ((width + 3) / 4) * ((height + 3) / 4) * blockSize;

    // Create DDS header
    DDSHeader header = {};
    header.size = 124;
    header.flags = 0x00021007; // DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT | DDSD_LINEARSIZE
    header.height = height;
    header.width = width;
    header.pitchOrLinearSize = linearSize;
    header.mipMapCount = mipMapCount;
    header.pixelFormat.size = 32;
    header.pixelFormat.flags = 0x4; // DDPF_FOURCC
    header.pixelFormat.fourCC = 0x31545844; // "DXT1" in little-endian
    header.caps[0] = 0x1000; // DDSCAPS_TEXTURE
    if (mipMapCount > 1) {
        header.caps[0] |= 0x400000; // DDSCAPS_MIPMAP
        header.caps[1] = 0x8;      // DDSCAPS_COMPLEX
    }

    // Write header
    file.write(reinterpret_cast<const char*>(&header), sizeof(header));

    // Write compressed data
    file.write(reinterpret_cast<const char*>(compressedData.data()), compressedData.size());

    // Close file
    file.close();

    return true;
}


/*double determinant2x2(double mat_00, double mat_01, double mat_10, double mat_11) {
    return mat_00 * mat_11 - mat_01 * mat_10;
}

//solves cubic equation ax^3 + bx^2 + cx + d = 0. returns only the largest real solution.
double solveCharacteristicEquation(double a, double b, double c, double d) {
    // Normalize the cubic equation to get: x^3 + px^2 + qx + r = 0
    double p = b / a, q = c / a, r = d / a;

    //change equation into depressed cubic form: t^3 + At + B = 0,  x = t - p/3

    double A = (3.0 * q - std::pow(p, 2)) / 3.0;
    double B = (2.0 * std::pow(p, 3) - 9.0 * p * q + 27.0 * r) / 27.0;

    double discriminant = std::pow(A, 3) / 27.0 + std::pow(B, 2) / 4.0;

    if (discriminant > 0) {
        // One real root
        double C = std::sqrt(discriminant);
        double t_1 = std::cbrt(-B / 2.0 + C) + std::cbrt(-B / 2.0 - C);
        double x_1 = t_1 - p / 3.0;

        return x_1;
    }
    else {
        // Three real roots
        double theta = std::acos(-B / (2.0 * std::sqrt(-std::pow(A, 3) / 27.0)));
        double twoSqrtA = 2.0 * std::sqrt(-A / 3.0);

        std::vector<double> solutoins;

        for (int k = 0; k < 3; k++) {
            double t_k = twoSqrtA * std::cos((theta + 2.0 * M_PI * k) / 3.0);
            double x_k = t_k - p / 3.0;
            solutoins.push_back(x_k);
        }

        return *std::max_element(solutoins.begin(), solutoins.end());
    }
}

// Solve the cubic characteristic equation for eigenvalues
double largestEigenvalue(const Eigen::Matrix3d& A) {
    // Coefficients of the characteristic polynomial: det(A - λI) = 0
    double c2 = -A.trace();  // Sum of diagonal elements
    double c1 = 0.5 * (A.trace() * A.trace() - (A * A).trace());
    double c0 = -A.determinant();

    // Solve for the largest root of the cubic equation λ³ + c2λ² + c1λ + c0 = 0
    double q = (3 * c1 - c2 * c2) / 9.0;
    double r = (9 * c2 * c1 - 27 * c0 - 2 * c2 * c2 * c2) / 54.0;
    double discriminant = q * q * q + r * r;

    double lambda_max;
    if (discriminant >= 0) {
        // One real root (largest)
        double sqrt_discriminant = std::sqrt(discriminant);
        double s = std::cbrt(r + sqrt_discriminant);
        double t = std::cbrt(r - sqrt_discriminant);
        lambda_max = s + t - c2 / 3.0;
    }
    else {
        // Three real roots (use trigonometric solution for largest root)
        double theta = std::acos(r / std::sqrt(-q * q * q));
        lambda_max = 2 * std::sqrt(-q) * std::cos(theta / 3.0) - c2 / 3.0;
    }

    return lambda_max;
}

//returns eigenvector corresponding to the largest eigenvalue
Eigen::Vector3d eigenvectorForLargest(const Eigen::Matrix3d& A, double lambda_max) {
    // Subtract lambda_max * I from A
    Eigen::Matrix3d AMinusLambda = A - lambda_max * Eigen::Matrix3d::Identity();

    // Extract rows of AMinusLambda
    Eigen::Vector3d row0 = AMinusLambda.row(0);
    Eigen::Vector3d row1 = AMinusLambda.row(1);
    Eigen::Vector3d row2 = AMinusLambda.row(2);

    Eigen::Vector3d eigenvector;

    // Use cross product of any two rows that are not parallel to find the null space
    if (row0.cross(row1).norm() > 1e-6) {
        eigenvector = row0.cross(row1); // Cross product of row 0 and row 1
    }
    else if (row0.cross(row2).norm() > 1e-6) {
        eigenvector = row0.cross(row2); // Cross product of row 0 and row 2
    }
    else if (row1.cross(row2).norm() > 1e-6) {
        eigenvector = row1.cross(row2); // Cross product of row 1 and row 2
    }
    else {
        std::cout << AMinusLambda << "\n";
        std::cout << A << "\n";
        std::cout << lambda_max << "\n";
        throw std::runtime_error("Matrix is degenerate or numerical issues occurred.");
    }

    // Normalize the eigenvector
    eigenvector.normalize();

    return eigenvector;
}*/


//find the two best colors for encoding, using PCA (claculate only the largest eigenvector)
/*std::pair<Eigen::Vector3d, Eigen::Vector3d> getColorProjections_LargestEigenvector(const Eigen::MatrixXd& colorData) {

    //std::cout << "Data: \n" << rgbData << "\n";

    //Compute mean of each channel
    Eigen::VectorXd mean = colorData.colwise().mean();

    //Center the data
    Eigen::MatrixXd centeredData = colorData.rowwise() - mean.transpose();

    //Compute covariance matrix
    Eigen::MatrixXd covarianceMatrix = (centeredData.transpose() * centeredData) / (16 - 1);

    if (covarianceMatrix.norm() < 1e-6) { //if covariance matrix is all zeros, the entire block is the same color

        return { colorData.row(0), colorData.row(0) };
    }

    //characteristic polynomial of 3x3 matrix
    double a = 1.0; // Coefficient of λ^3
    double b = covarianceMatrix(0, 0) + covarianceMatrix(1, 1) + covarianceMatrix(2, 2); // Coefficient of λ^2
    double c = determinant2x2(covarianceMatrix(1, 1), covarianceMatrix(1, 2), covarianceMatrix(2, 1), covarianceMatrix(2, 2)) +
        determinant2x2(covarianceMatrix(0, 0), covarianceMatrix(0, 2), covarianceMatrix(2, 0), covarianceMatrix(2, 2)) +
        determinant2x2(covarianceMatrix(0, 0), covarianceMatrix(0, 1), covarianceMatrix(1, 0), covarianceMatrix(1, 1)); // Coefficient of λ^1
    double d = covarianceMatrix.determinant(); // Coefficient of λ^0

    double maxEigenvalue = solveCharacteristicEquation(a, -b, c, -d);

    Eigen::Vector3d eigenvector = eigenvectorForLargest(covarianceMatrix, maxEigenvalue);

    Eigen::MatrixXd projectionLenghts = centeredData * eigenvector / eigenvector.norm();


    double minProjection = projectionLenghts.minCoeff();
    double maxProjection = projectionLenghts.maxCoeff();

    Eigen::Vector3d color0 = (minProjection * eigenvector + mean).array().round();
    Eigen::Vector3d color1 = (maxProjection * eigenvector + mean).array().round();

    color0 = checkColorInsideBounds(color0, eigenvector);
    color1 = checkColorInsideBounds(color1, eigenvector);

    return { color0, color1 };
}*/

//find the two best colors for encoding, using PCA (preform entire PCA process)
/*std::pair<Eigen::Vector3d, Eigen::Vector3d> getColorProjections_fullPCA(const Eigen::MatrixXd& colorData) {

    //std::cout << "Data: \n" << rgbData << "\n";

    //Compute mean of each channel
    Eigen::VectorXd mean = colorData.colwise().mean();

    //Center the data
    Eigen::MatrixXd centeredData = colorData.rowwise() - mean.transpose();

    //Compute covariance matrix
    Eigen::MatrixXd covarianceMatrix = (centeredData.transpose() * centeredData) / (16 - 1);

    // Compute eigenvalues and eigenvectors
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(covarianceMatrix);
    Eigen::VectorXd eigenValues = eigenSolver.eigenvalues();
    Eigen::MatrixXd eigenVectors = eigenSolver.eigenvectors();

    //std::cout << "eigenValues:\n" << eigenValues << "\n";

    //std::cout << "Eigenvectors:\n" << eigenVectors << "\n";
    //std::cout << "Centered data:\n" << centeredData << "\n";

    Eigen::MatrixXd PCASpaceData = (eigenVectors.transpose() * centeredData.transpose()).transpose();

    //std::cout << "Projected data\n" << PCASpaceData << "\n";

    //the eigenvector corresponding to the highest eigenvalue is the last column of the matrix
    Eigen::MatrixXd highestIgenvector(3, 3);
    highestIgenvector.setZero();
    highestIgenvector(0, 2) = eigenVectors(0, 2);
    highestIgenvector(1, 2) = eigenVectors(1, 2);
    highestIgenvector(2, 2) = eigenVectors(2, 2);

    //std::cout << "highestIgenvector: \n" << highestIgenvector << "\n";

    Eigen::MatrixXd projectedData = (highestIgenvector * PCASpaceData.transpose()).transpose(). rowwise() + mean.transpose();

    //std::cout << "ProjectedData: \n" << projectedData << "\n";


    //get indices of the two most extreme projections 
    int minIndex;
    PCASpaceData.col(2).minCoeff(&minIndex);

    int maxIndex;
    PCASpaceData.col(2).maxCoeff(&maxIndex);

    //std::cout << "Min projection index: " << minIndex << "\n";
    //std::cout << "Max projection index: " << maxIndex << "\n";

    Eigen::Vector3d color0 = projectedData.row(minIndex);
    Eigen::Vector3d color1 = projectedData.row(maxIndex);

    //std::cout << "Min Projection: \n" << minProjection << "\n";
    //std::cout << "Max Projection: \n" << maxProjection << "\n";

    color0 = checkColorInsideBounds(color0, eigenVectors.col(2));
    color1 = checkColorInsideBounds(color1, eigenVectors.col(2));

    return { color0, color1 };
}*/


//find the two best colors for encoding, using PCA (find the largest eigenvector using the power iteration method)
Eigen::Vector3d getEigenvector_PowerIteration(const Eigen::MatrixXd& colorData, Eigen::Vector3d* meanReturn, Eigen::MatrixXd* centeredDataReturn) {

    //Compute mean of each channel
    Eigen::VectorXd mean = colorData.colwise().mean();

    //Center the data
    Eigen::MatrixXd centeredData = colorData.rowwise() - mean.transpose();

    //Compute covariance matrix
    Eigen::MatrixXd covarianceMatrix = (centeredData.transpose() * centeredData) / (16 - 1);

    if (covarianceMatrix.norm() < 1e-6) { //if covariance matrix is all zeros, the entire block is the same color

        Eigen::Vector3d zero(0.0, 0.0, 0.0);
        return zero;
    }

    Eigen::Vector3d eigenvector = Eigen::MatrixXd::Random(3, 1);
    eigenvector.normalize();

    int i = 0;

    for (i = 0; i < 100; i++) {
        Eigen::Vector3d eigenvectorNew = covarianceMatrix * eigenvector;
        eigenvectorNew.normalize();

        if (std::abs((eigenvector - eigenvectorNew).norm()) < 1e-3) {
            eigenvector = eigenvectorNew;
            break;
        }

        eigenvector = eigenvectorNew;
    }

    //std::cout << "Iteration count: " << i << "\n";

    *meanReturn = mean;
    *centeredDataReturn = centeredData;

    return eigenvector;
}


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
    double* costImprovement
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

    if (eigenvector.norm() == 0) {//entire block has the same color
        return { RGBData.row(0), RGBData.row(0) };
    }

    Eigen::MatrixXd projectionLenghts = centeredData * eigenvector / eigenvector.norm();

    auto [lowerBound, upperBound] = findSearchSpaceBounds(mean, eigenvector, 0, 255); //find boundaries for search space

    // Define parameters for simulated annealing
    double minRange = std::max(projectionLenghts.minCoeff(), lowerBound);
    double maxRange = std::min(projectionLenghts.maxCoeff(), upperBound);
    double initialTemperature = 100.0;
    double coolingRate = 0.90;
    int maxIterations = 50;
    double randomStep = 0.1;

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

        // Clamp the values to the search range
        newVal0 = std::clamp(newVal0, minRange, maxRange);
        newVal1 = std::clamp(newVal1, minRange, maxRange);

        double newCost;
        if (distanceFunction == nullptr || distanceFunctionBlock == nullptr) { //use default distance metric
            newCost = computeCumulativeDistance(projectionLenghts, val0, val1);
        }
        else {
            Eigen::Vector3d col0 = colorTransformFunction((bestVal0 * eigenvector + mean));
            Eigen::Vector3d col1 = colorTransformFunction((bestVal1 * eigenvector + mean));
            Eigen::MatrixXd closestColorData = transformToClosestColors(searchSpaceData, col0, col1, distanceFunction);

            newCost = distanceFunctionBlock(searchSpaceData, closestColorData);
        }

        double acceptanceProbability = std::exp((currentCost - newCost) / temperature);

        // Accept or reject the new solution
        if (newCost < currentCost || probDist(rng) < acceptanceProbability) {
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
        *costImprovement = initialCost - computeCumulativeDistance(projectionLenghts, bestVal0, bestVal1);
    }
    else {
        Eigen::Vector3d col0 = colorTransformFunction((bestVal0 * eigenvector + mean));
        Eigen::Vector3d col1 = colorTransformFunction((bestVal1 * eigenvector + mean));
        Eigen::MatrixXd closestColorData = transformToClosestColors(searchSpaceData, col0, col1, distanceFunction);

        *costImprovement = initialCost - distanceFunctionBlock(searchSpaceData, closestColorData);
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

int main(int argc, char* argv[]) {

    //char* inputFileArg = argv[1];
    //char* outputFile = argv[2];
    //char* colorSpaceArg = argv[3];
    //char* distanceMetricArg = argv[4];

    char* inputFileArg = "C:/faks/diplomska/image.jpeg";
    char* outputFileArg = "C:/faks/diplomska/output.dds";
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

    double costImprovementSum = 0; //testing

    for (int blockI = 0; blockI < blockCountVertical; blockI++) {

        std::cout << "block row: " << blockI << std::endl;

        for (int blockJ = 0; blockJ < blockCountHorizontal; blockJ++) {

            //pack pixels of a 4x4 block into a matrix
            Eigen::MatrixXd PCAdata(16, channels);
            for (int offsetI = 0; offsetI < 4; offsetI++) {
                for (int offsetJ = 0; offsetJ < 4; offsetJ++) {

                    int pixelIndex = (blockI * 4 + offsetI) * width + (blockJ * 4 + offsetJ);

                    if ((blockI * 4 + offsetI) >= height || (blockJ * 4 + offsetJ) >= width) {
                        PCAdata(offsetI * 4 + offsetJ, 0) = 0;
                        PCAdata(offsetI * 4 + offsetJ, 1) = 0;
                        PCAdata(offsetI * 4 + offsetJ, 2) = 0;
                    }
                    else {
                        PCAdata(offsetI * 4 + offsetJ, 0) = data[pixelIndex * channels + 0];
                        PCAdata(offsetI * 4 + offsetJ, 1) = data[pixelIndex * channels + 1];
                        PCAdata(offsetI * 4 + offsetJ, 2) = data[pixelIndex * channels + 2];
                    }
                }
            }

            double costImprovement;//testing
            
            auto [color0, color1] = findOptimalColors(PCAdata, distanceFunction, distanceFunctionBlock, colorSpace, &costImprovement);
            

            costImprovementSum += costImprovement; //testing

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
            for (int i = 0; i < 16; i++) {

                if (color0Packed == color1Packed) {
                    pixelEncodings[i] = 0x00;
                    continue;
                }

                Eigen::Vector3d pixelColor = PCAdata.row(i);

                double minDist = distanceFunction(pixelColor, color0);
                pixelEncodings[i] = 0x00;

                double dist;

                dist = distanceFunction(pixelColor, color1);

                if (dist <= minDist) {
                    minDist = dist;
                    pixelEncodings[i] = 0x01;
                }

                dist = distanceFunction(pixelColor, color2);
                if (dist <= minDist) {
                    minDist = dist;
                    pixelEncodings[i] = 0x02;
                }

                dist = distanceFunction(pixelColor, color3);
                if (dist <= minDist) {
                    minDist = dist;
                    pixelEncodings[i] = 0x03;
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

    std::cout << "costImprovement: " << costImprovementSum / (blockCountHorizontal * blockCountVertical) << std::endl;

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