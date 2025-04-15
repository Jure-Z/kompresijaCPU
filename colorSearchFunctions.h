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

//double computeCumulativeDistance(Eigen::VectorXd data, double val0, double val1);

Eigen::MatrixXd transformToClosestColors(const Eigen::MatrixXd& colorData, Eigen::Vector3d color0, Eigen::Vector3d color1, double (*distanceFunction)(const Eigen::Vector3d&, const Eigen::Vector3d&), int blockType);

std::pair<double, double> findSearchSpaceBounds(const Eigen::Vector3d& mean, const Eigen::Vector3d& eigenvector, double lowerBoundary, double upperBoundary);

std::pair<Eigen::Vector3d, Eigen::Vector3d> findOptimalColorsCooling(
    const Eigen::MatrixXd& RGBData,
    double (*distanceFunction)(const Eigen::Vector3d&, const Eigen::Vector3d&),
    double (*distanceFunctionBlock)(const Eigen::MatrixXd&, const Eigen::MatrixXd&),
    ColorSpace colorSpace,
    int blockType,
    Eigen::Vector3d* meanReturn,
    double* cost, //distance between uncompressed and compressed block
    double* costImprovement //difference between initial and optimised color solutions
);

std::pair<Eigen::Vector3d, Eigen::Vector3d> findOptimalColorsLinear(
    const Eigen::MatrixXd& RGBData,
    double (*distanceFunction)(const Eigen::Vector3d&, const Eigen::Vector3d&),
    double (*distanceFunctionBlock)(const Eigen::MatrixXd&, const Eigen::MatrixXd&),
    ColorSpace colorSpace,
    int blockType,
    Eigen::Vector3d* meanReturn,
    double* cost, //distance between uncompressed and compressed block
    double* costImprovement //difference between initial and optimised color solutions
);

std::pair<Eigen::Vector3d, Eigen::Vector3d> findOptimalColorsHybrid(
    const Eigen::MatrixXd& RGBData,
    double (*distanceFunction)(const Eigen::Vector3d&, const Eigen::Vector3d&),
    double (*distanceFunctionBlock)(const Eigen::MatrixXd&, const Eigen::MatrixXd&),
    ColorSpace colorSpace,
    int blockType,
    Eigen::Vector3d* meanReturn,
    double* cost, //distance between uncompressed and compressed block
    double* costImprovement //difference between initial and optimised color solutions
);