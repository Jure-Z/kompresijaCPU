#include <Eigen/Dense>

// Function to compute the L1 (Manhattan) distance
double distanceL1(const Eigen::Vector3d& color1, const Eigen::Vector3d& color2);

// Function to compute the L2 (Euclidean) distance
double distanceL2(const Eigen::Vector3d& color1, const Eigen::Vector3d& color2);

// Weighted L1 distance for RGB
double weightedDistanceL1RGB(const Eigen::Vector3d& color1, const Eigen::Vector3d& color2);

// Weighted L2 distance for RGB
double weightedDistanceL2RGB(const Eigen::Vector3d& color1, const Eigen::Vector3d& color2);

// L1 distance for HSV with hue handling
double distanceL1HSV(const Eigen::Vector3d& color1, const Eigen::Vector3d& color2);

// L2 distance for HSV with hue handling
double distanceL2HSV(const Eigen::Vector3d& color1, const Eigen::Vector3d& color2);

// Weighted L1 distance for HSV with hue handling
double weightedDistanceL1HSV(const Eigen::Vector3d& color1, const Eigen::Vector3d& color2);

// Weighted L2 distance for HSV with hue handling
double weightedDistanceL2HSV(const Eigen::Vector3d& color1, const Eigen::Vector3d& color2);

// Function to compute the L1 distance for an entire block
double distanceL1Block(const Eigen::MatrixXd& block1, const Eigen::MatrixXd& block2);

// Function to compute the L2 distance for an entire block
double distanceL2Block(const Eigen::MatrixXd& block1, const Eigen::MatrixXd& block2);

// Function to compute the weighted L1 distance for an entire RGB block
double weightedDistanceL1RGBBlock(const Eigen::MatrixXd& block1, const Eigen::MatrixXd& block2);

// Function to compute the weighted L2 distance for an entire RGB block
double weightedDistanceL2RGBBlock(const Eigen::MatrixXd& block1, const Eigen::MatrixXd& block2);

// Function to compute the weighted L1 distance for an HSV block
double distanceL1HSVBlock(const Eigen::MatrixXd& block1, const Eigen::MatrixXd& block2);

// Function to compute the weighted L2 distance for an HSV block
double distanceL2HSVBlock(const Eigen::MatrixXd& block1, const Eigen::MatrixXd& block2);

// Function to compute the weighted L1 distance for an HSV block
double weightedDistanceL1HSVBlock(const Eigen::MatrixXd& block1, const Eigen::MatrixXd& block2);

// Function to compute the weighted L2 distance for an HSV block
double weightedDistanceL2HSVBlock(const Eigen::MatrixXd& block1, const Eigen::MatrixXd& block2);