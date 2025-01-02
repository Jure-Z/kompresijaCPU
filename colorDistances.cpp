#include <Eigen/Dense>
#include <cmath>

Eigen::Vector3d weightsRGB(0.3, 0.59, 0.11);
Eigen::Vector3d weightsHSV(1.0, 0.5, 0.5);


// Function to compute the L1 (Manhattan) distance
double distanceL1(const Eigen::Vector3d& color1, const Eigen::Vector3d& color2) {
    return (color1 - color2).cwiseAbs().sum();
}

// Function to compute the L2 (Euclidean) distance
double distanceL2(const Eigen::Vector3d& color1, const Eigen::Vector3d& color2) {
    return (color1 - color2).norm();
}

// Weighted L1 distance for RGB
double weightedDistanceL1RGB(const Eigen::Vector3d& color1, const Eigen::Vector3d& color2) {
    return ((color1 - color2).cwiseAbs().cwiseProduct(weightsRGB)).sum();
}

// Weighted L2 distance for RGB
double weightedDistanceL2RGB(const Eigen::Vector3d& color1, const Eigen::Vector3d& color2) {
    return std::sqrt(((color1 - color2).cwiseProduct(weightsRGB).array().square()).sum());
}

// Helper function to handle circular hue in HSV space
double computeHueDistance(double h1, double h2) {
    double delta = std::abs(h1 - h2);
    return std::min(delta, 360.0 - delta); // Shortest angular distance
}

// L1 distance for HSV with hue handling
double distanceL1HSV(const Eigen::Vector3d& color1, const Eigen::Vector3d& color2) {
    double hueDistance = computeHueDistance(color1[0], color2[0]) / 360.0; // Normalize hue difference to [0, 1]
    Eigen::Vector3d distances(hueDistance, std::abs(color1[1] - color2[1]), std::abs(color1[2] - color2[2]));
    return distances.sum();
}

// L2 distance for HSV with hue handling
double distanceL2HSV(const Eigen::Vector3d& color1, const Eigen::Vector3d& color2) {
    double hueDistance = computeHueDistance(color1[0], color2[0]) / 360.0; // Normalize hue difference to [0, 1]
    Eigen::Vector3d distances(hueDistance, std::abs(color1[1] - color2[1]), std::abs(color1[2] - color2[2]));
    return std::sqrt((distances.array().square()).sum());
}

// Weighted L1 distance for HSV with hue handling
double weightedDistanceL1HSV(const Eigen::Vector3d& color1, const Eigen::Vector3d& color2) {
    double hueDistance = computeHueDistance(color1[0], color2[0]) / 360.0; // Normalize hue difference to [0, 1]
    Eigen::Vector3d distances(hueDistance, std::abs(color1[1] - color2[1]), std::abs(color1[2] - color2[2]));
    return (distances.cwiseProduct(weightsHSV)).sum();
}

// Weighted L2 distance for HSV with hue handling
double weightedDistanceL2HSV(const Eigen::Vector3d& color1, const Eigen::Vector3d& color2) {
    double hueDistance = computeHueDistance(color1[0], color2[0]) / 360.0; // Normalize hue difference to [0, 1]
    Eigen::Vector3d distances(hueDistance, std::abs(color1[1] - color2[1]), std::abs(color1[2] - color2[2]));
    return std::sqrt((distances.cwiseProduct(weightsHSV).array().square()).sum());
}



// Function to compute the L1 distance for an entire block
double distanceL1Block(const Eigen::MatrixXd& block1, const Eigen::MatrixXd& block2) {
    if (block1.rows() != block2.rows() || block1.cols() != 3 || block2.cols() != 3) {
        throw std::invalid_argument("Blocks must be Nx3 matrices with the same number of rows.");
    }
    return (block1 - block2).rowwise().sum().sum(); // Sum over all rows and columns
}

// Function to compute the L2 distance for an entire block
double distanceL2Block(const Eigen::MatrixXd& block1, const Eigen::MatrixXd& block2) {
    if (block1.rows() != block2.rows() || block1.cols() != 3 || block2.cols() != 3) {
        throw std::invalid_argument("Blocks must be Nx3 matrices with the same number of rows.");
    }
    return std::sqrt((block1 - block2).array().square().rowwise().sum().sum()); // Sum of squared differences
}

// Function to compute the weighted L1 distance for an entire RGB block
double weightedDistanceL1RGBBlock(const Eigen::MatrixXd& block1, const Eigen::MatrixXd& block2) {
    if (block1.rows() != block2.rows() || block1.cols() != 3 || block2.cols() != 3) {
        throw std::invalid_argument("Blocks must be Nx3 matrices with the same number of rows.");
    }
    Eigen::MatrixXd weightedDiff = (block1 - block2).cwiseAbs().array().rowwise() * weightsRGB.transpose().array();
    return weightedDiff.rowwise().sum().sum(); // Sum over all rows and columns
}

// Function to compute the weighted L2 distance for an entire RGB block
double weightedDistanceL2RGBBlock(const Eigen::MatrixXd& block1, const Eigen::MatrixXd& block2) {
    if (block1.rows() != block2.rows() || block1.cols() != 3 || block2.cols() != 3) {
        throw std::invalid_argument("Blocks must be Nx3 matrices with the same number of rows.");
    }
    Eigen::MatrixXd weightedDiff = (block1 - block2).array().rowwise() * weightsRGB.transpose().array();
    return std::sqrt(weightedDiff.array().square().rowwise().sum().sum()); // Sum of squared differences
}



// Helper function to compute the shortest angular distance for hue
Eigen::VectorXd computeHueDistances(const Eigen::VectorXd& h1, const Eigen::VectorXd& h2) {
    Eigen::VectorXd delta = (h1 - h2).cwiseAbs();
    return delta.array().min((360.0 - delta.array()).abs()); // Shortest angular distance
}

// Function to compute the weighted L1 distance for an HSV block
double distanceL1HSVBlock(const Eigen::MatrixXd& block1, const Eigen::MatrixXd& block2) {
    if (block1.rows() != block2.rows() || block1.cols() != 3 || block2.cols() != 3) {
        throw std::invalid_argument("Blocks must be Nx3 matrices with the same number of rows.");
    }

    Eigen::VectorXd hueDistances = computeHueDistances(block1.col(0), block2.col(0)) / 360.0; // Normalize hue differences to [0, 1]
    Eigen::VectorXd satDistances = (block1.col(1) - block2.col(1)).cwiseAbs();
    Eigen::VectorXd valDistances = (block1.col(2) - block2.col(2)).cwiseAbs();

    // Combine distances with weights
    Eigen::MatrixXd weightedDiff(block1.rows(), 3);
    weightedDiff.col(0) = hueDistances;
    weightedDiff.col(1) = satDistances;
    weightedDiff.col(2) = valDistances;

    return weightedDiff.rowwise().sum().sum(); // Sum over all rows and columns
}

// Function to compute the weighted L2 distance for an HSV block
double distanceL2HSVBlock(const Eigen::MatrixXd& block1, const Eigen::MatrixXd& block2) {
    if (block1.rows() != block2.rows() || block1.cols() != 3 || block2.cols() != 3) {
        throw std::invalid_argument("Blocks must be Nx3 matrices with the same number of rows.");
    }

    Eigen::VectorXd hueDistances = computeHueDistances(block1.col(0), block2.col(0)) / 360.0; // Normalize hue differences to [0, 1]
    Eigen::VectorXd satDistances = block1.col(1) - block2.col(1);
    Eigen::VectorXd valDistances = block1.col(2) - block2.col(2);

    // Combine distances with weights
    Eigen::MatrixXd weightedDiff(block1.rows(), 3);
    weightedDiff.col(0) = hueDistances;
    weightedDiff.col(1) = satDistances;
    weightedDiff.col(2) = valDistances;

    return std::sqrt((weightedDiff.array().square()).rowwise().sum().sum()); // Sum of squared differences
}

// Function to compute the weighted L1 distance for an HSV block
double weightedDistanceL1HSVBlock(const Eigen::MatrixXd& block1, const Eigen::MatrixXd& block2) {
    if (block1.rows() != block2.rows() || block1.cols() != 3 || block2.cols() != 3) {
        throw std::invalid_argument("Blocks must be Nx3 matrices with the same number of rows.");
    }

    Eigen::VectorXd hueDistances = computeHueDistances(block1.col(0), block2.col(0)) / 360.0; // Normalize hue differences to [0, 1]
    Eigen::VectorXd satDistances = (block1.col(1) - block2.col(1)).cwiseAbs();
    Eigen::VectorXd valDistances = (block1.col(2) - block2.col(2)).cwiseAbs();

    // Combine distances with weights
    Eigen::MatrixXd weightedDiff(block1.rows(), 3);
    weightedDiff.col(0) = hueDistances * weightsHSV[0];
    weightedDiff.col(1) = satDistances * weightsHSV[1];
    weightedDiff.col(2) = valDistances * weightsHSV[2];

    return weightedDiff.rowwise().sum().sum(); // Sum over all rows and columns
}

// Function to compute the weighted L2 distance for an HSV block
double weightedDistanceL2HSVBlock(const Eigen::MatrixXd& block1, const Eigen::MatrixXd& block2) {
    if (block1.rows() != block2.rows() || block1.cols() != 3 || block2.cols() != 3) {
        throw std::invalid_argument("Blocks must be Nx3 matrices with the same number of rows.");
    }

    Eigen::VectorXd hueDistances = computeHueDistances(block1.col(0), block2.col(0)) / 360.0; // Normalize hue differences to [0, 1]
    Eigen::VectorXd satDistances = block1.col(1) - block2.col(1);
    Eigen::VectorXd valDistances = block1.col(2) - block2.col(2);

    // Combine distances with weights
    Eigen::MatrixXd weightedDiff(block1.rows(), 3);
    weightedDiff.col(0) = hueDistances * weightsHSV[0];
    weightedDiff.col(1) = satDistances * weightsHSV[1];
    weightedDiff.col(2) = valDistances * weightsHSV[2];

    return std::sqrt((weightedDiff.array().square()).rowwise().sum().sum()); // Sum of squared differences
}
