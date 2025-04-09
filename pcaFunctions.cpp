#include <Eigen/Dense>

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