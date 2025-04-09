#include <Eigen/Dense>

//double determinant2x2(double mat_00, double mat_01, double mat_10, double mat_11);
//double solveCharacteristicEquation(double a, double b, double c, double d);
//double largestEigenvalue(const Eigen::Matrix3d& A);
//Eigen::Vector3d eigenvectorForLargest(const Eigen::Matrix3d& A, double lambda_max);
//std::pair<Eigen::Vector3d, Eigen::Vector3d> getColorProjections_LargestEigenvector(const Eigen::MatrixXd& colorData);
//std::pair<Eigen::Vector3d, Eigen::Vector3d> getColorProjections_fullPCA(const Eigen::MatrixXd& colorData);

Eigen::Vector3d getEigenvector_PowerIteration(const Eigen::MatrixXd& colorData, Eigen::Vector3d* meanReturn, Eigen::MatrixXd* centeredDataReturn);