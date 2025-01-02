#include <Eigen/Dense>

//converts matrix of RGB values to HSV. HSV values are scaled to [0, 1]
Eigen::Vector3d RGB2HSV(const Eigen::Vector3d& RGBData);

Eigen::MatrixXd RGB2HSVBlock(const Eigen::MatrixXd& RGBData);

//Converts HSV vector to RGB
Eigen::Vector3d HSV2RGB(const Eigen::Vector3d& HSVData);

Eigen::Vector3d RGB2CIELAB(const Eigen::Vector3d& RGBData);

Eigen::MatrixXd RGB2CIELABBlock(const Eigen::MatrixXd& RGBData);

Eigen::Vector3d CIELAB2RGB(const Eigen::Vector3d& CIELABData);

Eigen::Vector3d RGB2RGB(const Eigen::Vector3d& RGBData);