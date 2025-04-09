#include <Eigen/Dense>

//converts matrix of RGB values to HSV. HSV values are scaled to [0, 255]
Eigen::MatrixXd RGB2HSVBlock(const Eigen::MatrixXd& RGBData) {
    Eigen::MatrixXd scaledRGB = RGBData.array() / 255.0;

    Eigen::VectorXd maxVal = scaledRGB.rowwise().maxCoeff();
    Eigen::VectorXd minVal = scaledRGB.rowwise().minCoeff();
    Eigen::VectorXd delta = maxVal - minVal;

    Eigen::VectorXd H = Eigen::VectorXd::Zero(RGBData.rows());
    Eigen::VectorXd S = Eigen::VectorXd::Zero(RGBData.rows());
    Eigen::VectorXd V = maxVal;

    for (int i = 0; i < RGBData.rows(); ++i) {
        if (delta(i) > 1e-6) {
            if (scaledRGB(i, 0) == maxVal(i)) {
                H(i) = 60.0 * std::fmod((scaledRGB(i, 1) - scaledRGB(i, 2)) / delta(i), 6.0);
            }
            else if (scaledRGB(i, 1) == maxVal(i)) {
                H(i) = 60.0 * ((scaledRGB(i, 2) - scaledRGB(i, 0)) / delta(i) + 2.0);
            }
            else {
                H(i) = 60.0 * ((scaledRGB(i, 0) - scaledRGB(i, 1)) / delta(i) + 4.0);
            }

            if (H(i) < 0) {
                H(i) += 360.0;
            }
        }

        S(i) = (maxVal(i) > 0) ? (delta(i) / maxVal(i)) : 0;
    }

    Eigen::MatrixXd HSVData(RGBData.rows(), 3);
    HSVData.col(0) = H;
    HSVData.col(1) = S;
    HSVData.col(2) = V;

    return HSVData;
}

//converts RGB vector to HSV. HSV values are scaled to [0, 255]
Eigen::Vector3d RGB2HSV(const Eigen::Vector3d& RGBData) {

    Eigen::Array3d scaledRGB = RGBData.array() / 255.0;

    double maxVal = scaledRGB.maxCoeff();
    double minVal = scaledRGB.minCoeff();
    double delta = maxVal - minVal;

    double H = 0;
    if (delta > 1e-6) {
        if (maxVal == scaledRGB(0)) {
            H = 60.0 * std::fmod((scaledRGB(1) - scaledRGB(2)) / delta, 6.0);
        }
        else if (maxVal == scaledRGB(1)) {
            H = 60.0 * ((scaledRGB(2) - scaledRGB(0)) / delta + 2.0);
        }
        else {
            H = 60.0 * ((scaledRGB(0) - scaledRGB(1)) / delta + 4.0);
        }

        if (H < 0) {
            H += 360.0;
        }
    }

    //H = H * (255 / 360); //scale to [0, 255]

    double S = (maxVal > 0) ? (delta / maxVal) : 0;

    double V = maxVal;

    Eigen::Vector3d HSVData;
    HSVData << H, S, V;

    return HSVData;
}

//Converts HSV vector to RGB
Eigen::Vector3d HSV2RGB(const Eigen::Vector3d& HSVData) {

    double H = HSVData(0);
    double S = HSVData(1);
    double V = HSVData(2);

    double c = V * S;
    double x = c * (1 - std::fabs(fmod(H / 60.0, 2) - 1));
    double m = V - c;

    double R, G, B;
    if (H >= 0 && H < 60) {
        R = c; G = x; B = 0;
    }
    else if (H >= 60 && H < 120) {
        R = x; G = c; B = 0;
    }
    else if (H >= 120 && H < 180) {
        R = 0; G = c; B = x;
    }
    else if (H >= 180 && H < 240) {
        R = 0; G = x; B = c;
    }
    else if (H >= 240 && H < 300) {
        R = x; G = 0; B = c;
    }
    else {
        R = c; G = 0; B = x;
    }

    R = R + m;
    G = G + m;
    B = B + m;

    R *= 255;
    G *= 255;
    B *= 255;

    Eigen::Vector3d RGBData;
    RGBData << R, G, B;

    return RGBData;
}

// Constants for the D65 illuminant
const double REF_X = 95.047;
const double REF_Y = 100.000;
const double REF_Z = 108.883;

// Helper function to apply the gamma correction for RGB to XYZ conversion
double gammaCorrect(double value) {
    return (value > 0.04045) ? pow((value + 0.055) / 1.055, 2.4) : (value / 12.92);
}

// Helper function for the XYZ to LAB conversion
double labF(double t) {
    const double delta = 6.0 / 29.0;
    return (t > pow(delta, 3)) ? pow(t, 1.0 / 3.0) : (t / (3 * pow(delta, 2)) + 4.0 / 29.0);
}

Eigen::Vector3d RGB2CIELAB(const Eigen::Vector3d& RGBData) {

        double R = gammaCorrect(RGBData(0) / 255);
        double G = gammaCorrect(RGBData(1) / 255);
        double B = gammaCorrect(RGBData(2) / 255);

        //Convert to XYZ using the RGB to XYZ matrix (D65 illuminant)
        double x = R * 0.4124564 + G * 0.3575761 + B * 0.1804375;
        double y = R * 0.2126729 + G * 0.7151522 + B * 0.0721750;
        double z = R * 0.0193339 + G * 0.1191920 + B * 0.9503041;

        //Scale to the reference white
        x *= 100.0;
        y *= 100.0;
        z *= 100.0;

        //Normalize XYZ for D65 illuminant
        x /= REF_X;
        y /= REF_Y;
        z /= REF_Z;

        //Convert XYZ to LAB
        double fx = labF(x);
        double fy = labF(y);
        double fz = labF(z);

        double l = 116.0 * fy - 16.0;
        double a = 500.0 * (fx - fy);
        double b = 200.0 * (fy - fz);

        Eigen::Vector3d CIELABData;
        CIELABData << l, a, b;

        return CIELABData;
}

Eigen::MatrixXd RGB2CIELABBlock(const Eigen::MatrixXd& RGBData) {

    int matrixRows = RGBData.rows();
    Eigen::MatrixXd CIELABData(matrixRows, 3);

    for (int i = 0; i < matrixRows; i++) {
        double R = gammaCorrect(RGBData(i, 0) / 255);
        double G = gammaCorrect(RGBData(i, 1) / 255);
        double B = gammaCorrect(RGBData(i, 2) / 255);

        //Convert to XYZ using the RGB to XYZ matrix (D65 illuminant)
        double x = R * 0.4124564 + G * 0.3575761 + B * 0.1804375;
        double y = R * 0.2126729 + G * 0.7151522 + B * 0.0721750;
        double z = R * 0.0193339 + G * 0.1191920 + B * 0.9503041;

        //Scale to the reference white
        x *= 100.0;
        y *= 100.0;
        z *= 100.0;

        //Normalize XYZ for D65 illuminant
        x /= REF_X;
        y /= REF_Y;
        z /= REF_Z;

        //Convert XYZ to LAB
        double fx = labF(x);
        double fy = labF(y);
        double fz = labF(z);

        double l = 116.0 * fy - 16.0;
        double a = 500.0 * (fx - fy);
        double b = 200.0 * (fy - fz);

        CIELABData.row(i) << l, a, b;
    }

    return CIELABData;
}

// Helper function for the LAB to XYZ conversion
double labFInverse(double t) {
    const double delta = 6.0 / 29.0;
    return (t > delta) ? (t * t * t) : (3 * pow(delta, 2) * (t - 4.0 / 29.0));
}

// Helper function to apply gamma correction for XYZ to RGB conversion
double gammaCorrectInverse(double value) {
    return (value > 0.0031308) ? (1.055 * pow(value, 1.0 / 2.4) - 0.055) : (12.92 * value);
}

Eigen::Vector3d CIELAB2RGB(const Eigen::Vector3d& CIELABData) {

    double scaledL = CIELABData(0);
    double scaledA = CIELABData(1);
    double scaledB = CIELABData(2);

    double fy = (scaledL + 16.0) / 116.0;
    double fx = fy + scaledA / 500.0;
    double fz = fy - scaledB / 200.0;

    double x = REF_X * labFInverse(fx);
    double y = REF_Y * labFInverse(fy);
    double z = REF_Z * labFInverse(fz);

    //Normalize XYZ to [0, 1]
    x /= 100.0;
    y /= 100.0;
    z /= 100.0;

    //Convert XYZ to linear RGB
    double R = x * 3.2406 + y * -1.5372 + z * -0.4986;
    double G = x * -0.9689 + y * 1.8758 + z * 0.0415;
    double B = x * 0.0557 + y * -0.2040 + z * 1.0570;

    //Apply gamma correction and scale to [0, 1]
    R = std::clamp(gammaCorrectInverse(R), 0.0, 1.0);
    G = std::clamp(gammaCorrectInverse(G), 0.0, 1.0);
    B = std::clamp(gammaCorrectInverse(B), 0.0, 1.0);

    Eigen::Vector3d RGBData;
    RGBData << R * 255, G * 255, B * 255;

    return RGBData;
}

Eigen::Vector3d RGB2RGB(const Eigen::Vector3d& RGBData) {
    return RGBData;
}