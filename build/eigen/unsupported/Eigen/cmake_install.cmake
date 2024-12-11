# Install script for directory: C:/faks/diplomska/kompresijaCPU/eigen/unsupported/Eigen

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "C:/Program Files (x86)/LearnWebGPU")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Devel" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/eigen3/unsupported/Eigen" TYPE FILE FILES
    "C:/faks/diplomska/kompresijaCPU/eigen/unsupported/Eigen/AdolcForward"
    "C:/faks/diplomska/kompresijaCPU/eigen/unsupported/Eigen/AlignedVector3"
    "C:/faks/diplomska/kompresijaCPU/eigen/unsupported/Eigen/ArpackSupport"
    "C:/faks/diplomska/kompresijaCPU/eigen/unsupported/Eigen/AutoDiff"
    "C:/faks/diplomska/kompresijaCPU/eigen/unsupported/Eigen/BVH"
    "C:/faks/diplomska/kompresijaCPU/eigen/unsupported/Eigen/EulerAngles"
    "C:/faks/diplomska/kompresijaCPU/eigen/unsupported/Eigen/FFT"
    "C:/faks/diplomska/kompresijaCPU/eigen/unsupported/Eigen/IterativeSolvers"
    "C:/faks/diplomska/kompresijaCPU/eigen/unsupported/Eigen/KroneckerProduct"
    "C:/faks/diplomska/kompresijaCPU/eigen/unsupported/Eigen/LevenbergMarquardt"
    "C:/faks/diplomska/kompresijaCPU/eigen/unsupported/Eigen/MatrixFunctions"
    "C:/faks/diplomska/kompresijaCPU/eigen/unsupported/Eigen/MoreVectorization"
    "C:/faks/diplomska/kompresijaCPU/eigen/unsupported/Eigen/MPRealSupport"
    "C:/faks/diplomska/kompresijaCPU/eigen/unsupported/Eigen/NonLinearOptimization"
    "C:/faks/diplomska/kompresijaCPU/eigen/unsupported/Eigen/NumericalDiff"
    "C:/faks/diplomska/kompresijaCPU/eigen/unsupported/Eigen/OpenGLSupport"
    "C:/faks/diplomska/kompresijaCPU/eigen/unsupported/Eigen/Polynomials"
    "C:/faks/diplomska/kompresijaCPU/eigen/unsupported/Eigen/Skyline"
    "C:/faks/diplomska/kompresijaCPU/eigen/unsupported/Eigen/SparseExtra"
    "C:/faks/diplomska/kompresijaCPU/eigen/unsupported/Eigen/SpecialFunctions"
    "C:/faks/diplomska/kompresijaCPU/eigen/unsupported/Eigen/Splines"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Devel" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/eigen3/unsupported/Eigen" TYPE DIRECTORY FILES "C:/faks/diplomska/kompresijaCPU/eigen/unsupported/Eigen/src" FILES_MATCHING REGEX "/[^/]*\\.h$")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("C:/faks/diplomska/kompresijaCPU/build/eigen/unsupported/Eigen/CXX11/cmake_install.cmake")

endif()

