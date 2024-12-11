# Install script for directory: C:/faks/diplomska/kompresijaCPU/eigen/Eigen

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
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/eigen3/Eigen" TYPE FILE FILES
    "C:/faks/diplomska/kompresijaCPU/eigen/Eigen/Cholesky"
    "C:/faks/diplomska/kompresijaCPU/eigen/Eigen/CholmodSupport"
    "C:/faks/diplomska/kompresijaCPU/eigen/Eigen/Core"
    "C:/faks/diplomska/kompresijaCPU/eigen/Eigen/Dense"
    "C:/faks/diplomska/kompresijaCPU/eigen/Eigen/Eigen"
    "C:/faks/diplomska/kompresijaCPU/eigen/Eigen/Eigenvalues"
    "C:/faks/diplomska/kompresijaCPU/eigen/Eigen/Geometry"
    "C:/faks/diplomska/kompresijaCPU/eigen/Eigen/Householder"
    "C:/faks/diplomska/kompresijaCPU/eigen/Eigen/IterativeLinearSolvers"
    "C:/faks/diplomska/kompresijaCPU/eigen/Eigen/Jacobi"
    "C:/faks/diplomska/kompresijaCPU/eigen/Eigen/KLUSupport"
    "C:/faks/diplomska/kompresijaCPU/eigen/Eigen/LU"
    "C:/faks/diplomska/kompresijaCPU/eigen/Eigen/MetisSupport"
    "C:/faks/diplomska/kompresijaCPU/eigen/Eigen/OrderingMethods"
    "C:/faks/diplomska/kompresijaCPU/eigen/Eigen/PaStiXSupport"
    "C:/faks/diplomska/kompresijaCPU/eigen/Eigen/PardisoSupport"
    "C:/faks/diplomska/kompresijaCPU/eigen/Eigen/QR"
    "C:/faks/diplomska/kompresijaCPU/eigen/Eigen/QtAlignedMalloc"
    "C:/faks/diplomska/kompresijaCPU/eigen/Eigen/SPQRSupport"
    "C:/faks/diplomska/kompresijaCPU/eigen/Eigen/SVD"
    "C:/faks/diplomska/kompresijaCPU/eigen/Eigen/Sparse"
    "C:/faks/diplomska/kompresijaCPU/eigen/Eigen/SparseCholesky"
    "C:/faks/diplomska/kompresijaCPU/eigen/Eigen/SparseCore"
    "C:/faks/diplomska/kompresijaCPU/eigen/Eigen/SparseLU"
    "C:/faks/diplomska/kompresijaCPU/eigen/Eigen/SparseQR"
    "C:/faks/diplomska/kompresijaCPU/eigen/Eigen/StdDeque"
    "C:/faks/diplomska/kompresijaCPU/eigen/Eigen/StdList"
    "C:/faks/diplomska/kompresijaCPU/eigen/Eigen/StdVector"
    "C:/faks/diplomska/kompresijaCPU/eigen/Eigen/SuperLUSupport"
    "C:/faks/diplomska/kompresijaCPU/eigen/Eigen/UmfPackSupport"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Devel" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/eigen3/Eigen" TYPE DIRECTORY FILES "C:/faks/diplomska/kompresijaCPU/eigen/Eigen/src" FILES_MATCHING REGEX "/[^/]*\\.h$")
endif()

