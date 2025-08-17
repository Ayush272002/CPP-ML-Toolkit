set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_TEST_NOQT ON CACHE BOOL "" FORCE)
set(EIGEN_LEAVE_TEST_IN_ALL_TARGET OFF CACHE BOOL "" FORCE)

include(FetchContent)

# Eigen
FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
)
FetchContent_MakeAvailable(eigen)

# spdlog
FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.13.0
)
FetchContent_MakeAvailable(spdlog)
cmake_minimum_required(VERSION 3.10)
project(LinearRegression LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include ${eigen_SOURCE_DIR} ${spdlog_SOURCE_DIR}/include)

# pybind11 from the venv
set(pybind11_DIR "${CMAKE_SOURCE_DIR}/venv/lib/python3.10/site-packages/pybind11/share/cmake/pybind11")
find_package(pybind11 REQUIRED)

add_executable(linear_regression_test
    src/main.cpp
    src/LinearRegression.cpp
)

pybind11_add_module(linear_regression_cpp
    src/bindings.cpp
    src/LinearRegression.cpp
)
