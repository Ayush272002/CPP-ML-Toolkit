#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "LinearRegression.hpp"
#include <spdlog/spdlog.h>

namespace py = pybind11;

PYBIND11_MODULE(linear_regression_cpp, m)
{
    spdlog::info("PYBIND11_MODULE linear_regression_cpp entered");
    py::class_<LinearRegression>(m, "LinearRegression")
        .def(py::init<bool, double, int>(), py::arg("fit_intercept") = true, py::arg("lr") = 0.01, py::arg("epochs") = 1000)
        .def("fit", [](LinearRegression &self, const Eigen::MatrixXd &X, const Eigen::VectorXd &y, bool verbose)
             {
          spdlog::info("pybind11: fit called from Python");
          spdlog::info("pybind11: X rows: {}, cols: {}", X.rows(), X.cols());
          spdlog::info("pybind11: y size: {}", y.size());
          self.fit(X, y, verbose); }, py::arg("X"), py::arg("y"), py::arg("verbose") = false)
        .def("predict", [](const LinearRegression &self, const Eigen::MatrixXd &X)
             {
          spdlog::info("pybind11: predict called from Python");
          spdlog::info("pybind11: X rows: {}, cols: {}", X.rows(), X.cols());
          return self.predict(X); })
        .def("get_weights", &LinearRegression::get_weights)
        .def("get_bias", &LinearRegression::get_bias)
        .def("get_loss_history", &LinearRegression::get_loss_history);
}
