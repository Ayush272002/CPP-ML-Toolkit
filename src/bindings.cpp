#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <spdlog/spdlog.h>
#include "LinearRegression.hpp"
#include "LogisticRegression.hpp"
#include "KNN.hpp"

namespace py = pybind11;

#ifdef BUILD_LINEAR_REGRESSION
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
#endif

#ifdef BUILD_LOGISTIC_REGRESSION
PYBIND11_MODULE(logistic_regression_cpp, m)
{
  spdlog::info("PYBIND11_MODULE logistic_regression_cpp entered");

  py::enum_<RegularizationType>(m, "RegularizationType")
      .value("None", RegularizationType::None)
      .value("L1", RegularizationType::L1)
      .value("L2", RegularizationType::L2)
      .export_values();

  py::class_<LogisticRegression>(m, "LogisticRegression")
      .def(py::init<bool, double, int, double, RegularizationType>(),
           py::arg("fit_intercept") = true,
           py::arg("lr") = 0.01,
           py::arg("epochs") = 1000,
           py::arg("reg_lambda") = 0.0,
           py::arg("reg_type") = RegularizationType::None)
      .def("fit", [](LogisticRegression &self, const Eigen::MatrixXd &X, const Eigen::VectorXd &y, bool verbose)
           {
         spdlog::info("pybind11: fit called from Python");
         spdlog::info("pybind11: X rows: {}, cols: {}", X.rows(), X.cols());
         spdlog::info("pybind11: y size: {}", y.size());
         self.fit(X, y, verbose); }, py::arg("X"), py::arg("y"), py::arg("verbose") = false)
      .def("predict", [](const LogisticRegression &self, const Eigen::MatrixXd &X, double threshold)
           {
         spdlog::info("pybind11: predict called from Python");
         spdlog::info("pybind11: X rows: {}, cols: {}", X.rows(), X.cols());
         return self.predict(X, threshold); }, py::arg("X"), py::arg("threshold") = 0.5)
      .def("predict_prob", &LogisticRegression::predict_prob)
      .def("get_weights", &LogisticRegression::get_weights)
      .def("get_bias", &LogisticRegression::get_bias)
      .def("get_loss_history", &LogisticRegression::get_loss_history)
      .def("set_learning_rate", &LogisticRegression::set_learning_rate)
      .def("set_epochs", &LogisticRegression::set_epochs)
      .def("set_fit_intercept", &LogisticRegression::set_fit_intercept)
      .def("set_regularization", &LogisticRegression::set_regularization)
      .def("compute_loss", &LogisticRegression::compute_loss);
}
#endif

#ifdef BUILD_KNN
PYBIND11_MODULE(knn_cpp, m)
{
  spdlog::info("PYBIND11_MODULE knn_cpp entered");

  py::enum_<DistanceMetric>(m, "DistanceMetric")
    .value("EUCLIDEAN", DistanceMetric::EUCLIDEAN)
    .value("MANHATTAN", DistanceMetric::MANHATTAN)
    .value("MINKOWSKI", DistanceMetric::MINKOWSKI)
    .export_values();

  py::enum_<KNNType>(m, "KNNType")
    .value("Classification", KNNType::Classification)
    .value("Regression", KNNType::Regression)
    .export_values();

  py::class_<KNN>(m, "KNN")
    .def(py::init<>())
    .def(py::init<int, KNNType, DistanceMetric, double>(),
       py::arg("k") = 3,
       py::arg("knn_type") = KNNType::Classification,
       py::arg("metric") = DistanceMetric::EUCLIDEAN,
       py::arg("p") = 2.0)
    .def("fit", &KNN::fit, py::arg("X"), py::arg("y"))
    .def("predict", py::overload_cast<const Eigen::VectorXd &>(&KNN::predict, py::const_), py::arg("x"))
    .def("predict", py::overload_cast<const Eigen::MatrixXd &>(&KNN::predict, py::const_), py::arg("X"))
    .def("get_k", &KNN::get_k)
    .def("set_k", &KNN::set_k)
    .def("get_knn_type", &KNN::get_knn_type)
    .def("set_knn_type", &KNN::set_knn_type)
    .def("get_distance_metric", &KNN::get_distance_metric)
    .def("set_distance_metric", &KNN::set_distance_metric)
    .def("get_minkowski_p", &KNN::get_minkowski_p)
    .def("set_minkowski_p", &KNN::set_minkowski_p);
}
#endif
