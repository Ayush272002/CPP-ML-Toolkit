#include "KNN.hpp"
#include <map>
#include <spdlog/spdlog.h>

// ----------------------------
// Constructors & Destructor
// ----------------------------
KNN::KNN()
    : k(3), knn_type(KNNType::Classification), metric(DistanceMetric::EUCLIDEAN), minkowski_p(2.0) {}

KNN::KNN(int k, KNNType knn_type, DistanceMetric metric, double p)
    : k(k), knn_type(knn_type), metric(metric), minkowski_p(p) {}

KNN::~KNN() {}

// ----------------------------
// Getters & Setters
// ----------------------------
int KNN::get_k() const { return k; }
void KNN::set_k(int k_val) { k = k_val; }

KNNType KNN::get_knn_type() const { return knn_type; }
void KNN::set_knn_type(KNNType type_val) { knn_type = type_val; }

DistanceMetric KNN::get_distance_metric() const { return metric; }
void KNN::set_distance_metric(DistanceMetric metric_val) { metric = metric_val; }

double KNN::get_minkowski_p() const { return minkowski_p; }
void KNN::set_minkowski_p(double p_val) { minkowski_p = p_val; }

// ----------------------------
// Fit
// ----------------------------
void KNN::fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y)
{
    spdlog::info("fit called: X rows={}, cols={}, y size={}", X.rows(), X.cols(), y.size());
    if (X.rows() != y.size())
    {
        spdlog::error("fit error: X rows {} != y size {}", X.rows(), y.size());
        throw std::invalid_argument("X rows must match y size");
    }
    X_train = X;
    y_train = y;
    spdlog::info("fit finished");
}

// ----------------------------
// Predict Multiple Samples
// ----------------------------
Eigen::VectorXd KNN::predict(const Eigen::MatrixXd &X) const
{
    spdlog::info("predict (batch) called: X rows={}, cols={}", X.rows(), X.cols());
    Eigen::VectorXd predictions(X.rows());
    for (int i = 0; i < X.rows(); ++i)
    {
        predictions(i) = predict(Eigen::VectorXd(X.row(i).transpose()));
        if (i < 5)
            spdlog::debug("predict (batch) sample {} done", i); // log first few
    }
    spdlog::info("predict (batch) finished");
    return predictions;
}

// Predict Single Sample
double KNN::predict(const Eigen::VectorXd &x) const
{
    spdlog::debug("predict (single) called");
    if (knn_type == KNNType::Classification)
        return predict_classification(x);
    else
        return predict_regression(x);
}

// ----------------------------
// Internal: Find Nearest Neighbors
// ----------------------------
std::vector<int> KNN::find_nearest_neighbors(const Eigen::VectorXd &x) const
{
    spdlog::debug("find_nearest_neighbors called");
    std::vector<std::pair<int, double>> distances;
    distances.reserve(X_train.rows());

    for (int i = 0; i < X_train.rows(); i++)
    {
        double dist = compute_distance(x, X_train.row(i).transpose());
        distances.emplace_back(i, dist);
    }

    std::nth_element(distances.begin(), distances.begin() + k, distances.end(),
                     [](const std::pair<int, double> &a, const std::pair<int, double> &b)
                     {
                         return a.second < b.second;
                     });

    std::vector<int> neighbors(k);
    for (int i = 0; i < k; i++)
        neighbors[i] = distances[i].first;

    return neighbors;
}

// ----------------------------
// Internal: Distance Computation
// ----------------------------
double KNN::compute_distance(const Eigen::VectorXd &a, const Eigen::VectorXd &b) const
{
    spdlog::debug("compute_distance called, metric={} ", static_cast<int>(metric));
    switch (metric)
    {
    case DistanceMetric::EUCLIDEAN:
        return (a - b).norm();
    case DistanceMetric::MANHATTAN:
        return (a - b).cwiseAbs().sum();
    case DistanceMetric::MINKOWSKI:
        return std::pow((a - b).cwiseAbs().array().pow(minkowski_p).sum(), 1.0 / minkowski_p);
    default:
        throw std::invalid_argument("Unknown distance metric");
    }
}

// ----------------------------
// Internal: Classification Prediction
// ----------------------------
double KNN::predict_classification(const Eigen::VectorXd &x) const
{
    spdlog::debug("predict_classification called");
    std::vector<int> nbrs = find_nearest_neighbors(x);
    std::map<double, int> vote_count;

    for (int idx : nbrs)
        vote_count[y_train(idx)]++;

    return std::max_element(vote_count.begin(), vote_count.end(),
                            [](const auto &a, const auto &b)
                            { return a.second < b.second; })
        ->first;
}

// ----------------------------
// Internal: Regression Prediction
// ----------------------------
double KNN::predict_regression(const Eigen::VectorXd &x) const
{
    spdlog::debug("predict_regression called");
    std::vector<int> neighbors = find_nearest_neighbors(x);
    double sum = 0.0;
    for (int idx : neighbors)
        sum += y_train(idx);

    return sum / k;
}
