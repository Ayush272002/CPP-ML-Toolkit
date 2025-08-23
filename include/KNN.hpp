#pragma once

#include <Eigen/Dense>
#include <vector>

enum class DistanceMetric
{
    EUCLIDEAN,
    MANHATTAN,
    MINKOWSKI
};

enum class KNNType
{
    Classification,
    Regression
};

class KNN
{
private:
    int k;
    KNNType knn_type;
    DistanceMetric metric;
    double minkowski_p;

    Eigen::MatrixXd X_train; // Stored training features
    Eigen::VectorXd y_train; // Stored training labels/targets

    // ----------------------------
    // Internal helpers
    // ----------------------------
    double compute_distance(const Eigen::VectorXd &a,
                            const Eigen::VectorXd &b) const;

    double predict_classification(const Eigen::VectorXd &x) const;
    double predict_regression(const Eigen::VectorXd &x) const;

    std::vector<int> find_nearest_neighbors(const Eigen::VectorXd &x) const;

public:
    KNN();
    KNN(int k, KNNType knn_type = KNNType::Classification, DistanceMetric metric = DistanceMetric::EUCLIDEAN, double p = 2.0);
    ~KNN();

    // ----------------------------
    // Getters & Setters
    // ----------------------------
    int get_k() const;
    void set_k(int k);

    KNNType get_knn_type() const;
    void set_knn_type(KNNType type);

    DistanceMetric get_distance_metric() const;
    void set_distance_metric(DistanceMetric metric);

    double get_minkowski_p() const;
    void set_minkowski_p(double p);

    // Fit the model with training data
    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y);

    // Predict label for a single sample
    double predict(const Eigen::VectorXd &x) const;

    // Predict labels for multiple samples
    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const;
};