#include <sstream>
#include "LinearRegression.hpp"
#include <cmath>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <random>  

// ----------------------------
// Constructors & Destructor
// ----------------------------
LinearRegression::LinearRegression()
    : learning_rate(0.01), epochs(1000), fit_intercept(true), lambda(0.0), reg_type("none"), bias(0.0) {}

LinearRegression::LinearRegression(bool fit_intercept_, double lr, int epochs_)
    : learning_rate(0.0), epochs(0), fit_intercept(false), lambda(0.0), reg_type("none"), bias(0.0)
{
    learning_rate = lr;
    epochs = epochs_;
    fit_intercept = fit_intercept_;
    lambda = 0.0;
    reg_type = "none";
    bias = 0.0;

    weights = Eigen::VectorXd();
    loss_history.clear();
}

LinearRegression::~LinearRegression() {}

// ----------------------------
// Getters
// ----------------------------
Eigen::VectorXd LinearRegression::get_weights() const { return weights; }
double LinearRegression::get_bias() const { return bias; }
std::vector<double> LinearRegression::get_loss_history() const { return loss_history; }

// ----------------------------
// Setters
// ----------------------------
void LinearRegression::set_learning_rate(double lr) { learning_rate = lr; }
void LinearRegression::set_epochs(int e) { epochs = e; }
void LinearRegression::set_fit_intercept(bool fit) { fit_intercept = fit; }
void LinearRegression::set_regularization(double l, const std::string &type)
{
    lambda = l;
    reg_type = type;
}

// ----------------------------
// Private Helpers
// ----------------------------
void LinearRegression::initialize_weights(int n_features)
{
    spdlog::info("initialize_weights called with n_features={}", n_features);
    weights = Eigen::VectorXd::Zero(n_features);
    bias = 0.0;
    loss_history.clear();
    spdlog::info("initialize_weights finished");
}

Eigen::VectorXd LinearRegression::compute_gradient(const Eigen::MatrixXd &X, const Eigen::VectorXd &y)
{
    spdlog::info("compute_gradient called");
    int n_samples = X.rows();
    Eigen::VectorXd y_pred = predict(X);
    Eigen::VectorXd error = y_pred - y;

    Eigen::VectorXd grad_w = (X.transpose() * error) / n_samples;
    double grad_b = 0.0;
    if (fit_intercept)
        grad_b = error.mean();

    if (reg_type == "L2")
        grad_w += lambda * weights;
    else if (reg_type == "L1")
        grad_w += lambda * weights.array().sign().matrix();

    if (fit_intercept)
    {
        Eigen::VectorXd grad(weights.size() + 1);
        grad.head(weights.size()) = grad_w;
        grad(weights.size()) = grad_b;
        spdlog::info("compute_gradient finished (with intercept)");
        return grad;
    }
    else
    {
        spdlog::info("compute_gradient finished (no intercept)");
        return grad_w;
    }
}

void LinearRegression::gradient_descent(const Eigen::MatrixXd &X, const Eigen::VectorXd &y)
{
    spdlog::info("gradient_descent called");
    int n_samples = X.rows();
    for (int i = 0; i < epochs; ++i)
    {
        if (i % 100 == 0)
            spdlog::info("gradient_descent epoch {}", i);
        Eigen::VectorXd y_pred = predict(X);
        Eigen::VectorXd error = y_pred - y;

        Eigen::VectorXd grad_w = (X.transpose() * error) / n_samples;
        double grad_b = fit_intercept ? error.mean() : 0.0;

        // regularization
        if (reg_type == "L2")
            grad_w += lambda * weights;
        else if (reg_type == "L1")
            grad_w += lambda * weights.array().sign().matrix();

        weights -= learning_rate * grad_w;
        if (fit_intercept)
            bias -= learning_rate * grad_b;

        loss_history.push_back(compute_loss(X, y));
    }
    spdlog::info("gradient_descent finished");
}

void LinearRegression::gradient_descent_batch(const Eigen::MatrixXd &X, const Eigen::VectorXd &y, int batch_size)
{
    spdlog::info("gradient_descent_batch called");
    int n_samples = X.rows();
    for (int i = 0; i < epochs; ++i)
    {
        // Shuffle indices for stochasticity
        std::vector<int> indices(n_samples);
        for (int j = 0; j < n_samples; ++j)
            indices[j] = j;

        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);

        for (int start = 0; start < n_samples; start += batch_size)
        {
            int end = std::min(start + batch_size, n_samples);
            int curr_batch_size = end - start;

            Eigen::MatrixXd X_batch(curr_batch_size, X.cols());
            Eigen::VectorXd y_batch(curr_batch_size);

            for (int j = 0; j < curr_batch_size; ++j)
            {
                X_batch.row(j) = X.row(indices[start + j]);
                y_batch(j) = y(indices[start + j]);
            }

            Eigen::VectorXd y_pred = predict(X_batch);
            Eigen::VectorXd error = y_pred - y_batch;

            Eigen::VectorXd grad_w = (X_batch.transpose() * error) / curr_batch_size;
            double grad_b = fit_intercept ? error.mean() : 0.0;

            // Regularization
            if (reg_type == "L2")
                grad_w += lambda * weights;

            else if (reg_type == "L1")
                grad_w += lambda * weights.array().sign().matrix();

            // Update parameters
            weights -= learning_rate * grad_w;
            if (fit_intercept)
                bias -= learning_rate * grad_b;
        }

        loss_history.push_back(compute_loss(X, y));
    }
    spdlog::info("gradient_descent_batch finished");
}

// ----------------------------
// Public Methods
// ----------------------------
void LinearRegression::fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y, bool verbose)
{
    spdlog::info("fit called");
    spdlog::info("X rows: {}, cols: {}", X.rows(), X.cols());
    spdlog::info("y size: {}", y.size());
    initialize_weights(X.cols());
    gradient_descent(X, y);
    if (verbose)
    {
        double final_loss = compute_loss(X, y);
        spdlog::info("Training completed. Final loss: {}", final_loss);
    }
    spdlog::info("fit finished");
}

void LinearRegression::fit_batch(const Eigen::MatrixXd &X, const Eigen::VectorXd &y, int batch_size, bool verbose)
{
    spdlog::info("fit_batch called");
    initialize_weights(X.cols());
    gradient_descent_batch(X, y, batch_size);
    if (verbose)
    {
        double final_loss = compute_loss(X, y);
        spdlog::info("Batch training completed. Final loss: {}", final_loss);
    }
    spdlog::info("fit_batch finished");
}

Eigen::VectorXd LinearRegression::predict(const Eigen::MatrixXd &X) const
{
    spdlog::info("predict called");
    Eigen::VectorXd y_pred = X * weights;
    if (fit_intercept)
        y_pred.array() += bias;

    spdlog::info("predict finished");
    return y_pred;
}

double LinearRegression::compute_loss(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) const
{
    spdlog::info("compute_loss called");
    Eigen::VectorXd y_pred = predict(X);
    double mse = (y - y_pred).squaredNorm() / y.size();

    if (reg_type == "L2")
        mse += (lambda / 2.0) * weights.squaredNorm();

    else if (reg_type == "L1")
        mse += lambda * weights.lpNorm<1>();

    spdlog::info("compute_loss finished");
    return mse;
}