#include "LogisticRegression.hpp"
#include <spdlog/spdlog.h>
#include <random>
#include <cmath>

// ----------------------------
// Constructors & Destructor
// ----------------------------
LogisticRegression::LogisticRegression()
    : learning_rate(0.01),
      epochs(1000),
      fit_intercept(true),
      lambda(0.0),
      reg_type(RegularizationType::None),
      bias(0.0) {}

LogisticRegression::LogisticRegression(bool fit_intercept, double lr, int epochs, double reg_lambda, RegularizationType reg_type)
    : learning_rate(lr),
      epochs(epochs),
      fit_intercept(fit_intercept),
      lambda(lambda),
      reg_type(reg_type),
      bias(0.0) {}

LogisticRegression::~LogisticRegression() {}

// ----------------------------
// Getters
// ----------------------------
Eigen::VectorXd LogisticRegression::get_weights() const { return weights; }
double LogisticRegression::get_bias() const { return bias; }
std::vector<double> LogisticRegression::get_loss_history() const { return loss_history; }

// ----------------------------
// Setters
// ----------------------------
void LogisticRegression::set_learning_rate(double lr) { learning_rate = lr; }
void LogisticRegression::set_epochs(int epochs) { this->epochs = epochs; }
void LogisticRegression::set_fit_intercept(bool fit_intercept) { this->fit_intercept = fit_intercept; }
void LogisticRegression::set_regularization(double lambda, RegularizationType type)
{
    spdlog::debug("set_regularization called with lambda={}, type={}", lambda, static_cast<int>(type));
    this->lambda = lambda;
    reg_type = type;
}

// ----------------------------
// Internal Helpers
// ----------------------------
void LogisticRegression::initialize_weights(int n_features)
{
    spdlog::debug("initialize_weights called with n_features={}", n_features);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0, 0.01);

    weights = Eigen::VectorXd(n_features);
    for (int i = 0; i < n_features; ++i)
        weights(i) = d(gen);

    bias = 0.0;
    loss_history.clear();
    spdlog::debug("initialize_weights finished");
}

void LogisticRegression::gradient_descent(const Eigen::MatrixXd &X, const Eigen::VectorXd &y)
{
    spdlog::debug("gradient_descent called");
    int n_samples = X.rows();

    for (int i = 0; i < epochs; i++)
    {
        Eigen::VectorXd grads = compute_gradients(X, y);
        Eigen::VectorXd grad_w = grads.head(weights.size());
        double grad_b = fit_intercept ? grads(weights.size()) : 0.0;

        // Update step
        weights -= learning_rate * grad_w;
        if (fit_intercept)
            bias -= learning_rate * grad_b;

        loss_history.push_back(compute_loss(X, y));
    }
    spdlog::debug("gradient_descent finished");
}

void LogisticRegression::gradient_descent_batch(const Eigen::MatrixXd &X, const Eigen::VectorXd &y, int batch_size)
{
    spdlog::debug("gradient_descent_batch called with batch_size={}", batch_size);
    int n_samples = X.rows();
    std::vector<int> indices(n_samples);
    for (int i = 0; i < n_samples; i++)
        indices[i] = i;

    std::mt19937 rng(std::random_device{}());

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        std::shuffle(indices.begin(), indices.end(), rng);

        for (int start = 0; start < n_samples; start += batch_size)
        {
            int end = std::min(start + batch_size, n_samples);
            int current_batch_size = end - start;

            Eigen::MatrixXd X_batch(current_batch_size, X.cols());
            Eigen::VectorXd y_batch(current_batch_size);

            for (int i = 0; i < current_batch_size; i++)
            {
                X_batch.row(i) = X.row(indices[start + i]);
                y_batch(i) = y(indices[start + i]);
            }

            // Compute gradients
            Eigen::VectorXd grads = compute_gradients(X_batch, y_batch);
            Eigen::VectorXd grad_w = grads.head(weights.size());
            double grad_b = fit_intercept ? grads(weights.size()) : 0.0;

            // Update
            weights -= learning_rate * grad_w;
            if (fit_intercept)
                bias -= learning_rate * grad_b;
        }

        // Loss per epoch
        loss_history.push_back(compute_loss(X, y));
    }
    spdlog::debug("gradient_descent_batch finished");
}

Eigen::VectorXd LogisticRegression::compute_gradients(const Eigen::MatrixXd &X, const Eigen::VectorXd &y)
{
    spdlog::debug("compute_gradients called");
    int n_samples = X.rows();
    Eigen::VectorXd preds = predict_prob(X);
    Eigen::VectorXd errors = preds - y;

    // Gradient for weights
    Eigen::VectorXd grad_w = (X.transpose() * errors) / n_samples;
    grad_w += apply_regularization_gradients(weights);

    // Gradient for bias
    if (fit_intercept)
    {
        grad_w.conservativeResize(grad_w.size() + 1);
        grad_w(grad_w.size() - 1) = errors.mean();
    }

    spdlog::debug("compute_gradients finished");
    return grad_w;
}

Eigen::VectorXd LogisticRegression::sigmoid(const Eigen::VectorXd &z) const { return (1.0 / (1.0 + (-z.array()).exp())).matrix(); }

double LogisticRegression::apply_regularization_loss(const Eigen::VectorXd &w) const
{
    spdlog::debug("apply_regularization_loss called");
    if (reg_type == RegularizationType::L1)
        return lambda * w.array().abs().sum();
    else if (reg_type == RegularizationType::L2)
        return 0.5 * lambda * w.squaredNorm();

    return 0.0;
}

Eigen::VectorXd LogisticRegression::apply_regularization_gradients(const Eigen::VectorXd &w) const
{
    spdlog::debug("apply_regularization_gradients called");
    if (reg_type == RegularizationType::L1)
        return lambda * w.array().sign().matrix();
    else if (reg_type == RegularizationType::L2)
        return lambda * w;

    return Eigen::VectorXd::Zero(w.size());
}

// ----------------------------
// Public Methods
// ----------------------------
void LogisticRegression::fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y, bool verbose)
{
    spdlog::info("fit called");
    spdlog::info("X rows: {}, cols: {}", X.rows(), X.cols());
    spdlog::info("y size: {}", y.size());
    initialize_weights(X.cols());
    gradient_descent(X, y);

    if (verbose)
        spdlog::info("Training complete. Final loss = {}", loss_history.back());
    spdlog::debug("fit finished");
}

void LogisticRegression::fit_batch(const Eigen::MatrixXd &X, const Eigen::VectorXd &y, int batch_size, bool verbose)
{
    spdlog::info("fit_batch called");
    spdlog::info("X rows: {}, cols: {}", X.rows(), X.cols());
    spdlog::info("y size: {}", y.size());
    initialize_weights(X.cols());
    gradient_descent_batch(X, y, batch_size);

    if (verbose)
        spdlog::info("Batch Training complete. Final loss = {}", loss_history.back());
    spdlog::debug("fit_batch finished");
}

Eigen::VectorXd LogisticRegression::predict_prob(const Eigen::MatrixXd &X) const
{
    spdlog::debug("predict_prob called");
    Eigen::VectorXd linear_output = X * weights;
    if (fit_intercept)
        linear_output.array() += bias;

    return sigmoid(linear_output);
}

Eigen::VectorXi LogisticRegression::predict(const Eigen::MatrixXd &X, double threshold) const
{
    spdlog::debug("predict called with threshold={}", threshold);
    Eigen::VectorXd probs = predict_prob(X);
    Eigen::VectorXi preds(probs.size());

    for (int i = 0; i < probs.size(); i++)
        preds(i) = (probs(i) >= threshold) ? 1 : 0;

    return preds;
}

double LogisticRegression::compute_loss(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) const
{
    spdlog::debug("compute_loss called");
    Eigen::VectorXd y_pred = predict_prob(X);

    Eigen::VectorXd eps = Eigen::VectorXd::Constant(y_pred.size(), 1e-15);
    y_pred = y_pred.array().max(eps.array()).min(1 - eps.array());

    double bce_loss = -(y.array() * y_pred.array().log() + (1 - y.array()) * (1 - y_pred.array()).log()).mean();

    double total_loss = bce_loss + apply_regularization_loss(weights);
    spdlog::debug("compute_loss finished, loss={}", total_loss);
    return total_loss;
}