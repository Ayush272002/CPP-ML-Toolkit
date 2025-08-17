#ifndef LINEAR_REGRESSION_HPP
#define LINEAR_REGRESSION_HPP

#include <Eigen/Dense>
#include <vector>
#include <string>

class LinearRegression
{
private:
    // ----------------------------
    // Model Parameters
    // ----------------------------
    Eigen::VectorXd weights; // weight vector
    double bias;

    // ----------------------------
    // Hyperparameters
    // ----------------------------
    double learning_rate;
    int epochs;           // number of iterations
    bool fit_intercept;   // whether to include bias
    double lambda;        // regularization strength
    std::string reg_type; // "none", "L1", "L2"

    // ----------------------------
    // Internal helpers
    // ----------------------------
    void initialize_weights(int n_features);
    void gradient_descent(const Eigen::MatrixXd &X, const Eigen::VectorXd &y);
    void gradient_descent_batch(const Eigen::MatrixXd &X, const Eigen::VectorXd &y, int batch_size);
    Eigen::VectorXd compute_gradient(const Eigen::MatrixXd &X, const Eigen::VectorXd &y);

    // History of loss during training
    std::vector<double> loss_history;

public:
    LinearRegression();
    LinearRegression(bool fit_intercept, double lr, int epochs);
    ~LinearRegression();

    void fit(const Eigen::MatrixXd &X,
             const Eigen::VectorXd &y,
             bool verbose = false);

    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const;

    double compute_loss(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) const;

    // getters
    Eigen::VectorXd get_weights() const;
    double get_bias() const;
    std::vector<double> get_loss_history() const;

    // setters
    void set_learning_rate(double lr);
    void set_epochs(int epochs);
    void set_fit_intercept(bool fit_intercept);
    void set_regularization(double lambda, const std::string &type = "none");

    void fit_batch(const Eigen::MatrixXd &X,
                   const Eigen::VectorXd &y,
                   int batch_size,
                   bool verbose = false);
};

#endif