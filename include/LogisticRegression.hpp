#ifndef LOGISTIC_REGRESSION_HPP
#define LOGISTIC_REGRESSION_HPP

#include <Eigen/Dense>
#include <vector>
#include <string>

enum class RegularizationType
{
    None,
    L1,
    L2
};

class LogisticRegression
{
private:
    // ----------------------------
    // Model Parameters
    // ----------------------------
    Eigen::VectorXd weights; // model weights
    double bias;             // bias/intercept

    // ----------------------------
    // Hyperparameters
    // ----------------------------
    double learning_rate;
    int epochs;                  // number of iterations
    bool fit_intercept;          // whether to include bias
    double lambda;               // regularization strength
    RegularizationType reg_type; // regularization type

    // ----------------------------
    // Internal Helpers
    // ----------------------------
    void initialize_weights(int n_features);

    void gradient_descent(const Eigen::MatrixXd &X, const Eigen::VectorXd &y);
    void gradient_descent_batch(const Eigen::MatrixXd &X, const Eigen::VectorXd &y, int batch_size);

    Eigen::VectorXd compute_gradients(const Eigen::MatrixXd &X, const Eigen::VectorXd &y);

    Eigen::VectorXd sigmoid(const Eigen::VectorXd &z) const;

    double apply_regularization_loss(const Eigen::VectorXd &w) const;

    Eigen::VectorXd apply_regularization_gradients(const Eigen::VectorXd &w) const;

    std::vector<double> loss_history;

public:
    // ----------------------------
    // Constructors & Destructor
    // ----------------------------
    LogisticRegression();
    LogisticRegression(bool fit_intercept, double lr, int epochs, double reg_lambda = 0.0, RegularizationType reg_type = RegularizationType::None);
    ~LogisticRegression();

    // ----------------------------
    // Getters
    // ----------------------------
    Eigen::VectorXd get_weights() const;
    double get_bias() const;
    std::vector<double> get_loss_history() const;

    // ----------------------------
    // Setters
    // ----------------------------
    void set_learning_rate(double lr);
    void set_epochs(int epochs);
    void set_fit_intercept(bool fit_intercept);
    void set_regularization(double lambda, RegularizationType type);

    // ----------------------------
    // Public Methods
    // ----------------------------
    void fit(const Eigen::MatrixXd &X,
             const Eigen::VectorXd &y,
             bool verbose = false);

    void fit_batch(const Eigen::MatrixXd &X,
                   const Eigen::VectorXd &y,
                   int batch_size,
                   bool verbose = false);

    // Predict probabilities (sigmoid outputs in [0,1])
    Eigen::VectorXd predict_prob(const Eigen::MatrixXd &X) const;

    // Predict class labels (0 or 1)
    Eigen::VectorXi predict(const Eigen::MatrixXd &X, double threshold = 0.5) const;

    // Compute Binary Cross-Entropy Loss
    double compute_loss(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) const;
};

#endif