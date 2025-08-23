# CPP-ML-Toolkit

This project is a collection of machine learning algorithms implemented in C++ with Python bindings via pybind11. It is designed for both educational and practical use, allowing you to experiment with and extend classic ML models efficiently.

## C++ Libraries Used

- **Eigen** — for linear algebra and matrix operations
- **spdlog** — for fast, modern logging and debugging

## Features
- **High-performance C++ implementations** with optimized matrix operations
- **Python bindings** for easy experimentation and visualization
- **Comprehensive logging** with debug and info levels for traceability
- **Visual comparisons** with scikit-learn implementations
- **Modern CMake build system** with automatic dependency management (Eigen, spdlog, pybind11)
- **Organized test structure** with performance metrics and plots

## Currently Implemented Algorithms
- **Linear Regression** — Full-batch and mini-batch gradient descent with regularization
- **Logistic Regression** — Binary classification with L1/L2 regularization support
- **K-Nearest Neighbors (KNN)** — Supports both classification and regression, multiple distance metrics (Euclidean, Manhattan, Minkowski)


## Directory Structure
```
cpp-ml-toolkit/
├── src/                # C++ source code
├── include/            # C++ headers
├── test/               # Python test and demo 
├── images/             # Output images (plots etc.)
├── data/               # Datasets
├── CMakeLists.txt      # Build configuration
├── .gitignore
└── README.md
```

## Performance Comparisons

The test scripts generate detailed performance comparisons with scikit-learn:

### Linear Regression
- **Dataset**: Boston Housing (normalized features)
- **Metrics**: Mean Squared Error (MSE)
- **Visualizations**: Loss curve during training, Predictions vs Actual scatter plot

### Logistic Regression  
- **Dataset**: Breast Cancer (highly imbalanced)
- **Metrics**: Accuracy and AUC (Area Under ROC Curve)
- **Visualizations**: Loss curve during training, Predicted probabilities comparison

### K-Nearest Neighbors (KNN)
- **Datasets**: Wine Quality
- **Metrics**: Accuracy, Mean Squared Error (MSE)
- **Visualizations**: Accuracy vs. k plot, Predictions vs Actual scatter plot


## Getting Started
1. **Clone the repository**:
    ```bash
    git clone https://github.com/Ayush272002/CPP-ML-Toolkit.git
    ```

2. **Install Python dependencies** (in your venv):
   ```bash
   pip install -r requirements.txt
   ```
3. **Install system dependencies** (for Ubuntu/Debian):
   ```bash
   sudo apt-get install libeigen3-dev python3-dev
   ```
4. **Build the project:**
   ```bash
   mkdir build && cd build
   cmake ..
   make
   ```
5. **Run Python demos/tests:**
   ```bash
   python3 test/LinearRegression.py
   ```

The tests will generate performance metrics and save comparison plots in the `images/` directory.

## Contributing
Contributions for new algorithms, bug fixes, and improvements are welcome! Please open an issue or pull request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
