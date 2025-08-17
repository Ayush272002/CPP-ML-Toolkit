# ML-Algorithms C++/Python Project

This project is a collection of machine learning algorithms implemented in C++ with Python bindings via pybind11. It is designed for both educational and practical use, allowing you to experiment with and extend classic ML models efficiently.

## C++ Libraries Used

- **Eigen** — for linear algebra and matrix operations
- **spdlog** — for fast, modern logging

## Features
- High-performance C++ implementations
- Python bindings for easy experimentation and visualization
- Modern CMake build system with automatic dependency management (Eigen, spdlog, pybind11)
- Organized output and test structure

## Currently Implemented Algorithms
- **Linear Regression** (C++ core, Python interface, and comparison with scikit-learn)

## Directory Structure
```
ml-algo/
├── src/                # C++ source code
├── include/            # C++ headers
├── test/               # Python test and demo 
├── images/             # Output images (plots etc.)
├── data/               # Datasets
├── CMakeLists.txt      # Build configuration
├── .gitignore
└── README.md
```

## Getting Started
1. **Clone the repository**
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

## Contributing
Contributions for new algorithms, bug fixes, and improvements are welcome! Please open an issue or pull request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
