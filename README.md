# Linear Regression (NumPy)

A simple implementation of **Linear Regression from scratch** using NumPy. This project demonstrates predicting continuous targets using gradient descent and visualizing results.

## Project Overview

- **Objective:** Predict a continuous variable (e.g., house prices, student scores) based on one or more features.
- **Approach:** Implement Linear Regression using NumPy without relying on libraries like scikit-learn.
- **Key Concepts:**
  - Linear hypothesis: `y_pred = X @ weights + bias`
  - Mean Squared Error (MSE) loss
  - Gradient Descent optimization

## Repository Structure

linear-regression/
├── README.md
├── data/
│ └── sample_data.csv # optional dataset
├── src/
│ ├── linear_regression.py # core implementation
│ ├── utils.py # helper functions (optional)
│ └── train.py # training / evaluation script
├── notebooks/
│ └── linear_regression_demo.ipynb # interactive demo and visualization
├── requirements.txt
└── .gitignore

## Getting Started

1. **Clone the repository**

```bash
git clone <your-repo-url>
cd linear-regression
```

1. **Install dependencies**

```bash
pip install -r requirements.txt
```

1. **Run training script**

```bash
python src/train.py
```

1. **Explore in Jupyter Notebook**

```bash
jupyter notebook notebooks/linear_regression_demo.ipynb
```

## Features

- Train linear regression model on custom datasets
- Compute and visualize training loss
- Predict continuous values on new inputs
- Optional: visualize 2D/3D data and best-fit line/plane

## License

This project is open for learning purposes.
