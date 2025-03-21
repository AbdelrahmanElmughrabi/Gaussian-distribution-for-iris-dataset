# Iris Gaussian Classifier

A simple implementation of a Gaussian classifier for the Iris dataset using Python. This classifier uses maximum likelihood estimation to fit multivariate Gaussian distributions to each iris class.

## Overview

The script:
1. Loads the Iris dataset (150 samples, 4 features, 3 classes)
2. Splits data into training (60 samples) and test sets
3. Fits a Gaussian distribution to each class
4. Classifies test samples using maximum likelihood
5. Outputs a confusion matrix for evaluation

## Dependencies

```sh
pip install numpy scipy scikit-learn matplotlib
```

## Usage

Run the script:
```sh
python iris_gaussian.py
```

## Output

The script outputs a 3x3 confusion matrix showing the classification results:
- Rows represent true classes
- Columns represent predicted classes
- Each cell shows the number of samples

Example:
```
[[20.  0.  0.]
 [ 0. 18.  2.]
 [ 0.  1. 19.]]
```

## Algorithm

1. For each class:
   - Compute mean vector (μ) and covariance matrix (Σ)
   - Use these to define class-conditional Gaussian distributions
2. Classify each test sample using maximum likelihood