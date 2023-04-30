# Boston Housing Price Prediction

This repository contains a Ruby script for predicting the median value of owner-occupied homes in the Boston area using the Boston Housing dataset. The script utilizes the Rumale library for machine learning and the Daru library for data manipulation.

## Dependencies

The script requires the following Ruby gems:

- matrix
- rumale
- daru
- http

You can install the dependencies using the following command:

```bash
gem install matrix rumale daru http
```

## How to Run

Run the script by executing the following command:

```bash
ruby boston_housing.rb
```

## Overview

The script performs the following steps:

1. Downloads the Boston Housing dataset from a remote URL if it doesn't already exist in the working directory.
2. Loads the dataset using the Daru library.
3. Prepares the data by separating the features (X) from the target variable (median home value, y).
4. Splits the data into training and testing sets with an 80-20 ratio, using a fixed random seed (43) for reproducibility.
5. Trains a Ridge Regression model (with regularization parameter set to 0, making it equivalent to Linear Regression) using the Rumale library.
6. Makes predictions on the test set.
7. Evaluates the model's performance by calculating Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared score.

## Output

The script outputs the following evaluation metrics:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R-squared Score

These metrics help assess the performance of the model on the test dataset. Lower values for MAE and MSE indicate better model performance, while an R-squared score closer to 1 indicates a better fit of the model to the data.