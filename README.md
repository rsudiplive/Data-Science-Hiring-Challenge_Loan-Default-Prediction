# FluidAI-Data-Science-Hiring-Challenge_Loan-Default-Default-Prediction
The task is to create a model with detailed EDA, that can predict whether a customer will possibly default on taking a loan/not. The goal is to find out if a customer has the potential of repaying the loan amount back!

1. Performed data cleaning steps like removing low-range outliers, missing value treatment with the detailed EDA, feature engineering steps like onehot encoding, normalization & scaling to transform the features.
2. Implemented LightGBM on the imbalanced dataset with an accuracy of 75%, having ROC-AUC scores to 64% qualifying to be a good model. With AUC scores for both train and test near to the 1 which means it has a good measure of separability.
3. Implemented other baseline models such as logistic regression, Random forest, Adaboost, XGBoost with an accuracy of 91% .
4. Tried sampling techniques for handling imbalance and then applying baseline models on top of it and checking the performance.

There are many other techniques that can be applied on this imbalanced dataset like PCA and KNN imputer for handling missing values. Also we could have performed feature scaling, perform sampling and then apply a simple ANN model to figure out the accuracy curves and checking the model performance across train and test. 
