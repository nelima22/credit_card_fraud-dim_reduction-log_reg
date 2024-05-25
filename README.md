# PCA and Logistic Regression for Credit Card Fraud Detection

## Objective
The objective of this project is to demonstrate the application of Principal Component Analysis (PCA) for dimensionality reduction followed by logistic regression for detecting fraudulent credit card transactions. We aim to show how PCA can simplify the dataset while retaining its most important features and subsequently use logistic regression to classify the data.

## Steps

### 1. Load the Dataset
We use the Credit Card Fraud Detection dataset from Kaggle, which contains transactions made by credit cards in September 2013 by European cardholders. The dataset includes 284,807 transactions with 492 cases of fraud.

### 2. Analyze and preprocess the Data
We visualize the class distribution:

![Alt text](https://github.com/nelima22/credit_card_fraud-dim_reduction-log_reg/blob/main/download%20(1).png)


Class imbalance is observed but will only be addressed by using `stratify = y` for this project.

The dataset is split into features and target (fraud or not fraud). The features are then standardized to have zero mean and unit variance, which is important for PCA. 

### 3. Apply PCA for Dimensionality Reduction
PCA is applied to reduce the dimensionality of the dataset.



![Alt text](https://github.com/nelima22/credit_card_fraud-dim_reduction-log_reg/blob/main/download.png)

Class imbalance is observed but will only be addressed by using `stratify = y` for this project.)
This plot shows the cumulative explained variance for all components.

We reduce the dataset to 24 out of 30 principal components from which 85% of the variance is explained.This step is crucial to simplify the dataset while retaining the most important information.

### 4. Train a Logistic Regression Model
A logistic regression model is trained on the transformed training data. Logistic regression is chosen for its simplicity and effectiveness in classification tasks.

### 5. Evaluate the Model
The trained model is used to make predictions on the test set. We evaluate the model using accuracy, confusion matrix, and classification report to understand its performance.

## Results
The results of the model, including accuracy, confusion matrix, and classification report, are printed out. These metrics help us understand how well the model is performing on the test set.


We see that the model has high precision and recall for Class `0` but low precision and even lower recall. 
The low recall means that we have a high number of false negatives as reflected in the confusion matrix. 
This is not good in our case because our model will be marking actual fraud cases as not fraud, causing out clients to money.
This occurence is most likely as a result of class imbalance. The model would need more observations for class 1
We can do this by applying upsampling or downsampling techniques on our data to try and balance our classes.
Additionally we can adjust our hyperparameters for our model.
We can also train other models on the data to see if they would perform better and choose the best from them.

## Code
The code for the project is provided in the file `creditcard_fraud_pca_log_reg.ipynb`. It includes all the necessary steps to load the data, preprocess it, apply PCA, train a logistic regression model, and evaluate the model.

## Conclusion
This project demonstrates a simple yet effective approach to dimensionality reduction using PCA and subsequent classification using logistic regression. It highlights the importance of preprocessing steps like standardization and shows how PCA can be used to retain essential features while reducing the complexity of the data.

## References
- [Credit Card Fraud Detection Dataset on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
