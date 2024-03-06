# Singapore-Flat-Resale-price-prediction

# Problem Statement:

The objective of this project is to develop a machine learning model and deploy it as a user-friendly web application that predicts the resale prices of flats in Singapore. This predictive model will be based on historical data of resale flat transactions, and it aims to assist both potential buyers and sellers in estimating the resale value of a flat.

# Prerequisites

Python -- Programming Language

pandas -- To Create a DataFrame with the scraped data

numpy -- Fundamental Python package for scientific computing in Python

streamlit -- Streamlit: A Python library used for creating interactive web applications and data visualizations.

scikit-learn -- Machine Learning library for the Python programming language

# Workflow

# Data Collection and 

Collect a dataset of resale flat transactions from the Singapore Housing and Development Board (HDB) for the years 1990 to Till Date.

Data Source : https://beta.data.gov.sg/collections/189/view

# Data Preprocessing

Handle missing values with mean/median/mode.

Treat Outliers using IQR or Isolation Forest from sklearn library.

Identify Skewness in the dataset and treat skewness with appropriate data transformations, such as log transformation

# Feature Engineering: 

Extract relevant features from the dataset, including town, flat type, storey range, floor area, flat model, and lease commence date.

Calculate latitude and longitude values for each MRT station and address(street and block number) usinf OneMapAPI.

These values can be used to obtain distance between nearest MRT and CBD which also impacts Resale price.

Refer: OneMapAPI - https://www.onemap.gov.sg/apidocs/

# Model Selection and Training:.

Split the dataset into training and testing/validation sets.

Choose an appropriate machine learning model for regression.

Train the model on the historical data.

Optimize model hyperparameters using techniques such as cross-validation and grid search to find the best-performing model.

Interpret the model results and assess its performance based on the defined problem statement.

# Model Evaluation: 

Evaluate the model's predictive performance using regression metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), 

or Root Mean Squared Error (RMSE) and R2 Score.

# Streamlit Web Application: 

Develop a user-friendly web application using Streamlit that allows users to input details of a flat (town, flat type, storey range, etc.). 

Utilize the trained machine learning model to predict the resale price based on user inputs.
