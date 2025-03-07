Salary Prediction Model
Produced by: Kirubel Temesgen
College ID: C00260396
Date: 10/11/2024

Overview
This project utilises Multiple Linear Regression to predict employee salaries based on business sectors and an additional user-selected parameter, such as region or business size. 
The model is deployed using Flask, allowing users to upload datasets, select relevant parameters, and visualise salary predictions.

Features
Upload CSV Files – Users can upload salary datasets with business sector and other attributes.
Predict Salaries – Model predicts salaries based on selected parameters.
Visualisations – Scatter plots show actual vs predicted salaries, categorised by business sector.
Error Handling – Redirects users if the model’s accuracy is too low.
Flask Web Application – Interactive web-based interface.

Installation
1. Clone the Repository
Run the following command in your terminal:

git clone https://github.com/KirubelCode/Salary-Prediction-Model.git
cd Salary-Prediction-Model

2. Install Dependencies
Ensure you have Python 3.x installed, then install the required libraries

3. Run the Application
python app.py

The application will be available at: http://127.0.0.1:5000/

Usage
Upload a CSV dataset with salary, business sector, and other relevant attributes.
Select salary and sector columns for prediction.
(Optional) Filter results based on business size or region.
View predictions and performance metrics (MSE, R² Score).
Visualise scatter plots of actual vs predicted salaries.

Technologies Used
Python
Flask (Web framework)
Scikit-Learn (Machine Learning)
Pandas (Data handling)
Matplotlib (Visualisation)
