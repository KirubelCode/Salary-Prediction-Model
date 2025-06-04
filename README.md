Salary Prediction Model
A Flask-based web application that uses Multiple Linear Regression to predict employee salaries based on industry sector and one additional factor (such as region or company size). This project is built with real-world usability in mind – users can upload their own salary dataset, choose parameters, and get instant salary predictions along with visualisations and accuracy metrics. It provides an intuitive way for HR teams and data analysts to explore salary patterns and make data-driven decisions. 

--------

Features
- CSV Data Upload: Upload a custom dataset (CSV file) containing salary information, business sectors, and other relevant features.
- Predictive Modeling: Select the salary column, the sector column, and an optional second factor (e.g. geographic region or company size). The app will train a Multiple Linear Regression model on the fly to predict salaries based on these inputs.
- Interactive Visualisations: After prediction, view a scatter plot of Actual vs Predicted salaries, color-coded by business sector. This helps visualise the model’s performance at a glance.
- Performance Metrics: The application displays key regression metrics such as Mean Squared Error (MSE) and R² score, so you can evaluate how well the model fits your data. If the model’s accuracy (R²) is below an acceptable threshold, the app will alert you (and redirect if necessary) to ensure results are meaningful.
- User-Friendly Web Interface: Built with Flask, the system provides a clean web UI. It guides you through uploading your data, selecting parameters, and viewing results, making the tool accessible even to those with minimal programming experience.


-------

Quick Setup
- git clone https://github.com/KirubelCode/Salary-Predicition-Model
- cd Salary-Predicition-Model
- Install Dependencies
- python SalaryPrediction.py
  
Then go to http://127.0.0.1:5000/ in your browser.


------

Project Structure
- Salary Prediction/ – App + ML code (Flask + sklearn)

- TestData/ – Sample datasets

- Functional Specification/ – Full system doc

------

Tech Used
Python · Flask · scikit-learn · Pandas · Matplotlib ..

