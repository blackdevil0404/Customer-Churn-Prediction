# Customer Churn Prediction using AI/ML

## Objective
The objective of this project is to predict whether a customer will discontinue a subscription-based service. Historical customer data is analyzed, considering factors like usage patterns, demographic details, and subscription duration. 

## Dataset
The dataset used is `Churn_Modelling.csv`, which contains customer details, including demographics, account information, and churn status. 

## Key Features
- **Missing Data Handling**: Appropriate imputation strategies are applied to deal with missing values.
- **Data Preprocessing**:
  - Removal of unnecessary columns (`RowNumber`, `CustomerId`, `Surname`).
  - Encoding categorical variables (`Geography`, `Gender`).
  - Scaling numerical features using `StandardScaler`.
- **Model Training**:
  - Random Forest Classifier.
  - XGBoost Classifier.
- **Model Evaluation**:
  - Accuracy, classification report, and confusion matrix.
  - Feature importance analysis for XGBoost.

## Dependencies
Ensure you have the following Python libraries installed in Google Colab:

```python
!pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

## Running in Google Colab
1. Upload the dataset to your Google Drive.
2. Mount your Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

3. Load the dataset from the appropriate path:

```python
import pandas as pd
file_path = "/content/drive/MyDrive/Churn_Modelling.csv"
df = pd.read_csv(file_path)
```

4. Execute the provided script in Google Colab.

## Project Structure
```
|-- customer_churn_prediction
    |-- Churn_Modelling.csv  # Dataset
    |-- churn_prediction.py  # Main script
    |-- README.md            # Project documentation
```

## Expected Outcome
- A model that effectively identifies customers likely to leave.
- Insights into the most significant factors contributing to churn.
- Visualizations of confusion matrices and feature importance.

## Results
- Random Forest Model Evaluation
```
    Random Forest Model Evaluation
    Accuracy: 0.864
    Classification Report:
                   precision    recall  f1-score   support
    
               0       0.87      0.97      0.92      1593
               1       0.78      0.46      0.58       407
    
        accuracy                           0.86      2000
       macro avg       0.83      0.71      0.75      2000
    weighted avg       0.86      0.86      0.85      2000
```
- XGBoost Model Evaluation
```
  XGBoost Model Evaluation
  Accuracy: 0.865
  Classification Report:
                 precision    recall  f1-score   support
  
             0       0.88      0.96      0.92      1593
             1       0.78      0.47      0.59       407
  
      accuracy                           0.86      2000
     macro avg       0.83      0.72      0.75      2000
  weighted avg       0.86      0.86      0.85      2000
```

- Confusion matrices are plotted using Seaborn.
  - Random Forest Model.<br>
    ![Figure_1](https://github.com/user-attachments/assets/400c0e69-ee44-40dc-8cee-2d833580fb8f)

  - XGBoost Model.<br>
    ![Figure_2](https://github.com/user-attachments/assets/2d86b423-335a-4780-8a50-0e78d9094462)


- Feature importance for XGBoost will be displayed.
![Figure_3](https://github.com/user-attachments/assets/dcb71253-8849-4547-9dfb-a643b61330a8)


## Author
Vrutwic Sangare

## License
This project is licensed under the MIT License.
