# Test04 – Supervised Machine Learning Assignment  
## House Price Prediction

## Project Title
House Price Prediction Using Supervised Machine Learning Algorithms

---

## Problem Statement
The objective of this project is to predict house prices based on various housing features using supervised machine learning techniques. Multiple regression models are implemented, trained, tested, and compared to evaluate their performance on unseen data.

---

## Dataset Description
The dataset contains 4,600 records with information related to residential houses.  
Each record includes numerical and categorical features such as number of bedrooms, bathrooms, living area size, location details, construction year, and other property characteristics.

**Target Variable:**
- `price` – Continuous value representing the house price.

**Feature Types:**
- Numerical: bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view, condition, sqft_above, sqft_basement, yr_built, yr_renovated
- Categorical: city, statezip
- Date: date (used for feature extraction)

---

## Data Cleaning and Preprocessing Steps
Before model training, the following preprocessing steps were applied:

1. **Missing Value Check**  
   - Verified that the dataset contained no missing values.

2. **Duplicate Removal**  
   - Duplicate records were removed to avoid bias.

3. **Date Feature Engineering**  
   - Converted the `date` column to datetime format.
   - Extracted `year` and `month` features.
   - Dropped the original `date` column.

4. **Removal of Irrelevant Features**  
   - Dropped `street` (high cardinality, low predictive value).
   - Dropped `country` (constant value).

5. **Outlier Treatment**  
   - Outliers in the target variable (`price`) were treated using the IQR method.

6. **Categorical Encoding**  
   - Applied One-Hot Encoding to categorical features.

7. **Feature Scaling**  
   - Standardized numerical features using StandardScaler.

8. **Train-Test Split**  
   - Dataset split into 80% training and 20% testing data.

---

## Algorithms Used
The following five supervised machine learning algorithms were implemented:

1. Linear Regression  
2. Decision Tree Regressor  
3. Random Forest Regressor  
4. K-Nearest Neighbors (KNN) Regressor  
5. Support Vector Machine (SVR)

---

## Evaluation Metrics
Since this is a regression problem, the models were evaluated using:

- R² Score  
- Mean Squared Error (MSE)  
- Root Mean Squared Error (RMSE)  
- Mean Absolute Error (MAE)

---

## Model Performance Results

| Model               | R² Score | RMSE       | MAE        |
|--------------------|----------|------------|------------|
| Linear Regression  | 0.7009   | 121,736.91 | 76,751.20  |
| Decision Tree      | 0.4509   | 164,941.74 | 110,012.01 |
| Random Forest      | 0.6734   | 127,212.48 | 83,718.03  |
| KNN                | 0.5619   | 147,330.16 | 99,939.64  |
| SVM                | -0.0362  | 226,585.31 | 177,512.21 |

---

## Visualization
A scatter plot of **Actual vs Predicted Prices** was used to visually assess model performance.  
The plot showed a clear positive linear trend, indicating that the models successfully learned the relationship between features and house prices.

---

## Conclusion
- Linear Regression achieved the best overall performance with the highest R² score and lowest error metrics.
- Random Forest performed competitively but did not outperform the linear model, indicating a largely linear relationship in the data.
- Decision Tree showed signs of overfitting.
- KNN produced moderate results.
- SVM performed poorly due to sensitivity to hyperparameters and kernel selection.

Overall, the results demonstrate effective data preprocessing and successful application of supervised learning models for house price prediction.
