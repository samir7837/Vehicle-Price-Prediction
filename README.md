# 🚗 Vehicle Price Prediction — Machine Learning Project

## 🧭 Overview
This project predicts **vehicle prices** using machine learning models trained on structured vehicle specifications.  
It demonstrates an **end-to-end ML workflow** — from data exploration and preprocessing to model training, comparison, and saving the best model.

---

## 🧩 Dataset
The dataset includes features like:
- `make`, `model`, `year`, `mileage`, `fuel`, `transmission`, `engine`, `body`, `doors`, `colors`, and `drivetrain`  
- **Target:** `price` — the numeric vehicle market value (USD)

---

## ⚙️ Workflow
1. **EDA:** Correlation heatmaps, feature distributions, and categorical analysis.  
2. **Feature Engineering:**  
   - Extracted `displacement` and `horsepower` from `engine`  
   - Added text-based features (word counts from `name` and `description`)  
3. **Preprocessing:**  
   - Numeric: scaling and imputation  
   - Categorical: one-hot encoding  
   - Text: TF-IDF vectorization for textual features  
4. **Modeling:**  
   - Linear Regression  
   - Ridge Regression  
   - Random Forest  
   - XGBoost  
   - LightGBM  
5. **Evaluation Metrics:** MAE, RMSE, and R²  

---

## 📊 Model Performance

| Model | MAE ↓ | RMSE ↓ | R² ↑ |
|:------|-------:|-------:|------:|
| **Ridge** | **3983.57** | **6831.94** | **0.847** |
| XGBoost | 4360.32 | 7348.60 | 0.823 |
| Random Forest | 5125.24 | 8076.88 | 0.787 |
| LightGBM | 5092.10 | 8758.40 | 0.749 |
| Linear Regression | 23399.48 | 47202.34 | -6.296 |

---

## 🧠 Insights
- **Ridge Regression** performed best with an **R² of 0.85**, suggesting that regularized linear models capture the key relationships effectively.  
- **Tree-based models** (XGBoost, Random Forest) were close behind but showed mild overfitting.  
- **Linear Regression** failed due to multicollinearity and lack of regularization.  
- Important predictors: **engine power, year, mileage, and make/model**.  

---

## 🚀 Future Enhancements

Advanced hyperparameter tuning

SHAP & LIME explainability

Web deployment using Streamlit

Automated retraining pipeline for continuous learning
