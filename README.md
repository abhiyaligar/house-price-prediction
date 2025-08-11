# 🏠 Indian Housing Price Prediction

Predict housing prices across major Indian cities using a robust machine learning pipeline — featuring clean preprocessing, advanced feature engineering, model comparison, and an interactive CLI tool for predictions.

---

## 🚀 Features
- ✅ **Multi-city support**: Mumbai, Pune, Bangalore, Chennai, Hyderabad, Kolkata, Thane
- ✅ **Advanced feature engineering**: city-based averages, derived ratios, interaction terms
- ✅ **Preprocessing pipeline**: missing values handled, log-transform for skewed variables, one-hot city encoding
- ✅ **Model comparison**: Decision Tree, Random Forest, Gradient Boosting
- ✅ **Visual evaluation**: R², MAE, RMSE plots
- ✅ **Interactive CLI prediction**: robust user input validation (no crashes on typos)
- ✅ **Feature importance plots** for interpretability


---
## ⚙️ Setup Instructions

### 1️⃣ Clone the repository
```bash
git clone https://github.com/yourusername/housing-price-prediction.git
cd housing-price-prediction
```
### 2️⃣ Create & activate virtual environment
```
python -m venv env
# Activate:
# Windows (cmd):
env\Scripts\activate
# macOS/Linux:
source env/bin/activate
```

### 3️⃣ Install dependencies
```
pip install -r requirements.txt
```

### 4️⃣ Add dataset
Place merged_housing_data.csv inside the /data folder.
Ensure column names match those used in preprocess.py.

### Train & Compare Models From the project root:
```
python src/train_model.py
```
- This will:
- Preprocess the dataset
- Train Decision Tree, Random Forest, Gradient Boosting
- Compare models visually & with metrics
- Save the best model in model_and_stats.pkl

### Example output:
```
📊 Model Performance Comparison:
--------------------------------------------------
Decision Tree      | R²: 0.65 | MAE: ₹1,450,210 | RMSE: ₹5,210,000
Random Forest      | R²: 0.82 | MAE: ₹921,800   | RMSE: ₹3,110,450
Gradient Boosting  | R²: 0.88 | MAE: ₹843,190   | RMSE: ₹2,600,700

✅ Random Forest model saved to model_and_stats.pkl
```

### Interactive Price Prediction
Once your model is trained and saved, run:
```
python src/predict.py
```

You’ll be prompted:
```
 Enter Price per SQFT (INR): 9000
 Enter Total Area (sqft): 1420
 Enter number of Bedrooms (BHK): 3
 Enter number of Bathrooms: 2
 Balcony? 1=Yes, 0=No: 1
 Enter city name (Mumbai, Pune, Bangalore, ...): Bangalore

💰 Predicted House Price: ₹13,215,470
```
### CLI Tool Highlights:
- Validates numeric & categorical input
- Prevents invalid city names
- Matches training pipeline’s preprocessing
- Produces realistic INR prices

### Visualizations
- Model performance comparison chart
- Feature importance plot for interpretability:
```
from src.train_model import plot_feature_importance
plot_feature_importance(model, feature_columns)
```
📝 Notes
- model_and_stats.pkl must be present before running predictions.
- To add new cities, update preprocess.py and retrain the model.


---
📄 License
- MIT License

