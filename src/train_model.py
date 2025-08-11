import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from preprocess import preprocess_data

def plot_model_performance_comparison(results):
    """
    Plot side-by-side bar charts comparing model R2, MAE, and RMSE.
    """
    model_names = list(results.keys())
    r2_scores = [results[m]['R2'] for m in model_names]
    mae_scores = [results[m]['MAE'] for m in model_names]
    rmse_scores = [results[m]['RMSE'] for m in model_names]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].bar(model_names, r2_scores, color=['skyblue', 'salmon', 'lightgreen'])
    axes[0].set_title('R² Score Comparison')
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel('R² Score')

    axes[1].bar(model_names, mae_scores, color=['skyblue', 'salmon', 'lightgreen'])
    axes[1].set_title('Mean Absolute Error (MAE) Comparison')
    axes[1].set_ylabel('MAE (INR)')

    axes[2].bar(model_names, rmse_scores, color=['skyblue', 'salmon', 'lightgreen'])
    axes[2].set_title('Root Mean Squared Error (RMSE) Comparison')
    axes[2].set_ylabel('RMSE (INR)')

    for ax in axes:
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    plt.suptitle('Model Performance Comparison')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def train_and_save_model(data_path):
    # Load and preprocess data
    data, city_columns, city_avg_pricepsqft, city_avg_area = preprocess_data(data_path)

    # Define feature columns explicitly
    feature_columns = [
        'Total_Area', 'Price_per_SQFT', 'Baths', 'BHK', 'Has_Balcony',
        'City_Avg_Area', 'City_Avg_Price_per_SQFT',
        'Total_Rooms', 'Area_per_BHK', 'Area_per_Room', 'Baths_per_BHK'
    ] + city_columns

    X, y = data[feature_columns], data['Price_INR']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define multiple models for comparison
    models = {
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42)
    }

    results = {}

    print("\n📊 Model Performance Comparison:")
    print("-" * 50)
    for name, mdl in models.items():
        mdl.fit(X_train, y_train)
        y_pred = mdl.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(np.exp(y_test) - 1, np.exp(y_pred) - 1)
        mse = mean_squared_error(np.exp(y_test) - 1, np.exp(y_pred) - 1)
        rmse = np.sqrt(mse)

        results[name] = {"R2": r2, "MAE": mae, "RMSE": rmse}
        print(f"{name:17} | R²: {r2:.4f} | MAE: ₹{mae:,.0f} | RMSE: ₹{rmse:,.0f}")

    # Plot the comparison
    plot_model_performance_comparison(results)

    # Save only the Random Forest model along with metadata
    final_model = models["Random Forest"]
    joblib.dump({
        'model': final_model,
        'feature_columns': feature_columns,
        'avg_pricepsqft': city_avg_pricepsqft,
        'avg_area': city_avg_area,
        'city_columns': city_columns
    }, "model_and_stats.pkl")

    print("\n✅ Random Forest model saved to model_and_stats.pkl")

if __name__ == "__main__":
    train_and_save_model("C:/Users/91974/Desktop/Project Housing Price Predictor/Data/merged_housing_data.csv")
  # Update path as necessary
