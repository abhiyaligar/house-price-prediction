import pandas as pd
import numpy as np
import joblib

# Load model and metadata once
stats = joblib.load("model_and_stats.pkl")
model = stats['model']
feature_columns = stats['feature_columns']
city_avg_pricepsqft = stats['avg_pricepsqft']
city_avg_area = stats['avg_area']
city_list = [c.replace("City_", "") for c in stats['city_columns']]

def get_float_input(prompt, min_value=0, allow_zero=False):
    while True:
        try:
            val = float(input(prompt))
            if val < min_value or (not allow_zero and val == 0):
                print(f"❌ Enter a value greater than {min_value}")
                continue
            return val
        except ValueError:
            print("❌ Invalid number, please try again.")

def get_int_input(prompt, allowed_values=None):
    while True:
        try:
            val = int(input(prompt))
            if allowed_values and val not in allowed_values:
                print(f"❌ Please enter one of {allowed_values}")
                continue
            return val
        except ValueError:
            print("❌ Invalid integer, please try again.")

def get_city_input(city_list):
    while True:
        city = input(f"Enter city name ({', '.join(city_list)}): ").strip()
        if city.lower() in [c.lower() for c in city_list]:
            return city
        print("❌ City not recognized. Please enter a valid city name.")

def predict_house_price_chat():
    price_per_sqft = get_float_input("Enter Price per SQFT (INR): ", min_value=1)
    total_area = get_float_input("Enter Total Area (sqft): ", min_value=100)
    bhk = get_float_input("Enter number of Bedrooms (BHK): ", min_value=1)
    baths = get_float_input("Enter number of Bathrooms: ", min_value=1)
    has_balcony = get_int_input("Balcony? 1=Yes, 0=No: ", allowed_values=[0, 1])
    city = get_city_input(city_list)

    avg_area = city_avg_area.get(f'City_{city}', 0)
    avg_price = city_avg_pricepsqft.get(f'City_{city}', 0)

    user_data = pd.DataFrame([{
        'Total_Area': np.log(total_area + 1),
        'Price_per_SQFT': np.log(price_per_sqft + 1),
        'Baths': np.log(baths + 1),
        'BHK': np.log(bhk + 1),
        'Has_Balcony': has_balcony,
        'City_Avg_Area': avg_area,
        'City_Avg_Price_per_SQFT': avg_price
    }])

    # Build derived features
    user_data['Total_Rooms'] = user_data['Baths'] + user_data['BHK']
    user_data['Area_per_BHK'] = user_data['Total_Area'] / user_data['BHK']
    user_data['Area_per_Room'] = user_data['Total_Area'] / user_data['Total_Rooms']
    user_data['Baths_per_BHK'] = user_data['Baths'] / user_data['BHK']

    # One-hot encode city columns to match training features
    for c in city_list:
        user_data[f'City_{c}'] = int(c.lower() == city.lower())

    # Reorder columns to training order
    user_data = user_data[feature_columns]

    log_price = model.predict(user_data)[0]
    predicted_price = np.exp(log_price) - 1
    print(f"\n💰 Predicted House Price: ₹{predicted_price:,.0f}")

if __name__ == "__main__":
    predict_house_price_chat()
