import pandas as pd
import numpy as np

def preprocess_data(filepath):
    data = pd.read_csv(filepath)

    # Extract City from Location
    data['City'] = data['Location'].str.split(',').str[-1].str.strip()

    # Drop irrelevant columns
    drop_cols = ['Price', 'Description', 'Property Title', 'Name', 'Balcony', 'Location']
    data.drop(columns=drop_cols, errors='ignore', inplace=True)

    # Fill missing BHK values with median
    data['BHK'].fillna(data['BHK'].median(), inplace=True)

    # One-hot encode City column
    data = pd.get_dummies(data, columns=['City'], prefix='City')
    city_columns = [col for col in data.columns if col.startswith('City_')]

    # Log-transform skewed numeric features
    for col in ['Price_INR', 'Price_per_SQFT', 'Total_Area', 'BHK', 'Baths']:
        data[col] = np.log(data[col] + 1)

    # Compute city-level aggregate statistics
    city_avg_pricepsqft = {col: data.loc[data[col] == 1, 'Price_per_SQFT'].mean() for col in city_columns}
    city_avg_area = {col: data.loc[data[col] == 1, 'Total_Area'].mean() for col in city_columns}

    # Add aggregated features by multiplying one-hot city columns by aggregate stats
    data['City_Avg_Price_per_SQFT'] = data[city_columns].mul(pd.Series(city_avg_pricepsqft)).sum(axis=1)
    data['City_Avg_Area'] = data[city_columns].mul(pd.Series(city_avg_area)).sum(axis=1)

    # Interaction and ratio features
    data['Total_Rooms'] = data['Baths'] + data['BHK']
    data['Area_per_BHK'] = data['Total_Area'] / data['BHK']
    data['Area_per_Room'] = data['Total_Area'] / data['Total_Rooms']
    data['Baths_per_BHK'] = data['Baths'] / data['BHK']

    # Handle balcony feature
    data['Has_Balcony'] = data['Balcony_num'].astype(int)
    data.drop(columns=['Balcony_num'], inplace=True)

    return data, city_columns, city_avg_pricepsqft, city_avg_area

 
