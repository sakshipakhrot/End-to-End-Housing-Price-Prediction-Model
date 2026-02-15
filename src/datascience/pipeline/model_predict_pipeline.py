import sys
import os
import pandas as pd
import pickle
from src.datascience import logger

class PredictPipeline:
    def __init__(self):
        # These paths must exist in your project folder
        self.model_path = os.path.join("artifacts", "model_trainer", "model.pkl")
        self.lookup_path = os.path.join("artifacts", "data_transformation", "city_coords_lookup.csv")
        self.zip_counts_path = os.path.join("artifacts", "data_preprocessing", "zip_counts.pkl")

    def predict(self, features):
        try:
            # 1. Load your trained model and lookup tables
            with open(self.model_path, 'rb') as f:
                model = pickle.load(f)
            with open(self.zip_counts_path, 'rb') as f:
                zip_counts = pickle.load(f)
            lookup_df = pd.read_csv(self.lookup_path)

            # 2. Convert Raw Inputs to Model Features (Automated for the client)
            # Map Zipcode to Frequency
            features["zipcode_freq"] = features["zipcode"].map(zip_counts).fillna(0)
            
            # Map City to Lat/Lng
            features = features.merge(lookup_df, on="city_full", how="left")
            features['lat'] = features['lat'].fillna(0)
            features['lng'] = features['lng'].fillna(0)

            # 3. Strict Column Order (Matching your training Index exactly)
            expected_order = [
                'year', 'quarter', 'month', 'median_list_price', 'median_ppsf', 
                'median_list_ppsf', 'homes_sold', 'pending_sales', 'new_listings', 
                'inventory', 'median_dom', 'avg_sale_to_list', 'sold_above_list', 
                'off_market_in_two_weeks', 'bank', 'bus', 'hospital', 'mall', 'park', 
                'restaurant', 'school', 'station', 'supermarket', 'Total Population', 
                'Median Age', 'Per Capita Income', 'Total Families Below Poverty', 
                'Total Housing Units', 'Median Rent', 'Median Home Value', 
                'Total Labor Force', 'Unemployed Population', 
                'Total School Age Population', 'Total School Enrollment', 
                'Median Commute Time', 'lat', 'lng', 'zipcode_freq'
            ]

            # Filter out the raw strings (city_full, zipcode) and keep only model features
            final_features = features[expected_order]
            
            return model.predict(final_features)

        except Exception as e:
            logger.error(f"Prediction Error: {e}")
            raise e

class CustomData:
    def __init__(self, **kwargs):
        # This dynamically assigns all 36+ fields sent from app.py to the class
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_data_as_data_frame(self):
        try:
            # Convert the object attributes back into a dictionary for Pandas
            data_dict = self.__dict__
            # Wrap values in lists for DataFrame creation
            return pd.DataFrame({k: [v] for k, v in data_dict.items()})
        except Exception as e:
            raise e