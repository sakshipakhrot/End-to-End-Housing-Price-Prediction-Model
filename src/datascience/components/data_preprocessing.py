import os
from src.datascience import logger
from src.datascience.entity.config_entity import DataPreProcessingConfig
from category_encoders import TargetEncoder
import pandas as pd
import pickle



class DataPreProcessing:
    def __init__(self, config: DataPreProcessingConfig):
        self.config = config

    def _add_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Internal helper for date feature logic."""
        df["date"] = pd.to_datetime(df["date"])
        df["year"] = df["date"].dt.year
        df["quarter"] = df["date"].dt.quarter
        df["month"] = df["date"].dt.month
        
        cols = ["year", "quarter", "month"]
        for i, col in enumerate(cols, 1):
            df.insert(i, col, df.pop(col))
        return df

    def data_preprocessing(self):
        try:
            # 1. LOAD DATA
            train_df = pd.read_parquet(self.config.train_path)
            eval_df = pd.read_parquet(self.config.eval_path)

            # 2. FREQUENCY ENCODING (Zipcode) - Calculate from Train
            logger.info("Calculating Zipcode frequencies from train set...")
            zip_counts = train_df["zipcode"].value_counts()

            # 4. DEFINE DATASETS FOR LOOPED PROCESSING
            dataset_map = {
                "train": train_df,
                "eval": eval_df
            }

            # Columns to drop to avoid leakage and redundant data
            drop_cols = ["date", "city_full", "city", "zipcode", "median_sale_price"]

            # Calculate frequency map
            zip_counts = train_df["zipcode"].value_counts().to_dict()
        
            # Define path based on your config
            save_path = os.path.join(self.config.root_dir, "zip_counts.pkl")
        
            # Save as Pickle (Write Binary mode)
            with open(save_path, 'wb') as f:
                pickle.dump(zip_counts, f)
            
            logger.info(f"Saved zipcode frequency pickle to: {save_path}")

            # 5. LOOP: APPLY REMAINING LOGIC AND SAVE
            for name, df in dataset_map.items():
                logger.info(f"Finalizing preprocessing for {name} dataset...")

                # Apply Date Features
                df = self._add_date_features(df)

                # Apply Frequency Encoding
                df["zipcode_freq"] = df["zipcode"].map(zip_counts).fillna(0)

                # Drop unused and leakage columns
                df.drop(columns=drop_cols, inplace=True, errors='ignore')

                # Save the final cleaned parquet
                save_path = os.path.join(self.config.root_dir, f"{name}_processed.parquet")
                df.to_parquet(save_path, index=False)
                
                logger.info(f"Shape of {name} after dropping columns: {df.shape}")

            logger.info("Data Preprocessing stage completed successfully.")

        except Exception as e:
            logger.error(f"Error in data_preprocessing: {e}")
            raise e