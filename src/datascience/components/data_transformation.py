import os
from src.datascience import logger
from sklearn.model_selection import train_test_split
from src.datascience.entity.config_entity import DataTransformationConfig
import re
import pandas as pd


city_mapping = {
    'Atlanta-Sandy Springs-Alpharetta': 'Atlanta-Sandy Springs-Roswell, GA',
    'Pittsburgh': 'Pittsburgh, PA',
    'Boston-Cambridge-Newton': 'Boston-Cambridge-Newton, MA-NH',
    'Tampa-St. Petersburg-Clearwater': 'Tampa-St. Petersburg-Clearwater, FL',
    'Baltimore-Columbia-Towson': 'Baltimore-Columbia-Towson, MD',
    'Portland-Vancouver-Hillsboro': 'Portland-Vancouver-Hillsboro, OR-WA',
    'Philadelphia-Camden-Wilmington': 'Philadelphia-Camden-Wilmington, PA-NJ-DE-MD',
    'New York-Newark-Jersey City': 'New York-Newark-Jersey City, NY-NJ',
    'Chicago-Naperville-Elgin': 'Chicago-Naperville-Elgin, IL-IN',
    'Orlando-Kissimmee-Sanford': 'Orlando-Kissimmee-Sanford, FL',
    'Seattle-Tacoma-Bellevue': 'Seattle-Tacoma-Bellevue, WA',
    'San Francisco-Oakland-Berkeley': 'San Francisco-Oakland-Fremont, CA',
    'San Diego-Chula Vista-Carlsbad': 'San Diego-Chula Vista-Carlsbad, CA',
    'Austin-Round Rock-Georgetown': 'Austin-Round Rock-San Marcos, TX',
    'St. Louis': 'St. Louis, MO-IL',
    'Sacramento-Roseville-Folsom': 'Sacramento-Roseville-Folsom, CA',
    'Phoenix-Mesa-Chandler': 'Phoenix-Mesa-Chandler, AZ',
    'Riverside-San Bernardino-Ontario': 'Riverside-San Bernardino-Ontario, CA',
    'San Antonio-New Braunfels': 'San Antonio-New Braunfels, TX',
    'Detroit-Warren-Dearborn': 'Detroit-Warren-Dearborn, MI',
    'Cincinnati': 'Cincinnati, OH-KY-IN',
    'Houston-The Woodlands-Sugar Land': 'Houston-Pasadena-The Woodlands, TX',
    'Charlotte-Concord-Gastonia': 'Charlotte-Concord-Gastonia, NC-SC',
    'Denver-Aurora-Lakewood': 'Denver-Aurora-Centennial, CO',
    'Los Angeles-Long Beach-Anaheim': 'Los Angeles-Long Beach-Anaheim, CA',
    'DC_Metro': 'Washington-Arlington-Alexandria, DC-VA-MD-WV',
    'Dallas-Fort Worth-Arlington': 'Dallas-Fort Worth-Arlington, TX',
    'Minneapolis-St. Paul-Bloomington': 'Minneapolis-St. Paul-Bloomington, MN-WI',
    'Las Vegas-Henderson-Paradise': 'Las Vegas-Henderson-North Las Vegas, NV',
    'Miami-Fort Lauderdale-Pompano Beach': 'Miami-Fort Lauderdale-West Palm Beach, FL'
}


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config=config

    def data_transformation(self):
        """
        Complete pipeline in one sequence: 
        Load -> Merge -> Clean -> Filter -> Split -> Save
        """
        try:
            # 1. LOAD DATA
            df = pd.read_parquet(self.config.data_path)
            metros = pd.read_csv(self.config.metro_path)
            logger.info(f"Loaded raw data: {df.shape}")


            # 2. CLEAN & MERGE
            df["city_full"] = df["city_full"].replace(city_mapping)
            df = df.merge(
                metros[["metro_full", "lat", "lng"]],
                how="left", 
                left_on="city_full", 
                right_on="metro_full"
            )
            df.drop(columns=["metro_full"], inplace=True)

            
            
            # 3. REMOVE DUPLICATES (Applied on the merged output)
            comparison_cols = df.columns.difference(['date', 'year']).tolist()
            df = df.drop_duplicates(subset=comparison_cols, keep=False)


            

            lookup_df = df[['city_full', 'lat', 'lng']].drop_duplicates(subset=['city_full'])
        
            # Define path based on your config
            save_path = os.path.join(self.config.root_dir, "city_coords_lookup.csv")
        
            # Save as CSV
            lookup_df.to_csv(save_path, index=False)
            logger.info(f"Saved city coordinates lookup to: {save_path}")



            
            # 4. REMOVE OUTLIERS (Applied on the de-duplicated output)
            df = df[df['median_list_price'] <= 19_000_000].copy()
            logger.info(f"Pre-processing complete. Shape for splitting: {df.shape}")

            # 5. SORT & SPLIT (Applied on the processed output)
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date")

            train_df = df[df["date"] < "2023-01-01"].copy()
            eval_df = df[df["date"] >= "2023-01-01"].copy()

            # 6. SAVE TO ARTIFACTS
            train_df.to_parquet(os.path.join(self.config.root_dir, "train.parquet"), index=False)
            eval_df.to_parquet(os.path.join(self.config.root_dir, "eval.parquet"), index=False)

            logger.info(f"Transformation and Splitting successful!")
            logger.info(f"Final Counts -> Train: {len(train_df)}, Eval: {len(eval_df)}")

        except Exception as e:
            logger.error(f"Error in initiate_data_transformation: {e}")
            raise e