import os
from src.datascience import logger
from src.datascience.entity.config_entity import ModelTrainerConfig
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, 
    GradientBoostingRegressor, 
    AdaBoostRegressor
)
from xgboost import XGBRegressor
from src.datascience.utils.common import evaluate_models, save_object
from sklearn.metrics import r2_score



class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def initiate_model_trainer(self):
        try:
            # 1. LOAD DATA
            train_df = pd.read_parquet(self.config.train_data_path)
            eval_df = pd.read_parquet(self.config.eval_data_path)

            X_train = train_df.drop('price', axis=1)
            y_train = train_df['price'] 

            X_eval = eval_df.drop('price', axis=1)
            y_eval = eval_df['price']
            
            # Use default settings for quick execution
            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "XGBRegressor": XGBRegressor()
            }
            
            # Empty parameter dictionaries bypasses GridSearchCV tuning
            params = {
                "Decision Tree": {},
                "Linear Regression": {},
                "XGBRegressor": {}
            }

            # 2. EVALUATE MODELS (Now much faster as it uses defaults)
            model_report: dict = evaluate_models(
                X_train=X_train, y_train=y_train, X_test=X_eval, y_test=y_eval,
                models=models
            )
            
            # 3. SELECT BEST MODEL
            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            # Re-fit the best model on the training data 
            # (Ensures the model object is fully trained before saving)
            best_model = models[best_model_name]
            best_model.fit(X_train, y_train)

            if best_model_score < 0.6:
                logger.warning(f"Best model {best_model_name} has low R2 score: {best_model_score}")
            
            logger.info(f"Best model found: {best_model_name} with score: {best_model_score}")

            # 4. SAVE MODEL
            model_save_path = os.path.join(self.config.root_dir, "model.pkl")
            
            save_object(
                file_path=model_save_path,
                obj=best_model
            )
            
            # 5. FINAL SCORE VERIFICATION
            predicted = best_model.predict(X_eval)
            r2_square = r2_score(y_eval, predicted)
            
            return r2_square
            
        except Exception as e:
            logger.error(f"Error in model_trainer: {e}")
            raise e