from src.datascience import logger
from src.datascience.pipeline.data_ingestion_pipeline import DataIngestionTrainingPipeline
from src.datascience.pipeline.data_transformation_pipeline import DataTransformationTrainingPipeline
from src.datascience.pipeline.data_preprocessing_pipeline import DataPreProcessingTrainingPipeline
from src.datascience.pipeline.model_trainer_pipeline import ModelTrainerPipeline

STAGE_NAME="Data Ingestion Stage"
try:
    logger.info(f"-------------> stage {STAGE_NAME} started <------------------")
    data_ingestion=DataIngestionTrainingPipeline()
    data_ingestion.initiate_data_ingestion()
    logger.info(f"-------------> stage {STAGE_NAME} completed <------------------")
except Exception as e:
    logger.exception(e)
    raise e


