from src.datascience import logger
from src.datascience.pipeline.data_ingestion_pipeline import DataIngestionTrainingPipeline
from src.datascience.pipeline.data_transformation_pipeline import DataTransformationTrainingPipeline
from src.datascience.pipeline.data_preprocessing_pipeline import DataPreProcessingTrainingPipeline
from src.datascience.pipeline.model_trainer_pipeline import ModelTrainerPipeline



STAGE_NAME = "Data Ingestion Stage"
try:
    logger.info(f"-------------> stage {STAGE_NAME} started <------------------")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.initiate_data_ingestion()
    logger.info(f"-------------> stage {STAGE_NAME} completed <------------------\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Data Transformation Stage"
try:
    logger.info(f"-------------> stage {STAGE_NAME} started <------------------")
    data_transformation = DataTransformationTrainingPipeline()
    data_transformation.initiate_data_transformation()
    logger.info(f"-------------> stage {STAGE_NAME} completed <------------------\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Data Preprocessing Stage"
try:
    logger.info(f"-------------> stage {STAGE_NAME} started <------------------")
    data_preprocessing = DataPreProcessingTrainingPipeline()
    data_preprocessing.initiate_data_preprocessing()
    logger.info(f"-------------> stage {STAGE_NAME} completed <------------------\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Model Trainer Stage"
try:
    logger.info(f"-------------> stage {STAGE_NAME} started <------------------")
    model_trainer = ModelTrainerPipeline()
    model_trainer.initiate_model_trainer()
    logger.info(f"-------------> stage {STAGE_NAME} completed <------------------\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


