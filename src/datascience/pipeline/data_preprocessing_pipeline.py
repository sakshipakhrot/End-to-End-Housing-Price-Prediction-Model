from src.datascience.config.configuration import ConfigurationManager
from src.datascience.components.data_preprocessing import DataPreProcessing
from src.datascience import logger


STAGE_NAME="Data PreProcessing"
class DataPreProcessingTrainingPipeline:
    def __init__(self):
        pass

    def initiate_data_preprocessing(self):
        config = ConfigurationManager()
        data_preprocessing_config = config.get_data_preprocessing_config()
        data_preprocessing = DataPreProcessing(config=data_preprocessing_config)
        data_preprocessing.data_preprocessing()

if __name__ == '__main__':
    try:
        logger.info("------------------> stage {STAGE_NAME} started <------------------")
        obj = DataPreProcessingTrainingPipeline()
        obj.initiate_data_preprocessing()
        logger.info("------------------> stage {STAGE_NAME} completed <------------------")
    except Exception as e:
        logger.exception(e)
        raise e
    