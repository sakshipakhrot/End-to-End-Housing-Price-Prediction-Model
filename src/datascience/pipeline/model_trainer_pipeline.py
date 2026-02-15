from src.datascience.config.configuration import ConfigurationManager
from src.datascience.components.model_trainer import ModelTrainer
from src.datascience import logger


STAGE_NAME = "Model Trainer"

class ModelTrainerPipeline:
    def __init__(self):
        pass

    def initiate_model_training(self):
        # Initialize the config manager
        config = ConfigurationManager()
        
        # Get the specific configuration for model training
        model_trainer_config = config.get_model_trainer_config()
        
        # Initialize the component with the config
        model_trainer = ModelTrainer(config=model_trainer_config)
        
        # Start the training process
        model_trainer.initiate_model_trainer()


if __name__ == '__main__':
    try:
        # Added 'f' before the string to enable variable injection
        logger.info(f"------------------> stage {STAGE_NAME} started <------------------")
        
        obj = ModelTrainerPipeline()
        obj.initiate_model_training()
        
        logger.info(f"------------------> stage {STAGE_NAME} completed <------------------\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
    