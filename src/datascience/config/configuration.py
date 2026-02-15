from src.datascience.constants import *
from src.datascience.utils.common import read_yaml, create_directories

from src.datascience.entity.config_entity import (DataIngestionConfig, DataTransformationConfig, DataPreProcessingConfig, ModelTrainerConfig)




class ConfigurationManager:
    def __init__(self,
                 config_filepath=CONFIG_FILE_PATH,
                 params_filepath=PARAMS_FILE_PATH,
                 schema_filepath=SCHEMA_FILE_PATH):
        self.config=read_yaml(config_filepath)
        self.params=read_yaml(params_filepath)
        self.schema=read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])


    def get_data_ingestion_config(self)-> DataIngestionConfig:
        config=self.config.data_ingestion
        create_directories([config.root_dir])

        data_ingestion_config=DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )
        return data_ingestion_config
    
        

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config=self.config.data_transformation
        create_directories([config.root_dir])
        data_transformation_config=DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            metro_path=config.metro_path
        )
        return data_transformation_config
    

    def get_data_preprocessing_config(self) -> DataPreProcessingConfig:
        config=self.config.data_preprocessing
        create_directories([config.root_dir])
        data_preprocessing_config=DataPreProcessingConfig(
            root_dir=config.root_dir,
            train_path=config.train_path,
            eval_path=config.eval_path
        )
        return data_preprocessing_config
    

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            train_data_path = config.train_data_path,
            eval_data_path = config.eval_data_path
            
        )

        return model_trainer_config