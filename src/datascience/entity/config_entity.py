from dataclasses import dataclass
from pathlib import Path

@dataclass()
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path



@dataclass
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    metro_path: Path


@dataclass
class DataPreProcessingConfig:
    root_dir: Path
    train_path: Path
    eval_path: Path


@dataclass
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    eval_data_path: Path
