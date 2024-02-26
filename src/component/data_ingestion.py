import sys
import os
# Get the parent directory of the current file (src)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# Add the parent directory to the Python path
sys.path.append(parent_dir)

from exception import CustomException
from logger import logging
import pandas as pd


from component.data_transformation import DataTransformation
from component.data_transformation import DataTransforConfig
from component.model_trainer import ModelTrainer
from component.model_trainer import ModelTrainerConfig
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join('artifacts','train.csv')
    test_data_path = os.path.join('artifacts','test.csv')
    raw_data_path = os.path.join('artifacts','raw.csv')
    
class DataIngestion:
    def __init__(self):
        self.ingestio_config = DataIngestionConfig()
    def initiat_data_ingestion(self):
        logging.info('entering data ingestion methode')
        try:
            df=pd.read_csv(r'notebook\data\stud.csv')
            logging.info('Read the dataset as dataframe')
            os.makedirs(os.path.dirname(self.ingestio_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestio_config.raw_data_path,index=False,header=True)
            logging.info('set train_test_split')
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestio_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestio_config.test_data_path,index=False,header=True)
            logging.info('Ingestion of the data is completed successfully')
            return(
                self.ingestio_config.train_data_path,
                self.ingestio_config.test_data_path
                )
        except Exception as e:
            logging.error('Ingestion of the data failed')
            raise CustomException('Ingestion of the data failed',sys)

if __name__=='__main__':
    data_ingestion = DataIngestion()
    train_data,test_data =data_ingestion.initiat_data_ingestion()
    data_transformation = DataTransformation()
    train_array,test_array,_ = data_transformation.initite_data_transformition(train_data, test_data)
    logging.info('data transformation is completed successfully')
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_trainer(train_array, test_array)
    
        