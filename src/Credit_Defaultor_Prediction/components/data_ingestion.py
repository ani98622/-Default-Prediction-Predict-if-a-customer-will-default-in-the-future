import os 
import sys 
from src.Credit_Defaultor_Prediction.exception import CustomException
from src.Credit_Defaultor_Prediction.logger import logging
import  pandas as pd
from src.Credit_Defaultor_Prediction.utils import read_data
from dataclasses import dataclass

@dataclass
class DataIngestionConfig :
    train_data_path : str = os.path.join('artifacts','train.feather')
    test_data_path : str = os.path.join('artifacts','test.feather')
    labels_data_path : str = os.path.join('artifacts','labels.csv')

class DataIngestion:
    def __init__(self) :
        self.ingestion_config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        try :
            # Reading Code  
            train,test,labels = read_data()
            logging.info("Data Ingestion is completed")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            
            labels.to_csv(self.ingestion_config.labels_data_path,index=False,header=True)
            train.to_feather(self.ingestion_config.train_data_path,index=False,header=True)
            test.to_feather(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Data Ingestion is completed")

            return labels,train,test
        
        except Exception as e : 
            raise CustomException(e,sys)