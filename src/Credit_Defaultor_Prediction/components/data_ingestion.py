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
        try:
            # Reading Code  
            train,test,labels = read_data()

            logging.info("Data Ingestion is completed")
            return train,test,labels
        except Exception as e:
            raise CustomException(e,sys)