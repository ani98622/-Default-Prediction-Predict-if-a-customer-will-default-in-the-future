from src.Credit_Defaultor_Prediction.logger import logging
from src.Credit_Defaultor_Prediction.exception import CustomException
from src.Credit_Defaultor_Prediction.components.data_ingestion import DataIngestion
from src.Credit_Defaultor_Prediction.components.data_transformation import DataTransformation
import sys


if __name__=="__main__":
    logging.info("The execution has started")

    try:
        data_ingestion = DataIngestion()
        data_ingestion.initiate_data_ingestion
       
        data_transformation = DataTransformation()
        data_transformation.initiate_data_transformation()
        

        model_trainer = ModelTrainer()
        print(model_trainer.initiate_model_trainer(train_arr,test_arr))
        
    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)
