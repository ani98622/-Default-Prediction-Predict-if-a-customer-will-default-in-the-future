from src.Credit_Defaultor_Prediction.components.model_trainer import ModelTrainers
from src.Credit_Defaultor_Prediction.logger import logging
from src.Credit_Defaultor_Prediction.exception import CustomException
from src.Credit_Defaultor_Prediction.components.data_ingestion import DataIngestion
from src.Credit_Defaultor_Prediction.components.data_transformation import DataTransformation
import sys


if __name__=="__main__":
    logging.info("The execution has started")

    try:
        data_ingestion = DataIngestion()
        labels,train,test = data_ingestion.initiate_data_ingestion()
       
        data_transformation = DataTransformation()
        (categorical,train_cols,X_train, y_train, X_cv, y_cv , X_test, file_path) = data_transformation.initiate_data_transformation()

        model_trainer = ModelTrainers() 
        print(model_trainer.initiate_model_trainer(train_cols,categorical,train,cv_data,test,X_test,X_train,y_train,X_cv,y_cv))
        
    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)
