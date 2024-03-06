import os
import sys
from src.Credit_Defaultor_Prediction.exception import CustomException
from src.Credit_Defaultor_Prediction.logger import logging
import pandas as pd
    
def read_data():
    logging.info("Reading data from Notebook data ")
    try:
        train = pd.read_feather("notebook/data/test.feather")
        test = pd.read_feather("notebook/data/train.feather")
        labels = pd.read_csv("notebook/data/labels.csv")
                             
        return train,test,labels
    
    except Exception as ex:
        raise CustomException(ex)