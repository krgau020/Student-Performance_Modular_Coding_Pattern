import os
import sys       #for using our custom exception

#from src.exception import CustomException
#from src.logger import logging     ---> not working to import logging and CustomException

import unittest
sys.path.insert(0, 'G:\gaurav education\MLgeneralizedpattern\src')

from exception import *
from logger import *   # we need logging info and exception info also. so importing both


import pandas as pd             #we need data frame 
from sklearn.model_selection import train_test_split
from dataclasses import dataclass  #we can create class variable using this




#where i have to save the training data , where test data will be saved -- these INPUT will be in Data_Ingestion class
@dataclass  #decorator   --> we can define our class variable directly . (dont need INIT-->we use __init__ to define class variable inside that class.)
class DataIngestionConfig:
    train_data_path : str= os.path.join('artifacts',"train.csv")
    test_data_path : str= os.path.join('artifacts',"test.csv")    #data_ingestion output will be save in these path
    raw_data_path : str= os.path.join('artifacts',"data.csv")     #these input are given to data_ingestion and now data ingestion know where to store all these


# in CLASS, if we have both function and class variable, use __init__  ----. not @dataclass
    
class Data_Ingestion:
    def __init__(self):               # when we call Data_Ingestion class, above three path will be saved in ingestion_config class variable.
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):  # here write the code to read the read the data from database.
        logging.info("Entered the data Ingestion method or component")
        try:
            df = pd.read_csv('notebook\Data\stud.csv')
            logging.info('Read the dataset in DataFrame')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False ,header= True)

            logging.info("Train Test Split Initiated")

            train_set, test_set = train_test_split(df, test_size=0.2 , random_state=0)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False ,header= True)

            test_set.to_csv(self.ingestion_config.test_data_path, index=False ,header= True)

            logging.info("Ingestion of data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )


        except Exception as e:
            raise CustomException(e, sys)
        

if __name__ == "__main__":
    obj = Data_Ingestion()
    obj.initiate_data_ingestion()