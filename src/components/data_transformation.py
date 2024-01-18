# Here we willl do the Data Cleaning , FEATURE ENGINEERING , CONVERTING CATEGORICAL FEATURE TO NUMERICAL, DATA standarizzation etc.


import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer       # allow to apply diffrent tranformer to diffrnt column of data. like for categorical data 
                                                #one hot encoding and for numerical data STANDARD SCALER then combine it with pipelines. 


from sklearn.impute import SimpleImputer         # to check missing value
from sklearn.pipeline import Pipeline               # for using pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler       # OneHotEncode -> converting categorical feature to numerical
                                                                    # StandardScaler -> scaling of numerical feature ; mean = 0 , variance = 1


from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object



@dataclass
class DataTransformationConfig:                 # input (or variable) that requires for DAtA TRANSFORMATiON
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")     #if we create any model and we want to save that into a pickle file. for that this path will be used.
                                                                                # we can create model file also and save the pickle file there
                                                                                

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()


    def get_data_transformer_object(self):      #Responsible for DATA TRANSFORMATION.  this is for creating all the pickle file which is doing transformation work like StandardScaler, Onehot etc
        
        try:
            numerical_columns = ["writing_score", "reading_score"]      #we have done this in notebook/EDA
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline= Pipeline(                    #after doing the operations given in steps, pipeline will combine the data
                steps=[
                ("imputer",SimpleImputer(strategy="median")),    # imputer will be used to check if there is any missing value then fill it with MEDIAN
                ("scaler",StandardScaler())                        # StandardScaler(with_mean = False) --> scaling only on std dev, mean will not be used

                ]
            )

            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(                         # combine both numerical and categorical pipeline
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)

                ]


            )

            return preprocessor
        
        
        except Exception as e:
            raise CustomException(e,sys)
        


        
    def initiate_data_transformation(self,train_path,test_path):     # train_path and test_path will be taken from Data_ingestion part

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")


            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)   
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)       #fit_transform for training input data --> tranform for StandardScaler and oneHot
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)                # fit_transform is used once for training, so "transform" for test data input


            train_arr = np.c_[                                                           # concatinate target and input feature as training data
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]       ### concatinate target and input feature as test data


            logging.info(f"Saved preprocessing object.")




            save_object(                                                     # preproccesing_obj must be saved as pickle file as all the transformation will occurs through this model.

                file_path=self.data_transformation_config.preprocessor_obj_file_path,         #save_object method is in utils
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        
        
        except Exception as e:
            raise CustomException(e,sys)