#using web app, for predition we give the input data; that data should interact with model (pickle file like preprocessor.pkl
# model.pkl) . all of these will be done here for prediction.

import sys
import pandas as pd

from src.exception import CustomException

from src.utils import load_object          # to load the pickle object


class PredictPileline:
    def __init__(self):
        pass


    def predict(self , features):
        try:
            model_path = 'artifacts\model.pkl'
            preprocessor_path = 'artifacts\proprocessor.pkl'            

            model = load_object(file_path = model_path)                    # our model , it wil predict 
            preprocessor = load_object(file_path = preprocessor_path)      #responsible for handling cat_features , scaling etc

            data_scaled = preprocessor.transform(features)      # for prediction, data must be scaled like all cat_feature in vector form(one_hot)
            preds = model.predict(data_scaled)

            return preds
        
        except Exception as e:
            raise(e, sys)








class CustomData:               #Responsible for mapping all the input that are given in HTML to the backEnd .
    def __init__(self,
                 gender : str ,
                 race_ethnicity : str ,
                 parental_level_of_education : str,
                 lunch : str ,
                 test_preparation_course : str,
                 reading_score : int,
                 writing_score : int
                 ):
        
        self.gender = gender                    # assigning the value. , these value will come from web app

        self.race_ethnicity = race_ethnicity

        self.parental_level_of_education = parental_level_of_education

        self.lunch = lunch

        self.test_preparation_course = test_preparation_course

        self.reading_score = reading_score

        self.writing_score = writing_score


    def get_data_as_data_frame(self):       # we train oue model in form of data_frame, so we need data as DataFrame
        try:

            custom_data_input_dict = {
                "gender" : [self.gender] ,
                "race_ethnicity" : [self.race_ethnicity] ,
                "parental_level_of_education" : [self.parental_level_of_education],
                "lunch" : [self.lunch] ,
                "test_preparation_course" : [self.test_preparation_course],
                "reading_score" : [self.reading_score],
                "writing_score" : [self.writing_score]

            }

            return pd.DataFrame(custom_data_input_dict)


        except Exception as e:
            raise CustomException(e, sys)
