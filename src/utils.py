import os
import sys

import numpy as np
import pandas as pd
import dill                    #dill help us to create pickle file

from src.exception import CustomException


def save_object(file_path , obj):              #saving all object using this function
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok= True)

        with open (file_path , "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
