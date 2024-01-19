from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData , PredictPileline

application = Flask(__name__)           #it will give the entry point

app = application


## Route for home page

@app.route('/')
def index():                                    # when we use "render index.html" , it will look for "templates"  file. 
    return render_template('index.html')        # in templates file, we will have index.html , home.html


@app.route('/prediction', methods=['GET', 'POST'])  # GET --  Read data      POST- insert data  , PUT -- Update data , DELETE

def predict_datapoint():            # predict_datapoint is function in home.html in POST form, from there data will be stored here
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(                                   # For POST req, we wil create our data here. In "CustomData" class
                gender= request.form.get('gender'),                                                      # "CustomData" Class will be actually created in Predict_pileline file in PIPELINE folder
                race_ethnicity=request.form.get('ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),    #from web page, input data came here. then it will go to Prediction_pipeline
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=float(request.form.get('writing_score')),
                writing_score=float(request.form.get('reading_score'))
        
        )                  

        pred_df = data.get_data_as_data_frame()         #after getting data, we need data as dataframe to predict
        print(pred_df)


        predict_pipeline = PredictPileline()        # after getting data as data frame, object of PredictPipeline class is created,
        results = predict_pipeline.predict(pred_df)      # so we can call "predict" function that is created over there. 
                                                        #it will return the predicted value
        return render_template('home.html' , results = results[0])




if __name__ == "__main__":
    app.run(host= "0.0.0.0", debug= True)





                                           