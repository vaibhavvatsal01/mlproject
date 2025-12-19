import os
import sys
import pandas as pd
from src.excepton import CustomException
from src.utils import load_object
class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("before loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("afterloading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)
class Customdata:
    def __init__(self,
         gender:str,
         race_ethnicity:str,
         parental_level_of_education:str,
         test_preparation_course:str,
         lunch:str,
         reading_score:int,
         wrinting_score:int
    ) :
        self.gender=gender
        self.race_ethnicity=race_ethnicity
        self.parental_level_of_education=parental_level_of_education
        self.test_preparation_course=test_preparation_course
        self.lunch=lunch
        self.reading_score=reading_score
        self.wrinting_score=wrinting_score
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict={
                "gender":[self.gender],
                "race_ethnicity":[self.race_ethnicity],
                "parental_level_of_education":[self.parental_level_of_education],
                "test_preparation_course":[self.test_preparation_course],   
                "lunch":[self.lunch],
                "reading_score":[self.reading_score],
                "wrinting_score":[self.wrinting_score],  
            }    
            return pd.DataFrame(custom_data_input_dict)
        except exception as e:
            raise CustomException(e,sys)
        



