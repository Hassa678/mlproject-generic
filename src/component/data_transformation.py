import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

sys.path.append(parent_dir)

from exception import CustomException
from logger import logging
from utils import save_objects

@dataclass
class DataTransforConfig:
    processor_obj_file_path = os.path.join('artifacts','proprecessor.pkl')
    
    
class DataTransformation:
    def __init__(self):
        self.data_transfor_config = DataTransforConfig()
    def get_data_transformation(self):
        "this function is responsible for data transformation."
        try:
            numerical_columnes = ['reading_score','writing_score']
            categorical_columnes = [
              'gender','race_ethnicity','parental_level_of_education',
              'lunch','test_preparation_course'  
            ]
            num_pipline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )
            
            cat_pipline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ]
            )
            logging.info(f'categorical columnes encoding completed:{categorical_columnes}')
            logging.info(f'numerical columnes standerscaler completed:{numerical_columnes}')
            
            preprocessor=ColumnTransformer([
                ('num_pipeline',num_pipline,numerical_columnes),
                ('cate_pipeline',cat_pipline,categorical_columnes)
                
                ]
                
            )
            
            return preprocessor
            
            
        except Exception as e:
            raise CustomException(e,sys)
    def initite_data_transformition(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)  
            
            logging.info("read train and test data")
            logging.info("Obtaining preprocessing objects")
            preprocessing_obj = self.get_data_transformation()
            
            target_column_name='math_score'
            input_feature_column_train = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_column_train = train_df[target_column_name]
            input_feature_column_test = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_column_test = test_df[target_column_name]
            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe.")
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_column_train)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_column_test)
            train_arr = np.c_[ input_feature_train_arr, np.array(target_feature_column_train)]
            test_arr = np.c_[ input_feature_test_arr, np.array(target_feature_column_test)]

            logging.info(f"Saved preprocessing object.")
            save_objects(

                fil_path=self.data_transfor_config.processor_obj_file_path,
                obj=preprocessing_obj
                )

            return (
                train_arr,
                test_arr,
                self.data_transfor_config.processor_obj_file_path,
                )
        except Exception as e:
            raise CustomException(e,sys)      