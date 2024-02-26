import os
import sys
from dataclasses import dataclass
from sklearn.ensemble import (AdaBoostRegressor,
                              GradientBoostingRegressor,
                           RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from exception import CustomException
from logger import logging
from utils import evaluate_module

from utils import save_objects

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","models.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            preprocessor_path = self.model_trainer_config.trained_model_file_path
            logging.info("Split training and testing data")
            x_train,y_train,x_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )    
            models = {
                        "Linear Regression": LinearRegression(),
                        "GradientBoostingRegressor": GradientBoostingRegressor(),
                        "K-Neighbors Regressor": KNeighborsRegressor(),
                        "Decision Tree": DecisionTreeRegressor(),
                        "Random Forest Regressor": RandomForestRegressor(),
                        "XGBRegressor": XGBRegressor(),
                        "AdaBoost Regressor": AdaBoostRegressor()
            }   
            model_report = evaluate_module(X_train=x_train, Y_train=y_train, X_test=x_test, Y_test=y_test, models=models)

            
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
            if best_model_score < 0.6:
                raise CustomException("NO best model found")
            
            logging.info(f"best model in both training and testing found:{best_model_name}")
            
            save_objects(
                fil_path= preprocessor_path,
                obj=best_model
            )
            
            predicted = best_model.predict(x_test)
            r2_square = r2_score(y_test, predicted)
            print(f"r2_square: {r2_square}")
            
        except Exception as e:
            raise CustomException(e,sys)
    