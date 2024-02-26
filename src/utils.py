import os
import sys
# Get the parent directory of the current file (src)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# Add the parent directory to the Python path
sys.path.append(parent_dir)
import pandas as pd
import pandas as pd
import dill
from exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_objects(fil_path,obj):
    try:
        dir_path = os.path.dirname(fil_path)
        os.makedirs(dir_path,exist_ok=True)
        
        with open(fil_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise   CustomException(e,sys)
    
def evaluate_module(X_train, Y_train, X_test, Y_test, models,param) -> dict:
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,Y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,Y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            train_model_score = r2_score(Y_train, y_train_pred)
            test_model_score = r2_score(Y_test, y_test_pred)
   
            report[list(models.keys())[i]] = test_model_score
            
        return report  
    except Exception as e:  # Handle specific exceptions
        raise CustomException(e, sys)
    

