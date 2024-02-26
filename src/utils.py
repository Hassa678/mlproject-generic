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

def save_objects(fil_path,obj):
    try:
        dir_path = os.path.dirname(fil_path)
        os.makedirs(dir_path,exist_ok=True)
        
        with open(fil_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise   CustomException(e,sys)