import pickle
import pandas as pd
from Config import Config

class Predictor(object):
    
    def model_load_LightGBM(self):
        with open(Config.model_path + "LightGBM.pickle","rb") as f:
            model = pickle.load(f)
        return(model)

    def model_load_catboost(self):
        with open(Config.model_path + "catboost.pickle","rb") as f:
            model = pickle.load(f)
        return(model)

    def run(self, test, model):
        sub = pd.read_csv(Config.data_path + Config.sub_data, engine="python")
        lmodel = model
        
        sub.iloc[:,1:] = 0
        for fold in range(5):
            sub.iloc[:,1:] +=  model[fold].predict_proba(test)/5


        sub.to_csv(Config.data_path  + Config.my_sub_data, index=False)
        print("Complete Prediction")
                
