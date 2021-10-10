import numpy as np
import pandas as pd


from sklearn.preprocessing import OneHotEncoder
from Config import Config



class Preprocessor(object):
    def __init__(self):    
        self.train = pd.read_csv(Config.data_path + Config.train_data)
        self.test = pd.read_csv(Config.data_path + Config.test_data)


    def preprocess(self, data):

        ## Dataset preprocess
        data["DAYS_BIRTH"] = (-data["DAYS_BIRTH"]/365).astype(int)
        data["DAYS_EMPLOYED"] = (-data["DAYS_EMPLOYED"]/365).astype(int)
        data["begin_month"] = -data["begin_month"]

        data.drop(["FLAG_MOBIL"], axis=1, inplace=True)
        data = data[data["family_size"]<=7]
        data["DAYS_EMPLOYED"][data["DAYS_EMPLOYED"] <=0] = 0

        ## Na Imputation
        data["occyp_type"] = data["occyp_type"].fillna("Not Check")


        ## One hot Encoding
        data.reset_index(drop=True, inplace=True)
        object_col = []
        for col in data.columns:
            if(data[col].dtype == "object"):
               object_col.append(col)


        one_hot = OneHotEncoder()
        one_hot.fit(data.loc[:,object_col])

        onehot_df = pd.DataFrame(one_hot.transform(data.loc[:,object_col]).toarray(), 
             columns=one_hot.get_feature_names(object_col))

        data.drop(object_col, axis=1, inplace=True)
        data = pd.concat([data, onehot_df], axis=1)


        return(data)

    
    def get_train_dataset(self):
        return self.preprocess(self.train)

    def get_test_dataset(self):
        return self.preprocess(self.test)


    def train_val_split(self):
        self.get_train_dataset()


        
        
