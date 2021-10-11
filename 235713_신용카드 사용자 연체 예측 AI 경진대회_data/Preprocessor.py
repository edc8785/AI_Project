import numpy as np
import pandas as pd


from sklearn.preprocessing import OneHotEncoder
from Config import Config



class Preprocessor(object):
    def __init__(self):    
        self.train = pd.read_csv(Config.data_path + Config.train_data, engine='python')
        self.test = pd.read_csv(Config.data_path + Config.test_data, engine='python')
        
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

        ## Income Log scale
        data["income_total"] = np.log1p(data["income_total"])


        ## Unique ID Variable 생성
        data['ID'] = \
            data['child_num'].astype(str) + '_' + data['income_total'].astype(str) + '_' +\
            data['DAYS_BIRTH'].astype(str) + '_' + data['DAYS_EMPLOYED'].astype(str) + '_' +\
            data['work_phone'].astype(str) + '_' + data['phone'].astype(str) + '_' +\
            data['email'].astype(str) + '_' + data['family_size'].astype(str) + '_' +\
            data['gender'].astype(str) + '_' + data['car'].astype(str) + '_' +\
            data['reality'].astype(str) + '_' + data['income_type'].astype(str) + '_' +\
            data['edu_type'].astype(str) + '_' + data['family_type'].astype(str) + '_' +\
            data['house_type'].astype(str) + '_' + data['occyp_type'].astype(str)

        data.reset_index(drop=True, inplace=True)
        
        return(data)


    def one_hot_encoder_fit(self):
        train = self.train.copy()
        test = self.test.copy()
        
        data = pd.concat([self.preprocess(train).drop('credit', axis=1), self.preprocess(test)])

        ## One hot Encoding
        object_col = []
        for col in data.columns:
            if(data[col].dtype == "object"):
               object_col.append(col)

        one_hot = OneHotEncoder()
        one_hot.fit(data.loc[:,object_col])

        return (one_hot, object_col)


    def one_hot_encoder_transform(self, data, one_hot, object_col):
        onehot_df = pd.DataFrame(one_hot.transform(data.loc[:,object_col]).toarray(), 
             columns=one_hot.get_feature_names(object_col))

        data.drop(object_col, axis=1, inplace=True)
        data = pd.concat([data, onehot_df], axis=1)
        return(data)


    

    def get_train_test_dataset(self):
        one_hot, object_col = self.one_hot_encoder_fit()
        
        train = self.preprocess(self.train)
        train = self.one_hot_encoder_transform(train, one_hot, object_col)

        test = self.preprocess(self.test)
        test = self.one_hot_encoder_transform(test, one_hot, object_col)
        
        return train, test


    def train_val_split(self):
        self.get_train_dataset()


        
        
