import pickle
from  Config import Config
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold

class Trainer(object):
    def run(self, train):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        folds=[]

        for train_idx, valid_idx in skf.split(train, train['credit']):
            folds.append((train_idx, valid_idx))


        lgb_models = []


        for fold in range(5):
            print(f'===================================={fold+1}============================================')
            train_idx, valid_idx = folds[fold]
            X_train, X_valid, y_train, y_valid = train.drop(['credit'],axis=1).iloc[train_idx].values, train.drop(['credit'],axis=1).iloc[valid_idx].values,\
                                                 train['credit'][train_idx].values, train['credit'][valid_idx].values 


            lgb = LGBMClassifier(n_estimators=1000)
            lgb.fit(X_train, y_train, 
                    eval_set=[(X_train, y_train), (X_valid, y_valid)], 
                    early_stopping_rounds=30,
                   verbose=100)
            lgb_models.append(lgb)
            print(f'================================================================================\n\n')


        with open(Config.model_path  + "LightGBM.pickle", "wb") as f:
            pickle.dump(lgb_models, f)

        print("Complete Training LightGBM")

        
