## Set Directory

current_working_dir = dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(current_working_dir)

source("preprocess.R")

## Import Library
library(caret)
library(Metrics)
library(xgboost)
library(mltools)



## Transform factor variable to one hot encoding
# one_hot function is available to data.table. So transform data.frame to data.table
one_hot_train = one_hot(as.data.table(train))
one_hot_test = one_hot(as.data.table(test))


## Hyper parameter tuning - cross validation
train_data = one_hot_train[,2:(ncol(one_hot_train)-1)]
x_train_cv <- data.matrix(train_data)
y_train_cv <- data.matrix(one_hot_train$credit)

xgb_cv <- xgb.cv(data=x_train_cv, label=y_train_cv, nfold=5, nrounds=1000, early_stopping_rounds = 10,
                 objective='multi:softprob', metrics="mlogloss", num_class=3, prediction=T, print_every_n = 10,
                 params=list(eta=0.05, max_depth=8, subsample=0.8,colsample_bytree=0.8,stratified=T)) 



## Training
train_x<-data.matrix(train_data)
train_y<-data.matrix(one_hot_train$credit)
xgb_train<-xgb.DMatrix(data=train_x,label=train_y)
watchlist <- list(train=xgb_train)
xgb_fit <- xgb.train(data = xgb_train, 
                     eta=0.05, 
                     max_depth=8, subsample=0.8,colsample_bytree=0.8,
                     nrounds= 545,  # xgb_cv °á°ú, Best iteration
                     objective= "multi:softprob",  
                     eval_metric= "mlogloss", num_class=3,             
                     watchlist=watchlist,
                     print_every_n = 10
)




## Prediction
xgb_test<-data.matrix(one_hot_test[,-1])
xgb_pred_test<-predict(xgb_fit,xgb_test)
xgb_pred_test_total<-matrix(xgb_pred_test,nrow=3) %>% t() %>% data.frame()
xgb_pred_test_total

sub[,2:4]<-xgb_pred_test_total
write.csv(sub,file="DATA/my_submission.csv",row.names=F)
