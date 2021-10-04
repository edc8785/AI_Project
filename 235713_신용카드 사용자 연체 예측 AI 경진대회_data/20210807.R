setwd("D:/Project/235713_신용카드 사용자 연체 예측 AI 경진대회_data/open")


library(data.table)
library(dplyr)
library(tidyverse)
library(randomForest)
library(caret)
library(mltools)
library(reshape2)

train = read.table("train.csv", sep=",", header=T)
test = read.table("test.csv", sep=",", header=T)

sub = read.table("sample_submission.csv", sep=",", header=T)

## 데이터 변수 설명


# - credit : 사용자의 신용카드 대금 연체를 기준으로 한 신용도 -> 낮을 수록 높은 신용의 신용카드 사용자를 의미함

# - index
# - gender : 성별
# - car : 차량 소유 여부
# - reality : 부동산 소유 여부
# - child_num : 자녀 수
# - income_total : 연간 소득
# - incomre_type : 소득 분류
# - edu_type : 교육 수준
# - family_type : 결혼 여부
# - house_type : 생활 방식
# - DAYS_BIRTH : 출생일
# - DAYS_EMPLOYED : 업무 시작일
# - FLAG_MODIL : 핸드폰 소유 여부
# - work_phone : 업무용 전화 소유 여부
# - phone : 전화 소유 여부
# - email : 이메일 소유 여부
# - occyp_type : 직업 유형
# - family_size : 가족 규모
# - begin_month : 신용카드 발급 월


## Data Preprocess ##


# Y variable Credit - 0, 1, 2 (multi classification)
train$credit = as.factor(train$credit) 


preprocess = function(data){
  dat = data %>%
    mutate (DAYS_BIRTH = as.integer(-DAYS_BIRTH / 365)) %>%
    mutate (DAYS_EMPLOYED = as.integer(-DAYS_EMPLOYED/365)) %>%
    mutate (begin_month = - begin_month) %>%
    filter (family_size <= 7)
    
  
  dat$FLAG_MOBIL = NULL
  
  return (dat)
}


train = preprocess(train)
test = preprocess(test)





## occyp_type에 8171개의 NA 존재



# Y Variable ratio 맞춰서 Train, Validation set 구성
idx = createDataPartition(train$credit, p=c(0.8, 0.2), list=FALSE)
train_tra = train[idx,]
train_val = train[-idx,]



forest = randomForest(credit~., data=train_tra[,-1])

pred = predict(forest, newdata = train_val)
confusionMatrix(pred,train_val$credit)


test_pred = predict(forest, newdata = test)
test$credit = test_pred

sub[,2:4] = one_hot(as.data.table(test$credit))
colnames(sub) = c("index",0,1,2)
write.table(sub, "my_submission.csv", sep=",", row.names = FALSE)

