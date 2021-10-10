## Set Directory

current_working_dir = dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(current_working_dir)



## Import Library
library(data.table)
library(dplyr)
library(tidyverse)


## Load Dataset
train = read.table("DATA/train.csv", sep=",", header=T)
test = read.table("DATA/test.csv", sep=",", header=T)
sub = read.table("DATA/sample_submission.csv", sep=",", header=T)

## 데이터 변수 설명
# https://www.dacon.io/competitions/official/235713/overview/description

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



## Data Preprocess

# FLAG_MOBIL =1 at all row -> delete variable 
table(train$FLAG_MOBIL) + table(test$FLAG_MOBIL) 

# if family_size > 7 -> outlier. Test set family_size max = 7
table(train$family_size); table(test$family_size)


preprocess = function(data){
  result = data %>%
    mutate (DAYS_BIRTH = as.integer(-DAYS_BIRTH / 365)) %>%
    mutate (DAYS_EMPLOYED = as.integer(-DAYS_EMPLOYED/365)) %>%
    mutate (begin_month = - begin_month) %>%
    filter (family_size <= 7)
    
  result$FLAG_MOBIL = NULL
  result$DAYS_EMPLOYED[result$DAYS_EMPLOYED < 0] = 0 

  return (result)
}


train = preprocess(train)
test = preprocess(test)
