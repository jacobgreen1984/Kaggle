# title : home credit 
# author : jacob 

# library 
options(scipen = 999)
rm(list=ls())
gc(reset=TRUE)
library(data.table)
library(e1071)
library(caret)
library(Metrics)
require(Matrix)
require(lightgbm)  
library(xgboost)
library(catboost)
library(rBayesianOptimization)

# path 
path_code   = "~/GitHub/2econsulting/Kaggle/competition/homecredit/base"
path_output = "~/GitHub/2econsulting/Kaggle_data/homecredit/output" 
path_input  = "~/GitHub/2econsulting/Kaggle_data/homecredit/input"

# train options 
y = "TARGET"
sample_rate = 1
kfolds = 5
early_stopping_rounds = 100
iterations = 10000
num_threads = 8
learning_rate = 0.02

# tuning code
source(file.path(path_code,"LGB/tuneLGB.R"))
source(file.path(path_code,"LGB/cvpredictLGB2.R"))

# table_nm
table_nm = "best"

# set file
file_data = file.path(table_nm,paste0(table_nm,"_train.csv"))
file_test = file.path(table_nm,paste0(table_nm,"_test.csv"))
file_cat = file.path(table_nm,"categorical_features.csv")

# read data
data = fread(file.path(path_input, file_data))
test = fread(file.path(path_input, file_test))
cat_features = fread(file.path(path_input, file_cat))$categorical_features

# LGB
params = list(
  learning_rate = 0.02,
  num_leaves = 30,
  colsample_bytree = 0.05,
  subsample = 1,
  subsample_freq = 1, 
  max_depth = -1,
  max_bin = 300,
  reg_alpha = 0,
  reg_lambda = 100,
  min_split_gain = 0.0222415,
  min_child_samples = 70,
  min_gain_to_split = 0.5,
  scale_pos_weight = 1, 
  is_unbalance = FALSE,
  verbose = -1,
  metric = "auc"
)
output <- cvpredictLGB2(data, test, k=kfolds, y=y, params=params, seed=0, cat_features=cat_features)
cat(">> cv_score :", output$score)





