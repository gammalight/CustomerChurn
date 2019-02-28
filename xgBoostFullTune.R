#this script is used to predict customer churn 
#the data was downloaded off of Kaggle

#load packages that I will need to
#read in data
#conduct exploratory analysis
#manipulate data
#build several models


#load necessary packages
library(reshape)
library(ggplot2)
library(caret)
library(RColorBrewer)
library(pROC)
library(randomForest)
library(caTools)
library(gtools)
library(sqldf)
library(gbm)
library(Matrix)
library(foreach)
library(doParallel)
library(xgboost)
library(klaR)
library(dplyr)


#Calculate Logloss
LogLoss <- function(DEP, score, eps=0.00001) {
  score <- pmin(pmax(score, eps), 1-eps)
  -1/length(DEP)*(sum(DEP*log(score)+(1-DEP)*log(1-score)))
}


#read in the customer churn csv data file
churnData <- read.csv("TelcoCustomerChurn.csv")


###############################################################
### Feature Engineering ###
#will get to this later, just want to build some models to get baseline
#ideas for new variables:
# - Phone only
# - Internet only
# - paperless billing and auto pay

#Okay lets create these variables
churnData <- churnData %>%
  mutate(PhoneOnly = if_else(PhoneService == 'Yes' & InternetService == 'No', 'Yes', 'No'),
         InternetOnly = if_else(PhoneService == 'No' & InternetService != 'No', 'Yes', 'No'),
         PhoneInternet = if_else(PhoneService == 'Yes' & InternetService != 'No', 'Yes', 'No'),
         PaperlessAutoPay = if_else(PaperlessBilling == 'Yes' & 
                                      PaymentMethod %in% c("Bank transfer (automatic)","Credit card (automatic)"), 'Yes', 'No'),
         churn = if_else(Churn == 'Yes',1,0)) %>%
  select(-Churn)


###############################################################
#Well lets build some models
#Start with a simple logistic regression model, ye ole workhorse, baseline finding model
###############################################################

#first drop all tenure 0 people
churnData <- churnData %>%
  select(-customerID) %>% #deselect CustomerID
  filter(tenure > 0) %>%
  droplevels()



## Create Dummy Variables ##
dmy <- dummyVars(" ~ gender + Partner + Dependents + PhoneService +
                 MultipleLines + InternetService + OnlineSecurity +
                 OnlineBackup + DeviceProtection + TechSupport +
                 StreamingTV + StreamingMovies + Contract + PaperlessBilling +
                 PaymentMethod + PhoneOnly + InternetOnly + PaperlessAutoPay +
                 PhoneInternet", 
                 data = churnData,
                 fullRank = FALSE)
dmyData <- data.frame(predict(dmy, newdata = churnData))
#print(head(dmyData))

#strip the "." out of the column names
colClean <- function(x){ colnames(x) <- gsub("\\.", "", colnames(x)); x } 
dmyData <- colClean(dmyData) 

#lets combine the new dummy variables back with the original continuous variables
churnDataFinal <- cbind(dmyData, churnData[,c(2,5,18,19,24)])


#lets get a traing and test data set using the createPartition function from Caret
set.seed(420)
inTrain <- createDataPartition(churnDataFinal$churn, p = 0.8, list = FALSE, times = 1)

trainchurnData <- churnDataFinal[inTrain,]
testchurnData <- churnDataFinal[-inTrain,]

prop.table(table(churnData$churn))
prop.table(table(trainchurnData$churn))
prop.table(table(testchurnData$churn))




##########################################################
#Lets use RFE and the backward selection method
#Hopefully this is a little more efficient then building
#a larger model and doing variable selection that way
##########################################################
## lets use bag ##
treebagFuncs$summary <- twoClassSummary
subsets <- c(7,14,21,28,35)
ctrl <- rfeControl(functions =  treebagFuncs,
                   method = "cv",
                   repeats = 5,
                   verbose = FALSE)

y <- as.factor(as.character(churnDataFinal$churn))
x <- churnDataFinal[,-54]

# Set up to do parallel processing
no_cores <- detectCores()  # Number of cores
cl <- makeCluster(no_cores - 2)
registerDoParallel(no_cores)		# Registrer a parallel backend for train
getDoParWorkers()

t1 <- Sys.time()
treeBagProfilePar <- rfe(x
                         ,y
                         ,metric = "ROC"
                         ,sizes = subsets
                         ,rfeControl = ctrl)
t2 <- Sys.time()
print(t2 - t1)
stopCluster(cl)


treeBagProfilePar
names(treeBagProfilePar$fit$X)
nms <- names(treeBagProfilePar$fit$X)
fix(nms)

treeBagProfilePar$fit$mtrees #$variable.importance

#trellis.par.set(caretTheme())
#plot(lmProfile, type = c("g", "o"))
Top_Bag_Vars <- c("TotalCharges", "MonthlyCharges", "tenure", "OnlineSecurityNo", 
  "TechSupportNo", "ContractMonthtomonth", "InternetServiceFiberoptic", 
  "PaymentMethodElectroniccheck", "genderFemale", "genderMale", 
  "PartnerNo", "PaperlessBillingNo", "OnlineBackupNo", "PartnerYes", 
  "DependentsNo", "PaperlessBillingYes", "MultipleLinesNo", "SeniorCitizen", 
  "DependentsYes", "DeviceProtectionNo", "OnlineSecurityYes", "MultipleLinesYes", 
  "OnlineBackupYes", "StreamingTVNo", "TechSupportYes", "StreamingMoviesNo", 
  "DeviceProtectionYes", "PaymentMethodBanktransferautomatic", 
  "StreamingTVYes", "StreamingMoviesYes", "PaymentMethodCreditcardautomatic", 
  "PaymentMethodMailedcheck", "ContractTwoyear", "PaperlessAutoPayNo", 
  "PaperlessAutoPayYes")

write.csv(Top_Bag_Vars, "Top_Bag_Vars.csv", row.names = FALSE)



## lets use rf and compare to Bg ##
rfFuncs$summary <- twoClassSummary
subsets <- c(7,14,21,28,35)
ctrl <- rfeControl(functions =  rfFuncs,
                   method = "cv",
                   repeats = 5,
                   verbose = FALSE)

y <- as.factor(as.character(churnDataFinal$churn))
x <- churnDataFinal[,-54]

# Set up to do parallel processing
no_cores <- detectCores()  # Number of cores
cl <- makeCluster(no_cores - 2)
registerDoParallel(no_cores)	# Registrer a parallel backend for train
getDoParWorkers()

t1 <- Sys.time()
rfFuncsProfilePar <- rfe(x
                        ,y
                        ,metric = "ROC"
                        ,sizes = subsets
                        ,rfeControl = ctrl)
t2 <- Sys.time()
print(t2 - t1)
stopCluster(cl)


rfFuncsProfilePar

rfFuncsProfilePar$optVariables

rfVarNames <- rfFuncsProfilePar$variables$var[117:137]
fix(rfVarNames)
Top_RF_Vars <- c("tenure", "TotalCharges", "MonthlyCharges", "InternetServiceFiberoptic", 
  "OnlineSecurityNo", "ContractMonthtomonth", "TechSupportNo", 
  "InternetServiceDSL", "ContractTwoyear", "TechSupportYes", "PaperlessBillingNo", 
  "OnlineSecurityYes", "OnlineBackupNo", "PaymentMethodElectroniccheck", 
  "MultipleLinesYes", "PaperlessBillingYes", "MultipleLinesNo", 
  "SeniorCitizen", "StreamingMoviesNointernetservice", "TechSupportNointernetservice", 
  "OnlineBackupNointernetservice")


write.csv(Top_RF_Vars, "Top_RF_Vars.csv", row.names = FALSE)


###################################################
## lets use naive bayes and compare to Bg and RF ##
nbFuncs$summary <- twoClassSummary
subsets <- c(7,14,21,28,35)
ctrl <- rfeControl(functions =  nbFuncs,
                   method = "cv",
                   repeats = 5,
                   verbose = FALSE)

y <- as.factor(as.character(churnDataFinal$churn))
x <- churnDataFinal[,-54]

# Set up to do parallel processing
no_cores <- detectCores()  # Number of cores
cl <- makeCluster(no_cores - 2)
registerDoParallel(no_cores)	# Registrer a parallel backend for train
getDoParWorkers()

t1 <- Sys.time()
nbFuncsProfilePar <- rfe(x
                         ,y
                         ,metric = "ROC"
                         ,sizes = subsets
                         ,rfeControl = ctrl)
t2 <- Sys.time()
print(t2 - t1)
stopCluster(cl)


nbFuncsProfilePar
#names(nbFuncsProfilePar[[3]]$fit$X)

nbFuncsProfileParNames <- nbFuncsProfilePar$variables$var[117:137]
fix(nbFuncsProfileParNames)

Top_NB_Vars <- c("tenure", "ContractMonthtomonth", "OnlineSecurityNo", "TechSupportNo", 
  "InternetServiceFiberoptic", "PaymentMethodElectroniccheck", 
  "OnlineBackupNo", "TotalCharges", "ContractTwoyear", "DeviceProtectionNo", 
  "MonthlyCharges", "PhoneInternetYes", "PhoneInternetNo", "InternetServiceNo", 
  "OnlineSecurityNointernetservice", "OnlineBackupNointernetservice", 
  "DeviceProtectionNointernetservice", "TechSupportNointernetservice", 
  "StreamingTVNointernetservice", "StreamingMoviesNointernetservice", 
  "PhoneOnlyNo")


write.csv(Top_NB_Vars, "Top_NB_Vars.csv", row.names = FALSE)


Top_Sub_Vars <- c("ContractMonthtomonth","ContractTwoyear","DependentsNo","DependentsYes",
                  "DeviceProtectionNo","DeviceProtectionNointernetservice",
                  "DeviceProtectionYes","genderFemale","genderMale","InternetServiceDSL",
                  "InternetServiceFiberoptic","InternetServiceNo",
                  "MonthlyCharges","MultipleLinesNo","MultipleLinesYes","OnlineBackupNo",
                  "OnlineBackupNointernetservice","OnlineBackupYes","OnlineSecurityYes","PaperlessAutoPayNo",
                  "OnlineSecurityNo","OnlineSecurityNointernetservice",
                  "PaperlessAutoPayYes","PaperlessBillingNo","PaperlessBillingYes","PartnerNo",
                  "PartnerYes","PaymentMethodBanktransferautomatic",
                  "PaymentMethodCreditcardautomatic","PaymentMethodElectroniccheck",
                  "PaymentMethodMailedcheck","PhoneInternetNo",
                  "PhoneInternetYes","PhoneOnlyNo","SeniorCitizen","StreamingMoviesNo",
                  "StreamingMoviesNointernetservice","StreamingMoviesYes",
                  "StreamingTVNo","StreamingTVNointernetservice",
                  "StreamingTVYes","TechSupportNo","tenure","TotalCharges")




#Now that we have a decent list of candidate variables to use in the final models
#Lets start building some models
#Start with xbBoost with grid search

y <- as.factor(make.names(trainchurnData$churn))
x <- trainchurnData[,Top_Sub_Vars]
xHoldout <- testchurnData[,Top_Sub_Vars]

# Get baseline model
grid_default <- expand.grid(
  nrounds = 100,
  max_depth = 6,
  eta = 0.3,
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 1
)

train_control <- caret::trainControl(
  method = "none"
  , verboseIter = TRUE
  , returnData = FALSE
  , summaryFunction=twoClassSummary	# Use AUC to pick the best model
  , returnResamp = "all" # save losses across all models
  , classProbs = TRUE # set to TRUE for AUC to be computedsummaryFunction = twoClassSummary
  , allowParallel = TRUE
)

xgb_base <- caret::train(
  x = as.matrix(x),
  y = y,
  trControl = train_control,
  tuneGrid = grid_default,
  method = "xgbTree",
  verbose = TRUE,
  metric = "ROC"
)


#######################################################
# Score
pred <- data.frame(predict(xgb_base,data.matrix(x),type="prob")) ##type= options are response, prob. or votes
pred <- pred[c(-1)]
names(pred) <- "score"
summary(pred)

# Apply Deciles
# 0.1 option makes 10 equal groups (.25 would be 4).  negative option (-pred$score) makes the highest score equal to 1
rank <- data.frame(quantcut(-pred$score, q=seq(0, 1, 0.1), labels=F))
names(rank) <- "rank"

# apply the rank
Final_Scored <- cbind(trainchurnData,pred,rank)

#Run AUC
#use the two different ways
auc_out <- colAUC(Final_Scored$score, Final_Scored$churn, plotROC=TRUE, alg=c("Wilcoxon","ROC"))
rocObj <- roc(Final_Scored$churn, Final_Scored$score)
auc(rocObj)

#Run Decile Report: do average of all model vars, avg DEP and min score, max score and avg score
decile_report <- sqldf("select rank, count(*) as qty, sum(churn) as Responders, min(score) as min_score,
                       max(score) as max_score, avg(score) as avg_score
                       from Final_Scored
                       group by rank")


#######################################################
#######################################################
#Lets see how well this model holds up against the hold out sample
#######################################################
#######################################################
# Score
pred <- data.frame(predict(xgb_base,data.matrix(xHoldout),type="prob")) ##type= options are response, prob. or votes
pred <- pred[c(-1)]
names(pred) <- "score"
summary(pred)

# Apply Deciles
# 0.1 option makes 10 equal groups (.25 would be 4).  negative option (-pred$score) makes the highest score equal to 1
rank <- data.frame(quantcut(-pred$score, q=seq(0, 1, 0.1), labels=F))
names(rank) <- "rank"

# apply the rank
Final_Scored <- cbind(testchurnData,pred,rank)

#Run AUC
#use the two different ways
auc_out <- colAUC(Final_Scored$score, Final_Scored$churn, plotROC=TRUE, alg=c("Wilcoxon","ROC"))
rocObj <- roc(Final_Scored$churn, Final_Scored$score)
auc(rocObj)

#Run Decile Report: do average of all model vars, avg DEP and min score, max score and avg score
decile_report <- sqldf("select rank
                       , count(*) as qty
                       , sum(churn) as Responders
                       , min(score) as min_score
                       , max(score) as max_score
                       , avg(score) as avg_score
                       from Final_Scored
                       group by rank")

write.csv(decile_report,"decile_report.csv")

#Calculate the Logloss metric
LogLoss(Final_Scored$DEP,Final_Scored$score)
#0.4132963

#find the Youden index to use as a cutoff for cunfusion matrix
coords(rocObj, "b", ret="t", best.method="youden") # default
#0.3083464

#Classify row as 1/0 depending on what the calculated score is
#play around with adjusting the score to maximize accuracy or any metric
Final_Scored <- Final_Scored %>%
  mutate(predClass = if_else(score > 0.51, 1, 0),
         predClass = as.factor(as.character(predClass)),
         DEPFac = as.factor(as.character(DEP)))

#Calculate Confusion Matrix
confusionMatrix(data = Final_Scored$predClass, 
                reference = Final_Scored$DEPFac)




########################################################################
########################################################################
########################################################################


# set up function to perform cross-valdation on the models that are built
xgb_trcontrol_1 = trainControl(method = "cv"
                               , number = 3
                               , verboseIter = TRUE
                               , returnData = FALSE
                               , summaryFunction=twoClassSummary	# Use AUC to pick the best model
                               , returnResamp = "all" # save losses across all models
                               , classProbs = TRUE # set to TRUE for AUC to be computedsummaryFunction = twoClassSummary
                               , allowParallel = TRUE
)

# set up the cross-validated hyper-parameter search
xgb_grid_1 = expand.grid(nrounds = seq(from = 50, to = 1000, by = 50)
                         , subsample = 1 #c(0.5, 0.6)
                         , eta = c(0.005, 0.01, 0.025, 0.05) #c(0.01, 0.1)
                         , max_depth = c(2, 3, 4, 6) #(5, 10, 15)
                         , gamma = 0 #c(0, 1, 2, 4) 
                         , colsample_bytree = 1 #c(0.4, 0.5)
                         , min_child_weight = 1 #c(5, 25, 50)
)

# Set up to do parallel processing
no_cores <- detectCores()  # Number of cores
cl <- makeCluster(no_cores - 2)
registerDoParallel(no_cores)		# Registrer a parallel backend for train
getDoParWorkers()

t1 <- Sys.time()
xgbTrain <- train(x = as.matrix(x)
                  , y = y
                  , trControl = xgb_trcontrol_1
                  , tuneGrid = xgb_grid_1
                  , method = "xgbTree"
                  , metric = "ROC")
t2 <- Sys.time()
print(t2 - t1)
stopCluster(cl)

xgbTrain$bestTune
plot(xgbTrain)  		# Plot the performance of the training models
res <- xgbTrain$results
res






# set up the cross-validated hyper-parameter search
xgb_grid_2 = expand.grid(nrounds = seq(from = 50, to = 1000, by = 50)
                         , subsample = 1 #c(0.5, 0.6)
                         , eta = 0.01
                         , max_depth = c(2, 3, 4, 5) #(5, 10, 15)
                         , gamma = 0 #c(0, 1, 2, 4) 
                         , colsample_bytree = 1 #c(0.4, 0.5)
                         , min_child_weight = c(1, 2, 3, 4, 5, 6)
)

# Set up to do parallel processing
no_cores <- detectCores()  # Number of cores
cl <- makeCluster(no_cores - 2)
registerDoParallel(no_cores)		# Registrer a parallel backend for train
getDoParWorkers()

t1 <- Sys.time()
xgbTrain <- train(x = as.matrix(x)
                  , y = y
                  , trControl = xgb_trcontrol_1
                  , tuneGrid = xgb_grid_2
                  , method = "xgbTree"
                  , metric = "ROC")
t2 <- Sys.time()
print(t2 - t1)
stopCluster(cl)

xgbTrain$bestTune
plot(xgbTrain)  		# Plot the performance of the training models
res <- xgbTrain$results
res






# set up the cross-validated hyper-parameter search
xgb_grid_3 = expand.grid(nrounds = seq(from = 50, to = 1000, by = 50)
                         , subsample = 1 #c(0.5, 0.6)
                         , eta = xgbTrain$bestTune$eta
                         , max_depth = xgbTrain$bestTune$max_depth
                         , gamma = 0 #c(0, 1, 2, 4) 
                         , colsample_bytree = 1 #c(0.4, 0.5)
                         , min_child_weight = xgbTrain$bestTune$min_child_weight
)

# Set up to do parallel processing
no_cores <- detectCores()  # Number of cores
cl <- makeCluster(no_cores - 2)
registerDoParallel(no_cores)		# Registrer a parallel backend for train
getDoParWorkers()

t1 <- Sys.time()
xgbTrain <- train(x = as.matrix(x)
                  , y = y
                  , trControl = xgb_trcontrol_1
                  , tuneGrid = xgb_grid_3
                  , method = "xgbTree"
                  , metric = "ROC")
t2 <- Sys.time()
print(t2 - t1)
stopCluster(cl)

xgbTrain$bestTune
plot(xgbTrain)  		# Plot the performance of the training models
res <- xgbTrain$results
res






# set up the cross-validated hyper-parameter search
xgb_grid_4 = expand.grid(nrounds = seq(from = 50, to = 1000, by = 50)
                         , subsample = c(0.5, 0.6, 0.7, 0.8)
                         , eta = xgbTrain$bestTune$eta
                         , max_depth = xgbTrain$bestTune$max_depth
                         , gamma = 0 #c(0, 1, 2, 4) 
                         , colsample_bytree = c(0.4, 0.6, 0.8, 1.0)
                         , min_child_weight = xgbTrain$bestTune$min_child_weight
)

# Set up to do parallel processing
no_cores <- detectCores()  # Number of cores
cl <- makeCluster(no_cores - 2)
registerDoParallel(no_cores)		# Registrer a parallel backend for train
getDoParWorkers()

t1 <- Sys.time()
xgbTrain <- train(x = as.matrix(x)
                  , y = y
                  , trControl = xgb_trcontrol_1
                  , tuneGrid = xgb_grid_4
                  , method = "xgbTree"
                  , metric = "ROC")
t2 <- Sys.time()
print(t2 - t1)
stopCluster(cl)

xgbTrain$bestTune
plot(xgbTrain)  		# Plot the performance of the training models
res <- xgbTrain$results
res







# set up the cross-validated hyper-parameter search
xgb_grid_5 = expand.grid(nrounds = xgbTrain$bestTune$nrounds
                         , subsample = xgbTrain$bestTune$subsample
                         , eta = xgbTrain$bestTune$eta
                         , max_depth = xgbTrain$bestTune$max_depth
                         , gamma = c(0, 0.05, 0.1, 0.5, 0.7, 0.9, 1.0)
                         , colsample_bytree = xgbTrain$bestTune$colsample_bytree
                         , min_child_weight = xgbTrain$bestTune$min_child_weight
)

# Set up to do parallel processing
no_cores <- detectCores()  # Number of cores
cl <- makeCluster(no_cores - 2)
registerDoParallel(no_cores)		# Registrer a parallel backend for train
getDoParWorkers()

t1 <- Sys.time()
xgbTrain <- train(x = as.matrix(x)
                  , y = y
                  , trControl = xgb_trcontrol_1
                  , tuneGrid = xgb_grid_5
                  , method = "xgbTree"
                  , metric = "ROC")
t2 <- Sys.time()
print(t2 - t1)
stopCluster(cl)

xgbTrain$bestTune
plot(xgbTrain)  		# Plot the performance of the training models
res <- xgbTrain$results
res







# set up the cross-validated hyper-parameter search
xgb_grid_6 = expand.grid(nrounds = xgbTrain$bestTune$nrounds
                         , subsample = xgbTrain$bestTune$subsample
                         , eta = c(0.005, 0.01, 0.015, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5)
                         , max_depth = xgbTrain$bestTune$max_depth
                         , gamma = xgbTrain$bestTune$gamma
                         , colsample_bytree = xgbTrain$bestTune$colsample_bytree
                         , min_child_weight = xgbTrain$bestTune$min_child_weight
)

# Set up to do parallel processing
no_cores <- detectCores()  # Number of cores
cl <- makeCluster(no_cores - 2)
registerDoParallel(no_cores)		# Registrer a parallel backend for train
getDoParWorkers()

t1 <- Sys.time()
xgbTrain <- train(x = as.matrix(x)
                  , y = y
                  , trControl = xgb_trcontrol_1
                  , tuneGrid = xgb_grid_6
                  , method = "xgbTree"
                  , metric = "ROC")
t2 <- Sys.time()
print(t2 - t1)
stopCluster(cl)

xgbTrain$bestTune
plot(xgbTrain)  		# Plot the performance of the training models
res <- xgbTrain$results
res








# set up the cross-validated hyper-parameter search
xgb_grid_final = expand.grid(nrounds = xgbTrain$bestTune$nrounds
                         , subsample = xgbTrain$bestTune$subsample
                         , eta = xgbTrain$bestTune$eta
                         , max_depth = xgbTrain$bestTune$max_depth
                         , gamma = xgbTrain$bestTune$gamma
                         , colsample_bytree = xgbTrain$bestTune$colsample_bytree
                         , min_child_weight = xgbTrain$bestTune$min_child_weight
)

# Set up to do parallel processing
no_cores <- detectCores()  # Number of cores
cl <- makeCluster(no_cores - 2)
registerDoParallel(no_cores)		# Registrer a parallel backend for train
getDoParWorkers()

t1 <- Sys.time()
xgbTrainFinal <- train(x = as.matrix(x)
                  , y = y
                  , trControl = xgb_trcontrol_1
                  , tuneGrid = xgb_grid_final
                  , method = "xgbTree"
                  , metric = "ROC")
t2 <- Sys.time()
print(t2 - t1)
stopCluster(cl)

xgbTrainFinal$bestTune
res <- xgbTrain$results
res



#######################################################
# Score
pred <- data.frame(predict(xgbTrainFinal,data.matrix(x),type="prob")) ##type= options are response, prob. or votes
pred <- pred[c(-1)]
names(pred) <- "score"
summary(pred)

# Apply Deciles
# 0.1 option makes 10 equal groups (.25 would be 4).  negative option (-pred$score) makes the highest score equal to 1
rank <- data.frame(quantcut(-pred$score, q=seq(0, 1, 0.1), labels=F))
names(rank) <- "rank"

# apply the rank
Final_Scored <- cbind(trainchurnData,pred,rank)

#Run AUC
#use the two different ways
auc_out <- colAUC(Final_Scored$score, Final_Scored$churn, plotROC=TRUE, alg=c("Wilcoxon","ROC"))
rocObj <- roc(Final_Scored$churn, Final_Scored$score)
auc(rocObj)
#0.8666


#Run Decile Report: do average of all model vars, avg DEP and min score, max score and avg score
decile_report <- sqldf("select rank, count(*) as qty, sum(churn) as Responders, min(score) as min_score,
                       max(score) as max_score, avg(score) as avg_score
                       from Final_Scored
                       group by rank")

#Calculate the Logloss metric
LogLoss(Final_Scored$churn,Final_Scored$score)
#0.397096

#######################################################
#######################################################
#Lets see how well this model holds up against the hold out sample
#######################################################
#######################################################
# Score
pred <- data.frame(predict(xgbTrainFinal,data.matrix(xHoldout),type="prob")) ##type= options are response, prob. or votes
pred <- pred[c(-1)]
names(pred) <- "score"
summary(pred)

# Apply Deciles
# 0.1 option makes 10 equal groups (.25 would be 4).  negative option (-pred$score) makes the highest score equal to 1
rank <- data.frame(quantcut(-pred$score, q=seq(0, 1, 0.1), labels=F))
names(rank) <- "rank"

# apply the rank
Final_Scored <- cbind(testchurnData,pred,rank)

#Run AUC
#use the two different ways
auc_out <- colAUC(Final_Scored$score, Final_Scored$churn, plotROC=TRUE, alg=c("Wilcoxon","ROC"))
rocObj <- roc(Final_Scored$churn, Final_Scored$score)
auc(rocObj)
#0.8501

#Run Decile Report: do average of all model vars, avg DEP and min score, max score and avg score
decile_report <- sqldf("select rank
                       , count(*) as qty
                       , sum(churn) as Responders
                       , min(score) as min_score
                       , max(score) as max_score
                       , avg(score) as avg_score
                       from Final_Scored
                       group by rank")

write.csv(decile_report,"decile_report.csv")

#Calculate the Logloss metric
LogLoss(Final_Scored$churn,Final_Scored$score)
#0.4003244

#find the Youden index to use as a cutoff for cunfusion matrix
coords(rocObj, "b", ret="t", best.method="youden") # default
#0.2685123

#Classify row as 1/0 depending on what the calculated score is
#play around with adjusting the score to maximize accuracy or any metric
Final_Scored <- Final_Scored %>%
  mutate(predClass = if_else(score > 0.51, 1, 0),
         predClass = as.factor(as.character(predClass)),
         DEPFac = as.factor(as.character(DEP)))

#Calculate Confusion Matrix
confusionMatrix(data = Final_Scored$predClass, 
                reference = Final_Scored$DEPFac)




########################################################################
########################################################################
########################################################################








#################################################
## lets use lm and compare to Bg and RF ##
lmFuncs$summary <- twoClassSummary
subsets <- c(7,14,21,28,35)
ctrl <- rfeControl(functions =  lmFuncs,
                   method = "cv",
                   repeats = 5,
                   verbose = FALSE)

y <- as.factor(as.character(churnDataFinal$churn))
x <- churnDataFinal[,-54]


# Set up to do parallel processing
no_cores <- detectCores()  # Number of cores
cl <- makeCluster(no_cores - 2)
registerDoParallel(no_cores)	# Registrer a parallel backend for train
getDoParWorkers()

t1 <- Sys.time()
lmFuncsProfilePar <- rfe(x
                         ,y
                         ,metric = "ROC"
                         ,sizes = subsets
                         ,rfeControl = ctrl)
t2 <- Sys.time()
print(t2 - t1)
stopCluster(cl)


lmFuncsProfilePar
names(lmFuncsProfilePar[[3]]$fit$X)



















































































