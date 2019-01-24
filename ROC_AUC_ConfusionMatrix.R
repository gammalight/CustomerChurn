#Lets play with the caret package

library(caret)
library(caretEnsemble)
library(e1071)

fitControl <- trainControl(## 10-fold CV
  method = "cv",
  number = 10)#,
  ## repeated ten times
  #repeats = 10)
predDataFinal <- cbind(trainchurnData[c("ContractMonthtomonth",
                                        "tenure",
                                        "OnlineSecurityNo",
                                        "TechSupportNo", 
                                        "TotalCharges",
                                        "InternetServiceFiberoptic",
                                        "MonthlyCharges", 
                                        "PaymentMethodElectroniccheck",
                                        "InternetServiceDSL",
                                        "OnlineBackupNo", 
                                        "PaperlessBillingNo",
                                        "ContractOneyear",
                                        "DeviceProtectionNo",
                                        "churn")])
predDataFinal$churn <- as.factor(as.character(predDataFinal$churn))


set.seed(420)
rfFit <- train(churn ~ ., data = predDataFinal, 
                 method = "rf", 
                 trControl = fitControl,
                 metric = "Accuracy",
                 ## This last option is actually one
                 ## for gbm() that passes through
                 verbose = FALSE)
rfFit
rfFit$results
rfFit$finalModel



preds <- predict(rfFit, predDataFinal, type = 'prob')
preds <- preds[c(-1)]
names(preds) <- "churnPreds"
summary(preds)


predDataFinal2 <- cbind(predDataFinal, preds)

rocObj <- roc(predDataFinal2$churn, predDataFinal2$churnPreds)
auc(rocObj)
#0.8398

#find the Youden index to use as a cutoff for cunfusion matrix
coords(rocObj, "b", ret="t", best.method="youden") # default
#0.3368883

predDataFinal2 <- predDataFinal2 %>%
  mutate(predClass = if_else(churnPreds > 0.2, 1, 0),
         predClass = as.factor(as.character(predClass)))

confusionMatrix(data = predDataFinal2$predClass, 
                reference = predDataFinal2$churn)












