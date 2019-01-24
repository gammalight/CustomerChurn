#this script is used to predict customer churn 
#the data was downloaded off of Kaggle

#load packages that I will need to
#read in data
#conduct exploratory analysis
#manipulate data
#build several models


#load necessary packages
library(dplyr)
library(reshape)
library(ggplot2)
library(caret)
library(caretEnsemble)
library(RcolorBrewer)
library(pROC)

#read in the customer churn csv data file
churnData <- read.csv("TelcoCustomerChurn.csv")

#take a quick look at the data
glimpse(churnData)

############################################
#lets get an idea of how data is distributed
############################################
summary(churnData)
#Gender
#Female - 3488
#Male - 3555

#SeniorCitizen
#Not Senior - 5901
#Senior - 1142

#Partner
#No - 3641
#yes - 3402

#Dependents
#No - 4933
#Yes - 2110

#Phone Service
#No - 682
#Yes - 6361

#Multiple Lines
#No - 3390
#Yes - 2971
#No Phone Service - 682

#Internet Service
#DSL - 2421
#Fiber Optic - 3096
#No - 1526

#Online Security
#No - 3498
#Yes - 2019
#No Internet Service - 1526

#Online Backup
#No - 3088
#Yes - 2429
#No Internet Service - 1526

#Device Protection
#No - 3095
#Yes - 2422
#No Internet Service - 1526

#Tech Support
#No - 3473
#Yes - 2044
#No Internet Service - 1526

#Streaming TV
#No - 2810
#Yes - 2707
#No Internet Service - 1526

#Streaming Movies
#No - 2785
#Yes - 2732
#No Internet Service - 1526

#Contract 
#Month to Month - 3875
#1 year - 1473
#2 year - 1695

#Paperless Billing
#No - 2872
#Yes - 4171

#Payment Method 
#Bank Transfer (auto) - 1544
#Credit Card (auto) - 1522
#Electronic Check - 2365
#Mailed Check - 1612

#######################
### Continuous Vars ###
#######################
#Tenure
#min - 0
#med - 29
#mean - 32.37
#max - 72

#Monthly Charges
#min - 18.25
#med - 70.35
#mean - 64.76
#max - 118.75

#Total Charges
#min - 18.8
#med - 1397.5
#mean - 2283.3
#Max - 8684.8
#NA's - 11

#Churn
#No - 5174 (73%)
#Yes - 1869 (27%)

### Lets do more exploratory analysis ###
#See if customer id is unique for each row
#use base R
length(unique(churnData$customerID)) == nrow(churnData)

#Use dplyr
churnData %>%
  summarise(nDisCnt = n_distinct(customerID),
            nCnt = n())
#Yes it is

#Lets investigate the NA's for Total Charges
#I bet they are new customers
#If so we can remove them from the data set
#this is because it will be their first month/charge and they dont have a chance to churn yet
churnData %>%
  filter(is.na(TotalCharges)) %>%
  summarise(avgTenure = mean(tenure),
            minTenure = min(tenure),
            maxTenure = max(tenure))
#Yes, these are all new customers

#now look at it from the new customer perspective
churnData %>%
  filter(tenure == 0) %>%
  summarise(avgTotalCharge = mean(TotalCharges),
            minTotalCharge = min(TotalCharges),
            maxTotalCharge = max(TotalCharges))
#Yes, all total charges are NA for customers of 0 tenure



###############################################################
###############################################################
#Lets do some exploratory plots 
#to see how categorical variables are distributed with dependent variable
#pre-processing
###############################################################
#Gender vs Churn
genderChurn <- melt(table(churnData$gender,churnData$Churn))
colnames(genderChurn) <- c("Gender", "Churn", "CumuPerc")
#plot
ggplot(genderChurn, aes(x=Gender, y=CumuPerc, fill = Churn)) +
  geom_bar(position="fill", stat="identity") +
  theme_linedraw()  #+ if you want to plot with different colors
  #scale_fill_brewer(palette = "Greys")
#gender appears to not have an effect on churn
###############################################################



###############################################################
#SeniorCitizen vs Churn
seniorCitChurn <- melt(table(churnData$SeniorCitizen,churnData$Churn))
colnames(seniorCitChurn) <- c("SeniorCitizen", "Churn", "CumuPerc")
#plot
ggplot(seniorCitChurn, aes(x=factor(SeniorCitizen), y=CumuPerc, fill = Churn)) +
  geom_bar(position="fill", stat="identity") +
  theme_linedraw()  #+ if you want to plot with different colors
#scale_fill_brewer(palette = "Greys")
#being a senior citizen increases liklehood of churn
###############################################################



###############################################################
#Partner vs Churn
partnerChurn <- melt(table(churnData$Partner,churnData$Churn))
colnames(partnerChurn) <- c("Partner", "Churn", "CumuPerc")
#plot
ggplot(partnerChurn, aes(x=Partner, y=CumuPerc, fill = Churn)) +
  geom_bar(position="fill", stat="identity") +
  theme_linedraw()  #+ if you want to plot with different colors
#scale_fill_brewer(palette = "Greys")
#not being a partner increases liklehood of churn
###############################################################



###############################################################
#Dependents vs Churn
dependentChurn <- melt(table(churnData$Dependents,churnData$Churn))
colnames(dependentChurn) <- c("Dependents", "Churn", "CumuPerc")
#plot
ggplot(dependentChurn, aes(x=Dependents, y=CumuPerc, fill = Churn)) +
  geom_bar(position="fill", stat="identity") +
  theme_linedraw()  #+ if you want to plot with different colors
#scale_fill_brewer(palette = "Greys")
#not having dependents increases likelhood of churn
###############################################################



###############################################################
#PhoneService vs Churn
phoneServiceChurn <- melt(table(churnData$PhoneService,churnData$Churn))
colnames(phoneServiceChurn) <- c("PhoneService", "Churn", "CumuPerc")
#plot
ggplot(phoneServiceChurn, aes(x=PhoneService, y=CumuPerc, fill = Churn)) +
  geom_bar(position="fill", stat="identity") +
  theme_linedraw()  #+ if you want to plot with different colors
#scale_fill_brewer(palette = "Greys")
#having phone service does not appear to have an effect on churn
###############################################################



###############################################################
#MultipleLines vs Churn
multipleLinesChurn <- melt(table(churnData$MultipleLines,churnData$Churn))
colnames(multipleLinesChurn) <- c("MultipleLines", "Churn", "CumuPerc")
#plot
ggplot(multipleLinesChurn, aes(x=MultipleLines, y=CumuPerc, fill = Churn)) +
  geom_bar(position="fill", stat="identity") +
  theme_linedraw()  #+ if you want to plot with different colors
#scale_fill_brewer(palette = "Greys")
#about same likelihood accross the board
###############################################################



###############################################################
#InternetService vs Churn
internetServiceChurn <- melt(table(churnData$InternetService,churnData$Churn))
colnames(internetServiceChurn) <- c("InternetService", "Churn", "CumuPerc")
#plot
ggplot(internetServiceChurn, aes(x=InternetService, y=CumuPerc, fill = Churn)) +
  geom_bar(position="fill", stat="identity") +
  theme_linedraw()  #+ if you want to plot with different colors
#scale_fill_brewer(palette = "Greys")
#fiberoptic subscribers are more likely to churn
#no internet service is less likely to chrun
#DSL more likely to churn than no internet service but less than Fiber
###############################################################



###############################################################
#OnlineSecurity vs Churn
onlineSecurityChurn <- melt(table(churnData$OnlineSecurity,churnData$Churn))
colnames(onlineSecurityChurn) <- c("OnlineSecurity", "Churn", "CumuPerc")
#plot
ggplot(onlineSecurityChurn, aes(x=OnlineSecurity, y=CumuPerc, fill = Churn)) +
  geom_bar(position="fill", stat="identity") +
  theme_linedraw()  #+ if you want to plot with different colors
#scale_fill_brewer(palette = "Greys")
#no online security is more likely to chrun
#online security is more likely to churn than no internet service
###############################################################



###############################################################
#OnlineBackup vs Churn
onlineBackupChurn <- melt(table(churnData$OnlineBackup,churnData$Churn))
colnames(onlineBackupChurn) <- c("OnlineBackup", "Churn", "CumuPerc")
#plot
ggplot(onlineBackupChurn, aes(x=OnlineBackup, y=CumuPerc, fill = Churn)) +
  geom_bar(position="fill", stat="identity") +
  theme_linedraw()  #+ if you want to plot with different colors
#scale_fill_brewer(palette = "Greys")
#no online backup is more likely to chrun
#online backup is more likely to churn than no internet service
#*** I bet phone only is less likely to churn than if you have internet only or both
###############################################################



###############################################################
#DeviceProtection vs Churn
deviceProtectionChurn <- melt(table(churnData$DeviceProtection,churnData$Churn))
colnames(deviceProtectionChurn) <- c("DeviceProtection", "Churn", "CumuPerc")
#plot
ggplot(deviceProtectionChurn, aes(x=DeviceProtection, y=CumuPerc, fill = Churn)) +
  geom_bar(position="fill", stat="identity") +
  theme_linedraw()  #+ if you want to plot with different colors
#scale_fill_brewer(palette = "Greys")
#no device protection is more likely to chrun
#device protection is more likely to churn than no internet service
#*** I bet phone only is less likely to churn than if you have internet only or both
###############################################################



###############################################################
#TechSupport vs Churn
techSupportChurn <- melt(table(churnData$TechSupport,churnData$Churn))
colnames(techSupportChurn) <- c("TechSupport", "Churn", "CumuPerc")
#plot
ggplot(techSupportChurn, aes(x=TechSupport, y=CumuPerc, fill = Churn)) +
  geom_bar(position="fill", stat="identity") +
  theme_linedraw()  #+ if you want to plot with different colors
#scale_fill_brewer(palette = "Greys")
#no tech support is more likely to chrun
#tech support is more likely to churn than no internet service
#*** I bet phone only is less likely to churn than if you have internet only or both
###############################################################



###############################################################
#StreamingTV vs Churn
streamingTVChurn <- melt(table(churnData$StreamingTV,churnData$Churn))
colnames(streamingTVChurn) <- c("StreamingTV", "Churn", "CumuPerc")
#plot
ggplot(streamingTVChurn, aes(x=StreamingTV, y=CumuPerc, fill = Churn)) +
  geom_bar(position="fill", stat="identity") +
  theme_linedraw()  #+ if you want to plot with different colors
#scale_fill_brewer(palette = "Greys")
#no streaming service is equally likely to chrun as streaming service
#no internet service is less likely to churn
#*** I bet phone only is less likely to churn than if you have internet only or both
###############################################################



###############################################################
#StreamingMovies vs Churn
streamingMoviesChurn <- melt(table(churnData$StreamingMovies,churnData$Churn))
colnames(streamingMoviesChurn) <- c("StreamingMovies", "Churn", "CumuPerc")
#plot
ggplot(streamingMoviesChurn, aes(x=StreamingMovies, y=CumuPerc, fill = Churn)) +
  geom_bar(position="fill", stat="identity") +
  theme_linedraw()  #+ if you want to plot with different colors
#scale_fill_brewer(palette = "Greys")
#no streaming service is equally likely to chrun as streaming service
#no internet service is less likely to churn
#*** I bet phone only is less likely to churn than if you have internet only or both
###############################################################



###############################################################
#Contract vs Churn
contractChurn <- melt(table(churnData$Contract,churnData$Churn))
colnames(contractChurn) <- c("Contract", "Churn", "CumuPerc")
#plot
ggplot(contractChurn, aes(x=Contract, y=CumuPerc, fill = Churn)) +
  geom_bar(position="fill", stat="identity") +
  theme_linedraw()  #+ if you want to plot with different colors
#scale_fill_brewer(palette = "Greys")
#month to month contracts are way more likely to churn than one or two year contracts
#two year contracts are far less likely to churn than one year contracts
#*** I bet people with two year contracts just havent had enough time to churn
#since they are under a longer contract obligation, they would have to pay to opt out and may be waiting for the contract to expire
#should look at tenure of the two-year contracts and see if thats true
###############################################################



###############################################################
#PaperlessBilling vs Churn
paperlessBillingChurn <- melt(table(churnData$PaperlessBilling,churnData$Churn))
colnames(paperlessBillingChurn) <- c("PaperlessBilling", "Churn", "CumuPerc")
#plot
ggplot(paperlessBillingChurn, aes(x=PaperlessBilling, y=CumuPerc, fill = Churn)) +
  geom_bar(position="fill", stat="identity") +
  theme_linedraw()  #+ if you want to plot with different colors
#scale_fill_brewer(palette = "Greys")
#paperless billing is more likely to churn
#i bet this is bc they forget to pay their bill
#should look at the interaction with auto pay
#i bet people with paperless billing that dont have autopay are more likely to churn
###############################################################



###############################################################
#PaymentMethod vs Churn
paymentMethodChurn <- melt(table(churnData$PaymentMethod,churnData$Churn))
colnames(paymentMethodChurn) <- c("PaymentMethod", "Churn", "CumuPerc")
#plot
ggplot(paymentMethodChurn, aes(x=PaymentMethod, y=CumuPerc, fill = Churn)) +
  geom_bar(position="fill", stat="identity") +
  theme_linedraw() #+ if you want to plot with different colors
#scale_fill_brewer(palette = "Greys")
#automatic payment is less likely to churn
#electronic check is more liekly to churn (of course, they are forgetting to pay the bill)
###############################################################



###############################################################
#Tenure vs Churn
#plot
ggplot(churnData, aes(x=Churn, y=tenure)) +
  geom_boxplot(alpha = 0.2, col='blue') +
  stat_summary(fun.y=mean, geom="point", shape=20, size=5, color="red", fill="red")
#higher tenure is less likely to churn

qplot(x=Churn , y=tenure , data=churnData , geom=c("boxplot","jitter") , fill=Churn)
#histogram of tenure
ggplot(churnData, aes(tenure)) +
  geom_histogram(bins = 20, aes(fill = ..count..)) +
  geom_vline(aes(xintercept = mean(tenure)),col='red',size=1)
###############################################################



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
         churn = if_else(Churn == 'Yes',1,0))


###############################################################
#phoneOnlyChurn vs Churn
phoneOnlyChurn <- melt(table(churnData$PhoneOnly,churnData$Churn))
colnames(phoneOnlyChurn) <- c("PhoneOnly", "Churn", "CumuPerc")
#plot
ggplot(phoneOnlyChurn, aes(x=PhoneOnly, y=CumuPerc, fill = Churn)) +
  geom_bar(position="fill", stat="identity") +
  theme_linedraw() #+ if you want to plot with different colors
#scale_fill_brewer(palette = "Greys")
#phone only is less likely to churn
###############################################################


###############################################################
#Internet only vs Churn
internetOnlyChurn <- melt(table(churnData$InternetOnly,churnData$Churn))
colnames(internetOnlyChurn) <- c("InternetOnly", "Churn", "CumuPerc")
#plot
ggplot(internetOnlyChurn, aes(x=InternetOnly, y=CumuPerc, fill = Churn)) +
  geom_bar(position="fill", stat="identity") +
  theme_linedraw() #+ if you want to plot with different colors
#scale_fill_brewer(palette = "Greys")
#both phone and internet is more likely to churn
###############################################################


###############################################################
#PhoneInternet vs Churn
phoneInternetChurn <- melt(table(churnData$PhoneInternet,churnData$Churn))
colnames(internetOnlyChurn) <- c("PhoneInternet", "Churn", "CumuPerc")
#plot
ggplot(internetOnlyChurn, aes(x=PhoneInternet, y=CumuPerc, fill = Churn)) +
  geom_bar(position="fill", stat="identity") +
  theme_linedraw() #+ if you want to plot with different colors
#scale_fill_brewer(palette = "Greys")
#internet only is equally likely to churn
###############################################################


###############################################################
#paperlessAutoPayChurn vs Churn
paperlessAutoPayChurn <- melt(table(churnData$PaperlessAutoPay,churnData$Churn))
colnames(paperlessAutoPayChurn) <- c("PaperlessAutoPay", "Churn", "CumuPerc")
#plot
ggplot(paperlessAutoPayChurn, aes(x=PaperlessAutoPay, y=CumuPerc, fill = Churn)) +
  geom_bar(position="fill", stat="identity") +
  theme_linedraw() #+ if you want to plot with different colors
#scale_fill_brewer(palette = "Greys")
#paperless autopay is slightly less likely to churn
###############################################################


###############################################################
#Well lets build some models
#Start with a simple logistic regression model, ye ole workhorse, baseline finding model
###############################################################

#first drop all tenure 0 people
churnData <- churnData %>%
  select(-customerID) %>% #deselect CustomerID
  filter(tenure > 0) %>%
  droplevels()


############################
## check for near zero variance columns ##
############################
nzv <- nearZeroVar(churnData, saveMetrics= TRUE)
nzv[1:10,]
#no near zero variance predictors exist


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

#check for linear combinations in the data
#a lot should exist since i dummied out and left all the data in the set
comboInfo <- findLinearCombos(dmyData)
comboInfo

## now remove the columns that are causing the linear dependencies
dmyData <- dmyData[, -comboInfo$remove]

#lets combine the new dummy variables back with the original continuous variables
churnDataFinal <- cbind(dmyData, churnData[,c(2,5,18,19,25)])


#lets get a traing and test data set using the createPartition function from Caret

set.seed(420)
inTrain <- createDataPartition(churnDataFinal$churn, p = 4/5, list = FALSE, times = 1)

trainchurnData <- churnDataFinal[inTrain,]
testchurnData <- churnDataFinal[-inTrain,]

prop.table(table(churnData$churn))
prop.table(table(trainchurnData$churn))
prop.table(table(testchurnData$churn))


#################################
## try using the recursive feature elimination function in caret ##
#################################
subsets <- c(1:25)
ctrl <- rfeControl(functions = lmFuncs,
                   method = "repeatedcv",
                   repeats = 5,
                   verbose = FALSE)

y <- ifelse(churnDataFinal$Churn == 'Yes', 1, 0)
x <- churnDataFinal[,-54]

lmProfile <- rfe(x,
                 y,
                 sizes = subsets,
                 rfeControl = ctrl)

lmProfile

trellis.par.set(caretTheme())
plot(lmProfile, type = c("g", "o"))



#############################
## train some models using caret functionality ##
#############################


lmModel <- train(churn ~ .,
                 data = trainchurnData, 
                 method = "glm")

summary(lmModel)



VarSelectME<- function(checkDF){
  
  ### paramaters below are auto set - no need to manually code unless you would like to override
  N_trees <- 500 ## enter the number of Random Forest Trees you want built - if you are not sure 100 is ussually enough to converge nicely when doing variable selection only
  Recs<-length(Final[,1]) ## total records in the file
  Recs_Dep1 <-sum(Final$DEP)   ## how many total records where DEP=1 (dependent variable equals 1)
  Node_Size<-round(50/(Recs_Dep1/Recs)) ## auto calculation for min terminal node size so on average at least 50 DEP=1s are present in each terminal node '
  Max_Nodes<-NULL ## maximum number of terminal nodes.  20 to 25 is
  Sample_Size<-round(.3*Recs)  ## iteration sample size - 20% is usually good
  
  library("randomForest")
  set.seed(100)
  temp <- randomForest(checkDF[,indvarsc],checkDF$DEP
                       ,sampsize=c(Sample_Size),do.trace=TRUE,importance=TRUE,ntree=N_trees,replace=FALSE,forest=TRUE
                       ,nodesize=Node_Size,maxnodes=Max_Nodes,na.action=na.omit)
  
  RF_VARS <- as.data.frame(round(importance(temp), 2))
  RF_VARS <- RF_VARS[order(-RF_VARS$IncNodePurity) ,]  
  best_vi=as.data.frame(head(RF_VARS,N_Vars))
  
  topvars <-as.vector(row.names(best_vi))
  ## topvars now contains the top N variables
  
  return(topvars)
}
###################################################################################################


#####################################################################################
## function to split out the binary attributes
createBinaryDF <- function(depVar, checkDF){
  binaryCols <- c(depVar)
  nameVecBin<- names(checkDF)
  
  for (n in nameVecBin){
    if (n != depVar){
      checkBinary<-summary(checkDF[,n])
      # c("Min.", "1st Qu.", "Median", "Mean", "3rd Qu.", "Max.")
      isBinary<- ifelse((checkBinary["Min."] == 0 &   checkBinary["Max."] == 1
                         & (checkBinary["Median"]== 0 ||checkBinary["Median"]== 1)
                         & (checkBinary["1st Qu."]== 0 ||checkBinary["1st Qu."]== 1)
                         & (checkBinary["3rd Qu."]== 0 ||checkBinary["3rd Qu."]== 1)),"yes","no")
      
      
      
      if  (isBinary == "yes") {
        binaryCols<-  append(binaryCols, n)
        print(here<- paste("Adding binary: ",n ,sep=""))  
      }  
    } 
  }
  return(checkDF[,binaryCols])
}
#####################################################################################


######################################################################################
### Scaling function
scaleME <- function(checkDF){
  
  require(stats)
  
  ## center and scale the vars 
  checkDF<- as.data.frame(scale(checkDF),center=TRUE,scale=apply(seg,1,sd,na.rm=TRUE))
  
  ## take the cubed root
  cube_root<- function(x){x^1/3}
  checkDF<-as.data.frame(cube_root(checkDF))
  
  ## run softmax - convert all vars to a range of 0 to 1  
  ## 2 lines below do not work for some reason so needed to run the loop
  ## range01 <- function(x){(x-min(x))/(max(x)-min(x))}
  ## checkDF <- range01(checkDF) 
  nameVecBin<- names(checkDF)
  for (n in nameVecBin) {
    checkDF[,n]<-(checkDF[,n]-min(checkDF[,n]))/(max(checkDF[,n])-min(checkDF[,n]))
  }
  
  return(checkDF)
}
######################################################################################





library(reshape)
inputdf <- rename(trainchurnData, c(team_name="X"))
inputdf <- rename(trainchurnData, c(churn="DEP"))

## other options to import are the sas7bdat library (example below) and the R2SAS library
## useR2SAS once available - not available on the CRAN as of 1/20/2012
## library(sas7bdat)
## inputdf <- read.sas7bdat("L:/Work/SAS Data/ESPN/Jack/Ad Click Model/Ad_click_master_rl.sas7bdat")

## if you need to manually convert attributes below are some options
## inputdf$PurchDate <- as.Date(inputdf$PurchDate,"%m/%d/%Y")
## inputdf$WheelTypeID <- as.integer(inputdf$WheelTypeID)

names1 <- names(inputdf)
#fix(names1)



names2 <- c("DEP","genderFemale", "genderMale", "PartnerNo", "DependentsNo", 
            "PhoneServiceNo", "MultipleLinesNo", "InternetServiceDSL", "InternetServiceFiberoptic", 
            "OnlineSecurityNo", "OnlineBackupNo", "DeviceProtectionNo", "TechSupportNo", 
            "StreamingTVNo", "StreamingMoviesNo", "ContractMonthtomonth", 
            "ContractOneyear", "PaperlessBillingNo", "PaymentMethodBanktransferautomatic", 
            "PaymentMethodCreditcardautomatic", "PaymentMethodElectroniccheck", 
            "PaperlessAutoPayNo", "SeniorCitizen", "tenure", "MonthlyCharges", 
            "TotalCharges")



inputdf_1 <- inputdf[names2]
inputdf_1[is.na(inputdf_1)] <- 0


#summary(inputdf_1)
################################################################
## 2)split out the binary attributes

## see bottom section for the function called here (createBinaryDF)
droppedBinDF <-createBinaryDF("DEP", inputdf_1)
## droppedBindf contains the X and DEP variables as well

## now create the file with all non-binary attributes
delVar <- names(droppedBinDF)
## delVar <- delVar[delVar != "X"]    ## Keep X
## delVar <- delVar[delVar != "DEP"]  ## Keep DEP
mydropvars <- !((names(inputdf_1)) %in% (delVar))
inputdf2 <- inputdf_1[mydropvars]




################################################################
## 3)scale the non-binary attributes
## see bottom section for the function called here (scaleME)
#inputdf2 <-scaleME(inputdf2)



#inputdf3<- data.frame(scale(inputdf2, center=TRUE, scale=TRUE))



################################################################
## 4)Calculate transformations of non-binary attributes
## see bottom section for the function called here (scaleME)
#trans_vars <-transformME(inputdf2)





################################################################
## 5)Final File: combine all attributes
Final <- cbind(droppedBinDF, inputdf2)
summary(Final)

## drop all interim data
rm(droppedBinDF)
rm(trans_vars)
rm(inputdf2)
rm(delvar)
rm(mydropvars)








###################################################################################
## Build the Models: Random Forest                                               ##
###################################################################################


##################################################################################
## Create the independent variable string (indvarsc)
## save(Final, file = "Final.RData")
## load("Final.RData")

myvars <- names(Final) %in% c("Row.names","DEP")
tmp <- Final[!myvars]
indvarsc <- names(tmp)
rm(myvars)
rm(tmp)

rm(decile_report)
rm(auc_out)
rm(RF_VARS)
rm(pred)
rm(rank)


###################################################################################
## determine the top N variables to use using Random Forest - if you want to use all the variables skip this step
## you can think of this technique as similar tostepwise in regression

## manually enter the number of top variables selected you would like returned (must be <= total predictive vars)
N_Vars <- 13

### Run the variable selection procedure below.  The final top N variables will be returned
Top_N_Vars<-VarSelectME(Final)
names(Top_N_Vars)
fix(Top_N_Vars)

Top_N_Vars <- c("ContractMonthtomonth", "tenure", "OnlineSecurityNo", "TechSupportNo", 
                "TotalCharges", "InternetServiceFiberoptic", "MonthlyCharges", 
                "PaymentMethodElectroniccheck", "InternetServiceDSL", "OnlineBackupNo", 
                "PaperlessBillingNo", "ContractOneyear", "DeviceProtectionNo"
)





###################################################################################
## build random forest model based on top N variables  
library("randomForest")

### paramaters below are auto set - no need to manually code unless you would like to override
N_trees <- 500 ## enter the number of Random Forest Trees you want built - if you are not sure 500 is usually enough to converge nicely when doing variable selection only
Recs<-length(Final[,1]) ## total records in the file
Recs_Dep1 <-sum(Final$DEP)   ## how many total records where DEP=1 (dependent variable equals 1)
Node_Size<-round(50/(Recs_Dep1/Recs)) ## auto calculation for min terminal node size so on average at least 50 DEP=1s are present in each terminal node '
Max_Nodes<-20 ## maximum number of terminal nodes.  20 to 25 is
Sample_Size<-round(.7*Recs)  ## iteration sample size - 20% is usually good

set.seed(100)
Final_RF<- randomForest(Final[,Top_N_Vars],Final$DEP
                              ,sampsize=c(Sample_Size)
                              ,do.trace=TRUE
                              ,importance=TRUE
                              ,ntree=N_trees
                              ,replace=FALSE
                              ,forest=TRUE
                              ,nodesize=Node_Size
                              ,maxnodes=Max_Nodes
                              ,na.action=na.omit)

RF_VARS <- as.data.frame(round(importance(Final_RF), 2))
RF_VARS <- RF_VARS[order(-RF_VARS$IncNodePurity) ,]

save(Final_RF, file = "Final_RF_2011.RData")
load("Final_RF_2011.RData")

# Score
pred <- data.frame(predict(Final_RF,Final[,Top_N_Vars]),type="prob") ##type= options are response, prob. or votes
pred <- pred[c(-2)]
names(pred) <- "score"
summary(pred)

# Apply Deciles
library(gtools)
# 0.1 option makes 10 equal groups (.25 would be 4).  negative option (-pred$score) makes the highest score equal to 1
rank <- data.frame(quantcut(-pred$score, q=seq(0, 1, 0.1), labels=F))
names(rank) <- "rank"

# apply the rank
Final_Scored <- cbind(Final,pred,rank)

#Run AUC
library(caTools)
auc_out <- colAUC(Final_Scored$score, Final_Scored$DEP, plotROC=TRUE, alg=c("Wilcoxon","ROC"))

#Run Decile Report: do average of all model vars, avg DEP and min score, max score and avg score
library(sqldf)
decile_report <- sqldf("select rank, count(*) as qty, sum(DEP) as Responders, min(score) as min_score,
                       max(score) as max_score, avg(score) as avg_score
                       from Final_Scored
                       group by rank")

write.csv(decile_report,"decile_report.csv")

Final_Scored$score2 <- ifelse(Final_Scored$score > 0.78, 1, Final_Scored$score)


LogLoss <- function(DEP, score, eps=0.00001) {
  score <- pmin(pmax(score, eps), 1-eps)
  -1/length(DEP)*(sum(DEP*log(score)+(1-DEP)*log(1-score)))
}
LogLoss(Final_Scored$DEP,Final_Scored$score)


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
                                        "churn")],
                       pred)


rocObj <- roc(predDataFinal$churn, predDataFinal$score)
auc(rocObj)

coords(rocObj, "b", ret="t", best.method="youden") # default



##############################################################################################







#build GLM model with all variables just to see what we get
churnModelTrainAll <- glm(Churn ~ ., 
                       data = trainchurnData, 
                       family = binomial(link="logit"))
summary(churnModelTrainAll)


#build GLM model with handpicked variables just to see what we get
churnModelTrain <- glm(Churn ~ DeviceProtectionYes, 
                       data = trainchurnData, 
                       family = binomial(link="logit"))
summary(churnModelTrain)



#build GLM model with a few hand selected variables
churnModelTrain <- glm(Churn ~ 
                         InternetServiceFiberoptic
                       + InternetServiceDSL
                       + PaperlessBillingNo
                       + ContractMonthtomonth
                       + PhoneServiceNo, 
                         data = trainchurnData, 
                         family = binomial(link="logit"))
summary(churnModelTrain)

predDat <- trainchurnData[c("InternetServiceFiberoptic",
                            "InternetServiceDSL",
                            "PaperlessBillingNo",
                            "ContractMonthtomonth",
                            "PhoneServiceNo")]

preds <- data.frame(churnPreds = predict(churnModelTrain, predDat, type = 'response'))
predDataFinal <- cbind(trainchurnData[c("InternetServiceFiberoptic",
                                        "InternetServiceDSL",
                                        "PaperlessBillingNo",
                                        "ContractMonthtomonth",
                                        "PhoneServiceNo",
                                        "Churn")],
                       preds)

rocObj <- roc(predDataFinal$Churn, predDataFinal$churnPreds)
auc(rocObj)





#build GLM model with a few hand selected variables
churnModelTrain <- glm(Churn ~ 
                         tenure + 
                         SeniorCitizen + 
                         PhoneOnlyYes +
                         PaymentMethodElectroniccheck +
                         PaperlessBillingYes +
                         ContractMonthtomonth +
                         MonthlyCharges +
                         TotalCharges +
                         
                         PartnerYes +
                         DependentsYes +
                         MultipleLinesYes +
                         InternetServiceFiberoptic +
                         OnlineSecurityYes +
                         OnlineBackupYes +
                         DeviceProtectionYes +
                         TechSupportYes +
                         StreamingTVYes +
                         PaperlessBillingYes +
                         PaymentMethodBanktransferautomatic +
                         PaymentMethodCreditcardautomatic, 
                       data = trainchurnData, 
                       family = binomial(link="logit"))
summary(churnModelTrain)

predDat <- trainchurnData[c("tenure",
                            "SeniorCitizen",
                            "PhoneOnlyYes",
                            "PaymentMethodElectroniccheck",
                            "PaperlessBillingYes",
                            "ContractMonthtomonth",
                            "MonthlyCharges",
                            "TotalCharges",
                            "PartnerYes",
                            "DependentsYes",
                            "MultipleLinesYes",
                            "InternetServiceFiberoptic",
                            "OnlineSecurityYes",
                            "OnlineBackupYes",
                            "DeviceProtectionYes",
                            "TechSupportYes",
                            "StreamingTVYes",
                            "PaperlessBillingYes",
                            "PaymentMethodBanktransferautomatic",
                            "PaymentMethodCreditcardautomatic")]

preds <- data.frame(churnPreds = predict(churnModelTrain, predDat, type = 'response'))
predDataFinal <- cbind(trainchurnData[c("tenure",
                                        "SeniorCitizen",
                                        "PhoneOnlyYes",
                                        "PaymentMethodElectroniccheck",
                                        "PaperlessBillingYes",
                                        "ContractMonthtomonth",
                                        "MonthlyCharges",
                                        "TotalCharges",
                                        "PartnerYes",
                                        "DependentsYes",
                                        "MultipleLinesYes",
                                        "InternetServiceFiberoptic",
                                        "OnlineSecurityYes",
                                        "OnlineBackupYes",
                                        "DeviceProtectionYes",
                                        "TechSupportYes",
                                        "StreamingTVYes",
                                        "PaperlessBillingYes",
                                        "PaymentMethodBanktransferautomatic",
                                        "PaymentMethodCreditcardautomatic",
                                        "Churn")],
                       preds)

rocObj <- roc(predDataFinal$Churn, predDataFinal$churnPreds)
auc(rocObj)




coords(rocObj, "b", ret="t", best.method="youden") # default

testData <- testchurnData[c("tenure",
                            "SeniorCitizen",
                            "PhoneOnlyYes",
                            "PaymentMethodElectroniccheck",
                            "PaperlessBillingYes",
                            "ContractOneyear",
                            "ContractTwoyear",
                            "MonthlyCharges",
                            "TotalCharges")]

predsTest <- data.frame(churnPreds = predict(churnModelTrain, testData, type = 'response'))
predTestDataFinal <- cbind(testchurnData[c("tenure",
                                        "SeniorCitizen",
                                        "PhoneOnlyYes",
                                        "PaymentMethodElectroniccheck",
                                        "PaperlessBillingYes",
                                        "ContractOneyear",
                                        "ContractTwoyear",
                                        "MonthlyCharges",
                                        "TotalCharges",
                                        "Churn")],
                           predsTest)

rocObjTest <- roc(predTestDataFinal$Churn, predTestDataFinal$churnPreds)
auc(rocObjTest)





































