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
length(unique(churnData$customerID))

#Use dplyr
churnData %>%
  summarise(nDisCnt = n_distinct(customerID))
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
#InternetService vs Churn
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
#InternetService vs Churn
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
#InternetService vs Churn
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
#InternetService vs Churn
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
#InternetService vs Churn
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
#InternetService vs Churn
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
#InternetService vs Churn
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
#InternetService vs Churn
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
#InternetService vs Churn
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
                                      PaymentMethod %in% c("Bank transfer (automatic)","Credit card (automatic)"), 'Yes', 'No'))


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
#PhoneInternet vs Churn
internetOnlyChurn <- melt(table(churnData$PhoneInternet,churnData$Churn))
colnames(internetOnlyChurn) <- c("PhoneInternet", "Churn", "CumuPerc")
#plot
ggplot(internetOnlyChurn, aes(x=PhoneInternet, y=CumuPerc, fill = Churn)) +
  geom_bar(position="fill", stat="identity") +
  theme_linedraw() #+ if you want to plot with different colors
#scale_fill_brewer(palette = "Greys")
#both phone and internet is more likely to churn
###############################################################


###############################################################
#phoneOnlyChurn vs Churn
internetPhoneChurn <- melt(table(churnData$InternetOnly,churnData$Churn))
colnames(internetOnlyChurn) <- c("InternetOnly", "Churn", "CumuPerc")
#plot
ggplot(internetOnlyChurn, aes(x=InternetOnly, y=CumuPerc, fill = Churn)) +
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


## Create Dummy Variables ##
dmy <- dummyVars(" ~ gender + Partner + Dependents + PhoneService +
                 MultipleLines + InternetService + OnlineSecurity +
                 OnlineBackup + DeviceProtection + TechSupport +
                 StreamingTV + StreamingMovies + Contract + PaperlessBilling +
                 PaymentMethod + PhoneOnly + InternetOnly + PaperlessAutoPay +
                 PhoneInternet", 
                 data = churnData,
                 fullRank = TRUE)
dmyData <- data.frame(predict(dmy, newdata = churnData))
#print(head(dmyData))

#strip the "." out of the column names
colClean <- function(x){ colnames(x) <- gsub("\\.", "", colnames(x)); x } 
dmyData <- colClean(dmyData) 


#lets combine the new dummy variables back with the original continuous variables
churnDataFinal <- cbind(dmyData, churnData[,c(2,5,18,19,20)])


#lets get a traing and test data set using the createPartition function from Caret

set.seed(420)
inTrain <- createDataPartition(churnDataFinal$Churn, p = 4/5, list = FALSE, times = 1)

trainchurnData <- churnDataFinal[inTrain,]
testchurnData <- churnDataFinal[-inTrain,]

prop.table(table(churnData$Churn))
prop.table(table(trainchurnData$Churn))
prop.table(table(testchurnData$Churn))



#build GLM model
churnModelTrain <- glm(Churn ~ ., 
                       data = trainchurnData, 
                       family = binomial(link="logit"))
summary(churnModelTrain)








































