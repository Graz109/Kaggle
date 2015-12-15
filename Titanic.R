library(ggplot2)
library(DAAG) #cv.lm
library(caret) #create train data
library(boot) #bootstrapping
library(MASS)
library(class) #knn.cv
library(data.table)
library(stringr)
library(randomForest)
library(lattice)

set.seed(100)

cv.lm <- function(response_var, reg_formula, dataset, cv_folds = 10, weights = NULL){
  #store all mse from each model
  mse <- c()
  rmse <- c()
  mape <- c()
  
  #select rows to use fro prediction in each fold
  folds <- list()
  
  while(length(folds) != cv_folds){
    #In rare instances createFolds will produce less folds than desired 
    folds <- createFolds(dataset[,1], k = cv_folds, list = TRUE, returnTrain = FALSE)
  }
  
  for (i in 1:cv_folds) {
    #The amount of data used for training in each fold is equal to nrow(dataset)/cv_folds
    train_data_sample <- dataset[-(unlist(folds[i])),] #fold build data
    
    if (is.null(weights) == FALSE)
      weights_sample <- weights[-(unlist(folds[i]))]
    
    test_data <- dataset[(unlist(folds[i])),] #cross validation predicting data
    test_data_actual <- test_data[names(test_data) %in% response_var] #actual values to calculate MSE
    test_data <- test_data[!(names(test_data) %in% test_data)] #remove response from predicting data
    
    if (is.null(weights))
      lm <- lm(as.formula(reg_formula), data = train_data_sample)
    else
      lm <- lm(as.formula(reg_formula), data = train_data_sample, weights = weights_sample)
    
    pred.lm <- predict(lm,  data.frame(test_data))
    
    #MSE
    mse[i] <- mean((as.numeric(test_data_actual[,1])-pred.lm)^2)
    #RMSE
    rmse[i] <- sqrt(mse[i])
    #MAPE
    abs_residuals_div_act <- abs((test_data_actual[,1]-pred.lm)/test_data_actual[,1])
    mape[i] = sum(abs_residuals_div_act)/length(pred.lm)
    
  }
  
  #average cross validated errors from all folds
  cv.mse <- prettyNum(mean(mse) , big.mark = ",", format = "g")
  cv.rmse <- prettyNum(mean(rmse), big.mark = ",", format = "g")
  cv.mape <- prettyNum(mean(mape), big.mark = ",", format = "g")
  
  
  return(c(cv.rmse, cv.mape))
  
}

rmse <- function(observed, predicted) {
  resid_squ = (observed - predicted)^2
  mean_resid_squ = mean(resid_squ)
  rmse <- sqrt(mean_resid_squ)
  return(rmse)
}

classification_error <- function(observed, predicted){
  predicted <- ifelse(predicted>.50, 1, 0)
  diff = (observed == predicted)
  diff_count = length(diff[diff==FALSE])
  error_rate = diff_count/length(diff)
  return(error_rate)
   
}






#Read data
train <- read.csv("C:/Users/grazim/Desktop/Kaggle/Titanic/train.csv")
test <- read.csv("C:/Users/grazim/Desktop/Kaggle/Titanic/test.csv")
#Convert relevant character columns to factors
test$Pclass <- as.factor(test$Pclass)
test$Survived <- as.factor(test$Survived)
#Convert relevant character columns to factors
train$Pclass <- as.factor(train$Pclass)
train$Survived <- as.factor(train$Survived)

male <- subset(train, Sex == "male")
female <- subset(train, Sex == "female")

View(train)

# VARIABLE DESCRIPTIONS:
#   survival        Survival
# (0 = No; 1 = Yes)
# pclass          Passenger Class
# (1 = 1st; 2 = 2nd; 3 = 3rd)
# name            Name
# sex             Sex
# age             Age
# sibsp           Number of Siblings/Spouses Aboard
# parch           Number of Parents/Children Aboard
# ticket          Ticket Number
# fare            Passenger Fare
# cabin           Cabin
# embarked        Port of Embarkation
# (C = Cherbourg; Q = Queenstown; S = Southampton)
# 
# SPECIAL NOTES:
#   Pclass is a proxy for socio-economic status (SES)
# 1st ~ Upper; 2nd ~ Middle; 3rd ~ Lower
# 
# Age is in Years; Fractional if Age less than One (1)
# If the Age is Estimated, it is in the form xx.5
# 
# With respect to the family relation variables (i.e. sibsp and parch)
# some relations were ignored.  The following are the definitions used
# for sibsp and parch.
# 
# Sibling:  Brother, Sister, Stepbrother, or Stepsister of Passenger Aboard Titanic
# Spouse:   Husband or Wife of Passenger Aboard Titanic (Mistresses and Fiances Ignored)
# Parent:   Mother or Father of Passenger Aboard Titanic
# Child:    Son, Daughter, Stepson, or Stepdaughter of Passenger Aboard Titanic
# 
# Other family relatives excluded from this study include cousins,
# nephews/nieces, aunts/uncles, and in-laws.  Some children travelled
# only with a nanny, therefore parch=0 for them.  As well, some
# travelled with very close friends or neighbors in a village, however,
# the definitions do not support such relations.\\

##########################################
#Attempt to create feature called "title"#
##########################################
train$Name <- as.character(train$Name)
#Break each string up into a 3 part list. The second element is the title of the person
names_title <- sapply(train$Name , FUN=function(x) {str_trim(strsplit(x, split='[,.]')[[1]][2])})
train$title <- names_title

table(train$title)

#Combine the less prevelant titles into more frequent ones

#Keepers:
#-"Mr"
#-"Master"
#-"Mrs"
#-"Miss"
#-"Rev"
#-"Dr"

#Others:
#-"Capt" --> "Mr"
#-"Col"
#-"Don" --> "Status" M
#-"Jonkheer" -->  "Status" M/F
#-"Lady" --> "Status" F
#-"Major" -- "Mr"
#-"Mlle" --> "Miss"
#-"Mme" --> "Mrs" 
#-"Ms" -->    "Miss"
#-"Sir" --> "Status" M
#-"the Countess" --> "Status" F

#New Titles
#-"Status"
train$title = ifelse(train$title == "Capt", "Mr", train$title)
train$title = ifelse(train$title == "Col", "Mr", train$title)
train$title = ifelse(train$title == "Don", "Mr", train$title)
train$title = ifelse(train$title == "Jonkheer", "Mr", train$title)
train$title = ifelse(train$title == "Lady", "Mrs", train$title)
train$title = ifelse(train$title == "Major", "Mr", train$title)
train$title = ifelse(train$title == "Mlle", "Miss", train$title)
train$title = ifelse(train$title == "Mme", "Mrs", train$title)
train$title = ifelse(train$title == "Ms", "Miss", train$title)
train$title = ifelse(train$title == "Sir", "Mr", train$title)
train$title = ifelse(train$title == "the Countess", "Miss", train$title)

table(train$title)



#Create Title for Test data
test$Name <- as.character(test$Name)
#Break each string up into a 3 part list. The second element is the title of the person
names_title <- sapply(test$Name , FUN=function(x) {str_trim(strsplit(x, split='[,.]')[[1]][2])})
test$title <- names_title

table(test$title)

test$title = ifelse(test$title == "Col", "Mr", test$title)
test$title = ifelse(test$title == "Dona", "Mrs", test$title)
test$title = ifelse(test$title == "Ms", "Miss", test$title)
test$title = ifelse(test$title == "Sir", "Mr", test$title)
test$title = ifelse(test$title == "the Countess", "Miss", test$title)



####################
###MISSING VALUES###
####################
#Visualize the age of respondents
hist(train$Age,30, xlim = c(0, 85), ylim = c(0, 50))
#by gender
hist(male$Age, 30, xlim = c(0, 85), ylim = c(0, 50))
hist(female$Age, 30, xlim = c(0, 85), ylim = c(0, 50))

#There were a lot more males on board than therer were females


mean_age <- mean(train$Age, na.rm= TRUE)
mean_age_male <- mean(male$Age, na.rm= TRUE)
mean_age_female <- mean(female$Age, na.rm= TRUE)

#Test if the mean age of men is significantly different than the mean of women
t.test(male$Age, female$Age)
#There is a significant difference between the ages of the two population

#Look at the data by class
counts <- table(train$Pclass, train$Age)
barplot(counts, col = c("darkblue", "red", "green"))
#DB - 1
#red - 2
#green - 3

#Most on board were middle aged
#Unexpected amount of children


#Count1 of class
counts1 <- table(train$Age, train$Pclass)
barplot(counts1)
#There is a disproportionate of lower class individual son nteh boat than there are upper and middle class

#Find how many people survived
View(table(train$Survived))
#More died than survived

#Attempt to use linear regression to predict the age of the respondents
age_train <- train[!is.na(train$Age),]
na.age <- train[is.na(train$Age),]
lm_age <- lm(Age ~ Pclass + Sex + SibSp + Parch + Fare + title, data = age_train)
par(mfrow= c(2,2)) # display plots in a grid
plot(lm_age)
##appears to be heterskestasisity in the model
#take the log of age
lm_log_age <- lm(log(Age) ~ Pclass + Sex + SibSp + Parch + Fare + title, data = age_train)
lm_log_age1 <- lm(log(Age) ~ Pclass + Sex + SibSp + Parch + title, data = age_train) #Fare is very insignificant
lm_log_age1 <- lm(log(Age) ~ Pclass + SibSp + Parch + title, data = age_train) #Sex is very insignificant

#Miss and Master appear to be the only title that is significant titles.
#--> Likely due to those titles being reserved for younger kids.  I suspect there is less of a range of master/miss than for ohter titles
range_master <- range(age_train$Age[age_train$title == "Master"])
hist(age_train$Age[age_train$title == "Master"])
range_miss <- range(age_train$Age[age_train$title == "Miss"])
hist(age_train$Age[age_train$title == "Miss"])
range_Mr <- range(age_train$Age[age_train$title == "Mr"])
hist(age_train$Age[age_train$title == "Mr"])
range_Mrs <- range(age_train$Age[age_train$title == "Mrs"])
hist(age_train$Age[age_train$title == "Mrs"])


#Create a master/miss dummy column to use and drop other titles
title_dummy <- model.matrix(~title+0, age_train)
age_train <- cbind(age_train, title_dummy)

title_dummy_na <- model.matrix(~title+0, na.age)
na.age <- cbind(na.age, title_dummy_na)

title_dummy_test <- model.matrix(~title+0, test)
test <- cbind(test, title_dummy_test)

# age_train$SibSp <- as.factor(age_train$SibSp)
# na.age$SibSp <- as.factor(na.age$SibSp)

lm_log_age3 <- lm(log(Age) ~ Pclass + SibSp + Parch + titleMr + titleMiss, data = age_train)
cv_lm3_error <- exp(as.numeric(cv.lm('Age', 'log(Age) ~ Pclass + SibSp + Parch + titleMr + titleMiss', age_train, cv_folds = 10)))[2]

lm_age3_nolog <- lm(Age ~ Pclass + SibSp + Parch + titleMr + titleMiss, data = age_train)
cv_lm3_nolog_error <- as.numeric(cv.lm('Age', 'Age ~ Pclass + SibSp + Parch + titleMr + titleMiss', age_train, cv_folds = 10))[2]

#Normailty clearly doesn't hold.  Bootstrab to estimate the actual standard errors of predictors
#Sample n observations with replacement B times.  Calculate the estimates in each iteration.  Calculate the mean value of Beta values as well as SE
coef.fn <- function(data, index) {
  lm_log_age3 <- lm(log(Age) ~ Pclass + SibSp + Parch + titleMr + titleMiss, data = age_train)
  coef <- coefficients(lm_log_age1)
  return(coef)
}

#Bootstrapping the coef resulted in slightly higher std. error than summary() (likely due to lack of normality assumpting being met)
boot.se <- boot(data = age_train, coef.fn, R=1000)

#use model to predict the missing ages
na.age.pred <- predict(lm_age3_nolog, na.age)
na.age.pred <- exp(na.age.pred)

#Replace NA, round to nearest integer
na.age$Age <- na.age.pred

#If age is less than 0, default to the minimum age in the dataset
min_age = min(na.omit(train$Age))

na.age$Age[na.age$Age < 0] <- min_age

#Add "titleRev" to na.age for proper merging
titleRev <- rep(0, nrow(na.age))
na.age$titleRev <- titleRev

train_corrected_age <- rbind(age_train, na.age)

#Attempt to run a logistic regression model to predict survival
logit <- glm(formula = Survived ~ Age + Pclass + Sex + SibSp + Parch + Fare, family = binomial(),data = train_corrected_age)
logit1 <- glm(formula = Survived ~ Age + Pclass + Sex + SibSp + Fare, family = binomial(), data = train_corrected_age)
logit2 <- glm(formula = Survived ~ Age + Pclass + Sex + SibSp, family = binomial(),data = age_train)

#Use cross validation to test prediction accuracy
logit2.cv <- cv.glm(data = age_train, logit2, cost = classification_error, K = 10)

#Add title to the logit model to see if improvement are had
logit3 <- glm(formula = Survived ~ Age + Pclass + Sex + SibSp + titleMaster + titleMiss, family = binomial(),data = age_train)
logit3.cv <- cv.glm(data = age_train, logit3, cost = classification_error, K = 10)


#Predict missing ages in test data using previous model
age_test <- test[!is.na(test$Age),]
na.age.test <- test[is.na(test$Age),]
na.age.pred.test <- predict(lm_age3_nolog, na.age.test)
na.age.pred.test <- exp(na.age.pred.test)

na.age.test$Age <- na.age.pred.test

age_test <- rbind(age_test, na.age.test)

#predict test data
predict.logit.test_prob <-predict(logit3, type = "response", newdata = age_test)
predict.logit.test.50 <- ifelse(predict.logit.test_prob>.50, 1, 0)

#Append PassengerId onto the prediction
output <- cbind(age_test["PassengerId"], predict.logit.test.50)

write.csv(output, "C:/Users/grazim/desktop/kaggle/titanic/logit_predictions.csv", row.names = FALSE)
#.746 success rate


train_corrected_age$title <- as.factor(train_corrected_age$title)
#Attempt a random forest algorithm
RF <- randomForest(Survived ~ Pclass + Sex + Age + Parch + SibSp + title, data = train_corrected_age, importance = TRUE, ntree = 5000)
age_test$title <- as.factor(age_test$title)
predict.RF <- predict(RF, newdata = age_test, type = "response")

#Append PassengerId onto the prediction
output_RF <- cbind(age_test["PassengerId"], as.factor(predict.RF))

write.csv(output_RF, "C:/Users/grazim/desktop/kaggle/titanic/RF_predictions.csv", row.names = FALSE)
#.76555 success rate


#Attempt KNN algorithm


rows <- 1:nrow(train_corrected_age)

#create list to store each fold's error
error <- data.frame(matrix(nrow = 100, ncol = 3))
colnames(error) <- c("fold", "num_neb", "success_rate")

trainIndex <- createMultiFolds(train_corrected_age$Survived, times = 100)
#Run knn with 100 folds
for(i in 1:100){
  selected_rows <- as.numeric(unlist(trainIndex[i]))
  train <- train_corrected_age[selected_rows,]
  train <- train[c("Survived", "Pclass", "Sex", "Age", "SibSp", "Parch")]
  
  test <- train_corrected_age[!(rows %in% selected_rows),]
  true <- test["Survived"]
  test <- test[c("Pclass", "Sex", "Age", "SibSp", "Parch")]
  
  #For each model, run 1:50 NNs
  for(j in 1:50){

    KNN <- knn3(Survived ~ Pclass + Sex + Age + Parch + SibSp, data = train_corrected_age, k=j)
    KNN_Pred <- predict(KNN, newdata = test, type = "class")
    confus <- confusionMatrix(data = KNN_Pred, reference =  true$Survived)
    error_fold <- confus$overall[1]
    error <- rbind(error, c(i, j, error_fold))
  }
}
error <- na.omit(error)

error_dt <- data.table(error)

avg_error <- error_dt[,list(avg_error = mean(success_rate)), by = c("num_neb")]

avg_error <- data.frame(avg_error)
#Find where error is minimized
plot(avg_error[,1], 1-avg_error[,2])

#Predict Test Data
KNN <- knn3(Survived ~ Pclass + Sex + Age + Parch + SibSp, data = train_corrected_age, k=1)
predict_knn_prob <- predict.knn3(KNN, newdata = age_test, type = "prob")[,2] #Return probabilities of being in the sruvived group
predict_knn <- ifelse(predict_knn_prob>.50, 1, 0)

output_knn <- cbind(age_test["PassengerId"], predict_knn)
colnames(output_knn) <- c("PassengerID", "Survived")
write.csv(output_RF, "C:/Users/grazim/desktop/kaggle/titanic/knn1_predictions.csv", row.names = FALSE)
#.76077 success rate 1 KNN
#.76555 success rate 5 KNN




#Average KNN and Logistic Regression probabilities
knn_logit_prob_avg <- rowMeans(cbind(as.numeric(predict_knn_prob), as.numeric(predict.logit.test_prob)))
knn_logit_avg <- ifelse(knn_logit_prob_avg>.5, 1, 0)
output_knn_logit <- cbind(age_test["PassengerId"], knn_logit_avg)
write.csv(output_knn_logit, "C:/Users/grazim/desktop/kaggle/titanic/knn_logit_predictions.csv", row.names = FALSE)
#.75598 success rate




#QDA
QDA <- qda(Survived ~ Pclass + Sex + Age + Parch + SibSp, data = train_corrected_age)
predict_QDA_Prob <- predict(object = QDA, newdata = age_test, method = "predictive")
predict_QDA <- predict_QDA_Prob$class
output_QDA <- cbind(age_test["PassengerId"], predict_QDA)
write.csv(output_QDA, "C:/Users/grazim/desktop/kaggle/titanic/QDA_predictions.csv", row.names = FALSE)
#.71292 success rate








#Look at using "Ticket Number" as a potential feature.  If a person had the same ticket as another, were they more or less likely to survive?
#Basically looking at if singles or familes.couples survive more?

#Count frequency of each ticket number
train_corrected_age_dt <- data.table(train_corrected_age)
ticket_freq <- train_corrected_age_dt[, list(count = .N), by = "Ticket"]
ticket_freq <- ticket_freq[order(count),]
ticket_freq_gt1 <- ticket_freq$Ticket[ticket_freq$count >1]

train_corrected_age$not_alone <- as.factor(ifelse(train_corrected_age$Ticket %in% ticket_freq_gt1, 1, 0))


#Add to the logistic regression model
logit4 <- glm(formula = Survived ~ Age + Pclass + Sex + SibSp + titleMaster + titleMiss + not_alone, family = binomial(),data = train_corrected_age)
logit4.cv <- cv.glm(data = train_corrected_age, logit4, cost = classification_error, K = 10)


#Does a character vs numeric ticket mean someting? 
train_corrected_age$char_ticket <- grepl("^[A-Z]", train_corrected_age$Ticket)
logit5 <- glm(formula = Survived ~ Age + Pclass + Sex + SibSp + titleMaster + titleMiss + char_ticket, family = binomial(),data = train_corrected_age)
logit5.cv <- cv.glm(data = train_corrected_age, logit5, cost = classification_error, K = 10)

#Look more closely into "Embarked"
#Combine C and Q and compare it to 
train_corrected_age$Embarked <- as.character(train_corrected_age$Embarked)
train_corrected_age$Embarked_comb = ifelse(train_corrected_age$Embarked == "S", "S", "QC")

logit6 <- glm(formula = Survived ~ Age + Pclass + Sex + SibSp + titleMaster + titleMiss + Embarked_comb + Embarked_comb, family = binomial(),data = train_corrected_age)
logit6.cv <- cv.glm(data = train_corrected_age, logit6, cost = classification_error, K = 10)

#Create Groups
breaks = c(-1,25,50,75,1000)
fare_cat = cut(train_corrected_age$Fare, breaks = breaks)

train_corrected_age$fare_cat <- fare_cat
logit7 <- glm(formula = Survived ~ Age + Pclass + Sex + SibSp + titleMaster + titleMiss + fare_cat, family = binomial(),data = train_corrected_age)
logit7.cv <- cv.glm(data = train_corrected_age, logit7, cost = classification_error, K = 10)




##Attempt to look at locatoin on boat (cabin)

#Pull the first letter from the cabin.
#For missing data, use "NA" as a catagory

train_corrected_age$cabin_let <- substr(train_corrected_age$Cabin,1,1)
# train_corrected_age$cabin_let <- ifelse(train_corrected_age$cabin_let %in% c("A","B", "C", "D"), "ABCD", train_corrected_age$cabin_let)
# train_corrected_age$cabin_let <- ifelse(train_corrected_age$cabin_let %in% c("C","D"), "CD", train_corrected_age$cabin_let)
# train_corrected_age$cabin_let <- ifelse(train_corrected_age$cabin_let %in% c("E","F", "G"), "EFG", train_corrected_age$cabin_let)
train_corrected_age$cabin_let <- ifelse(train_corrected_age$cabin_let == "", "NA", train_corrected_age$cabin_let)
train_corrected_age$cabin_let <- ifelse(train_corrected_age$cabin_let == "T", "NA", train_corrected_age$cabin_let)
train_corrected_age$cabin_let <- as.factor(train_corrected_age$cabin_let)

age_test$cabin_let <- substr(age_test$Cabin,1,1)
# age_test$cabin_let <- ifelse(age_test$cabin_let %in% c("A","B"), "AB", age_test$cabin_let)
# age_test$cabin_let <- ifelse(age_test$cabin_let %in% c("C","D"), "CD", age_test$cabin_let)
# age_test$cabin_let <- ifelse(age_test$cabin_let %in% c("E","F", "G"), "EFG", age_test$cabin_let)
age_test$cabin_let <- ifelse(age_test$cabin_let == "", "NA", age_test$cabin_let)
age_test$cabin_let <- ifelse(age_test$cabin_let == "T", "NA", age_test$cabin_let)
age_test$cabin_let <- as.factor(age_test$cabin_let)


logit8 <- glm(formula = Survived ~ Age + Pclass + Sex + SibSp + titleMaster + titleMiss + cabin_let, family = binomial(),data = train_corrected_age)
logit8.cv <- cv.glm(data = train_corrected_age, logit8, cost = classification_error, K = 10)


logit9 <- glm(formula = Survived ~ Age + Pclass + SibSp + titleMaster + titleMiss+titleDr + titleRev, family = binomial(),data = age_train)
logit9.cv <- cv.glm(data = age_train, logit3, cost = classification_error, K = 10)





#Explore a relationship between Fare, Cabin_let, and Pclass
#Did the mroe affluent populations stay in speciic cabins
#Can I somehow fill in missing values in Cabin_let using those trends?
#Some cabins have multiple listed. Does this mean that 3 capbins were payed for.Should Fare be divided by the number of cabins?
#NOTE: There may be som VIF with Pclass



#Compare Pclass and Cabin Deck
p <- ggplot(aes(x = Pclass, colour = cabin_let, data = na.omit(train_corrected_age))) + geom_histogram()

train_corrected_age$Pclass_num <- as.numeric(train_corrected_age$Pclass)
hist(train_corrected_age$Pclass_num)
plot(table(train_corrected_age$Pclass))


table(train_corrected_age$Pclass, train_corrected_age$fare_cat)


#Did people of higher status embark from a specific location?
#Can a catagorize where people stayed on the ship based on their status, where they embarked from, fare, etc? 
table(train_corrected_age$Embarked, train_corrected_age$Pclass)

table(train_corrected_age$Embarked, train_corrected_age$cabin_let)



###############################
#Current winning model: logit3#
###############################