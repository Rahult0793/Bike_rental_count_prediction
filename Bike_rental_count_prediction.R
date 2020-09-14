#Loading Libraries

x = c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "unbalanced", "C50", "dummies", "e1071", "Information",
      "MASS", "rpart", "gbm", "ROSE", 'sampling', 'DataCombine', 'inTrees')

install.packages(x)
lapply(x, require, character.only = TRUE)

# Importing the dataset
dataset = read.csv('data.csv')

#Checking for missing Data
sum(is.na(dataset))


#Replacing all outliers in "windspeed" and "casual" variables
#with NA and imputing the NA values
outliers <-data.frame(dataset$windspeed, dataset$casual)

library("VIM")

numeric_index = sapply(outliers,is.numeric) #selecting only numeric
numeric_data = outliers[,numeric_index]
cnames = colnames(numeric_data)

for(i in cnames){
  val = outliers[,i][outliers[,i] %in% boxplot.stats(outliers[,i])$out]
  print(length(val))
  outliers[,i][outliers[,i] %in% val] = NA
}

cleaned <- kNN(outliers, k = 5)
dataset$windspeed <- cleaned$dataset.windspeed
dataset$casual <- cleaned$dataset.casual

summary(dataset)
dataset$season <- as.factor(dataset$season)
dataset$yr <- as.factor(dataset$yr)
dataset$mnth <- as.factor(dataset$mnth)
dataset$holiday <- as.factor(dataset$holiday)
dataset$weekday <- as.factor(dataset$weekday)
dataset$workingday <- as.factor(dataset$workingday)
dataset$weathersit <- as.factor(dataset$weathersit)


###################################################################

## Correlation Plot 
corrgram(dataset[,numeric_index], order = F,
         upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot")

#####################################################################

#Normalization : 
cnames = c("casual", "registered")

for(i in cnames){
  print(i)
  dataset[,i] = (dataset[,i] - min(dataset[,i]))/
    (max(dataset[,i] - min(dataset[,i])))
}

dataset <- subset(dataset, select = -c(dteday, instant))

##############################################################
#Using Random Forest for Feature selection : 

imppred <- randomForest(cnt ~ ., data = dataset,
                        ntree = 100, keep.forest = FALSE, importance = TRUE)
importance(imppred, type = 1)

symnum(cor(dataset))

#################Splitting data into test and Train#############################

require(caTools)
set.seed(101) 
sample = sample.split(dataset$cnt, SplitRatio = .75)
train = subset(dataset, sample == TRUE)
test  = subset(dataset, sample == FALSE)


#####################################################################
#Linear Regression : 

lrmodel <- lm(cnt ~ ., 
              data = train)
summary(lrmodel)

y_pred_lm = predict(lrmodel, newdata = test[,-14])

##################################################################
# Decision Tree regression : 

fit = rpart(cnt ~ ., data = train, method = "anova")
summary(fit)

#Predict for new test cases
y_pred_dt = predict(fit, test[,-14])


###########################################################
#Random Forest Regression

library(randomForest)
set.seed(1234)
regressor = randomForest(x = train[,-14],
                         y = train$cnt,
                         ntree = 800)

# Predicting a new result with Random Forest Regression
y_pred_rf = predict(regressor, test[,-14])

############################################################
#SVM Regression

library(e1071)
regressor = svm(formula = cnt ~ .,
                data = train,
                type = 'eps-regression',
                kernel = 'linear')

summary(regressor)
# Predicting a new result
y_pred_svm = predict(regressor, test[,-14])

###########################################################
# Performance measurement

predicted <- y_pred_lm
actual <- test[,14] 

rmse_linear <- (mean((y_pred_lm - actual)^2))^0.5
MAE_linear <- mean(abs(y_pred_lm - actual))

rmse_tree <- (mean((y_pred_dt - actual)^2))^0.5
MAE_tree <- mean(abs(y_pred_dt - actual))

rmse_rf <- (mean((y_pred_rf - actual)^2))^0.5
MAE_rf <- mean(abs(y_pred_rf - actual))


rmse_svm <- (mean((y_pred_svm - actual)^2))^0.5
MAE_svm <- mean(abs(y_pred_svm - actual))

#Linear model
rmse_linear
MAE_linear

#Decisison_tree_Regression
rmse_tree
MAE_tree

#Random Forest regression
rmse_rf
MAE_rf

#SVM model
rmse_svm
MAE_svm


#############################################################
#The Input and Output

Input <- test[,-14]
test$predicted_count <- y_pred_svm
output <- test
head(output)
write.csv(Input,"Input.csv")
write.csv(output,"Output.csv")
