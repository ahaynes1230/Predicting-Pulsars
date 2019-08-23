# Predicting-Pulsars
Use of R to test several models on predicting a pulsar star


#Get & Set Directory 
getwd()
setwd("/Users/ashanihaynes")

#Load Dataset 
list.files()
pulsar <- read.csv('pulsar_stars.csv')

#Clean Data
is.na(pulsar) #no missing values
#change column names
colnames(pulsar) = c('mean.IP', 'std.IP', 'exk.IP', 'ske.IP', 'mean.DM', 'std.DM',
                     'exk.DM', 'ske.DM', 'target')


## EXPLORATORY ANALYSIS ##
head(pulsar)
summary(pulsar)

#Looking at data structure 
install.packages("DataExplorer")
library(DataExplorer)
plot_intro(pulsar)
plot_bar(pulsar)

#Correlation matrix
install.packages("corrplot")
library(corrplot)
pulsar.cor <- cor(pulsar)
View(pulsar.cor)
#visualization
corrplot(pulsar.cor, method = "circle")
corrplot(pulsar.cor, method = "number")

#Setting correct variable class
pulsar['target'] = as.factor(pulsar$target)

#Distribution: pulsars and non pulsars
install.packages("ggplot2")
library(ggplot2)
plot1 <- ggplot(pulsar, aes(x = target, fill = target)) + geom_bar()
plot1

mean.IP <- ggplot(pulsar, aes(x = target, y = mean.IP, color = target)) + geom_boxplot()
mean.IP

std.IP <- ggplot(pulsar, aes(x = target, y = std.IP, color = target)) + geom_boxplot()
std.IP

exk.IP <- ggplot(pulsar, aes(x = target, y = exk.IP, color = target)) + geom_boxplot()
exk.IP

ske.IP <- ggplot(pulsar, aes(x = target, y = ske.IP, color = target)) + geom_boxplot()
ske.IP

mean.DM <- ggplot(pulsar, aes(x = target, y = mean.DM, color = target)) + geom_boxplot()
mean.DM

std.DM <-ggplot(pulsar, aes(x = target, y = std.DM, color = target)) + geom_boxplot()
std.DM

exk.DM <- ggplot(pulsar, aes(x = target, y = exk.DM, color = target)) + geom_boxplot()
exk.DM

ske.DM <- ggplot(pulsar, aes(x = target, y = ske.DM, color = target)) + geom_boxplot()
ske.DM

# Prep Data #
set.seed(1)

#prep for normalize, accuracy, precision & recall function 
normalize <- function(x) {(x -min(x))/ (max(x) -min(x))}
accuracy <- function(x) {sum(diag(x)/sum(rowSums(x))) * 100}
precision <- function(x) {x[2,2]/(x[2,2]+ x[1,2])}
recall <- function(x) {x[2,2]/(x[2,2] + x[2,1])}
f1.score <- function(x) {(2*precision(x)*recall(x)) / sum(precision(x), recall(x))*100}

#Normalize
target.class <- pulsar$target
pulsar <- as.data.frame(lapply(pulsar[,c(1:8)], normalize))
pulsar <- cbind(pulsar, target.class)
pulsar$target.class <- as.factor(pulsar$target.class)

summary(pulsar)
head(pulsar)

#Split into test & train set // Create index
index <- sample(1:nrow(pulsar), round(0.3*nrow(pulsar)))
train.p <- pulsar[index, ]
test.p <- pulsar[index, ]


## MODELS ##

# LOGISTIC REGRESSION #

install.packages("forecast")
library(forecast)

#model
lrm <- glm(target ~., family = binomial, data = p.train)
summary(lrm)

#test accuracy
prediction.prob <- predict(lrm, p.test, type = 'response')
pred <- numeric(nrow(p.test))
pred[prediction.prob > 0.5] = 1
pred[prediction.prob <= 0.5] = 0
table <- table(as.numeric(p.test$target), as.numeric(pred))

#accuracy rate
accuracy.log <- accuracy(table)
#precision
precise.log <- precision(table)
#recall 
recall.log <- recall(table)
#f1 score
f1.log <- f1.score(table)
c('Accuracy:', accuracy.log, 'F1-Score:', f1.log)

#graph of accuracy w/variables
install.packages("GGally")
library(GGally)
install.packages("broom")
library(broom)
ggcoef(lrm)
#exk.Ip has greatest impact

#Taking deeper look into what excess kurtosis is and what it may represent 
hist(pulsar$exk.IP, freq = F, main = "Excess Kurtosis", col = 'lightgoldenrod')

## RANDOM FOREST ##

install.packages("randomForest")
library(randomForest)

#model 
rf <- randomForest(target.class~., data = pulsar)
#accuracy 
accuracy.rf <- accuracy(rf$confusion)
#precision
precise.rf <- precision(rf$confusion)
#recall
recall.rf <- recall(rf$confusion)
#f1 score
f1.rf <- f1.score(table)
#table of accuracy for random forest
c('Accuracy of Random Forest:', accuracy.rf, 'Accuracy of Logistic Model:', accuracy.log)

#talk about meaning of MeanDecreaseGini chart
varImpPlot(rf, main = "Random Forest")

#because we are noticing the same results that were seen in the linear 
#regression model, i am going to try and better fit the random forest model
plot(rf$err.rate[, 1], type = "l", xlab = "Number of Trees", ylab = "Error OOB")

#steadiness at about 300 trees
rf.300 <- randomForest(target.class~., data = pulsar, ntrees = 300)
#accuracy rate
accuracy.rf.300 <- accuracy(rf.300$confusion)
#precision
precise.rf.300 <- precision(rf.300$confusion)
#recall
recall.rf.300 <- recall(rf.300$confusion)
#f1 score
f1.rf.300 <- f1.score(rf.300$confusion)
#table
c('Accuracy of Random Forest w/ 300 Trees:', accuracy.rf.300, 'Accuracy of Random Forest', accuracy.rf)

#400 trees
rf.400 <- randomForest(target.class~., data = pulsar, ntrees = 400)
#accuracy rate
accuracy.rf.400 <- accuracy(rf.400$confusion)
#precision 
precise.rf.400 <- precision(rf.400$confusion)
#recall
recall.rf.400 <- recall(rf.400$confusion)
#f1 score
f1.rf.400 <- f1.score(rf.400$confusion)

c('Accuracy of Random Forest w/ 400 Trees:', accuracy.rf.400, 'Accuracy of Random Forest w/ 300 Trees:', accuracy.rf.300)


## KNN ##

install.packages("class")
library(class)
install.packages("stringr")
library(stringr)

set.seed(10)

#model
knn.pred <- knn(train = train.p[,-9],
                test = test.p[,-9],
                cl = train.p$target.class,
                k = 11, prob = TRUE)
#confusion matrix of predictions (pulsar/non pulsar)
KNN.CM <- table(pred = knn.pred, true = test.p$target.class)
KNN.CM

#accuracy 
accuracy.knn <- accuracy(table)
#precision
precise.knn <- precision(table)
#recall
recall.knn <- recall(table)
#f1 score
f1.knn <- f1.score(table)

c('Accuracy:', accuracy.knn, 'F1 Score:', f1.knn)

## SUPPORT VECTOR MACHINE ##

install.packages("e1071")
library(e1071)

#model - linear
svmfit <- svm(factor(target.class)~., data = train.p, scale = FALSE, kernel = "linear", cost = 5)
#prediction 
predict.sv <- predict(svmfit, test.p)
#table
table.svm <- table(as.numeric(test.p$target.class), as.numeric(predict.sv))
#accuracy 
accuracy.svm.linear <- accuracy(table.svm)

c('Linear:', accuracy.svm.linear) 

#looking for best cost 
Y <- NULL
k <- 30
for(i in 1:k) {
  svmfit = svm(factor(target.class)~., data = train.p, scale = FALSE, kernel = "linear", cost = i)
  pred <- predict(svmfit, test.p)
  table <- table(as.numeric(test.p$target.class), as.numeric(pred))
  Y[i] <- accuracy(table)
}
n = str_c("cost=", as.character(which.max(Y)))

#plotting accuracy in function of cost 
plot(x = 1:k, y = Y, main = "Accuracy in Function of Cost", xlab = "Cost", ylab = "Accuracy in Percentage %")
abline(v = which.max(Y), lty = 2, lwd = 1)
text(x = which.max(Y)-2, y=min(Y), labels = n)

#model with best kernel and best cost 
svmfit <- svm(factor(target.class)~., data = train.p, scale = FALSE, kernel = "linear", cost = 22)
pred <- predict(svmfit, test.p)
table <- table(as.numeric(test.p$target.class), as.numeric(pred))
#accuracy 
accuracy.SVM <- accuracy(table)
#precision 
precise.SVM <- precision(table)
#recall 
recall.SVM <- recall(table)
#f1 score
f1.SVM <- f1.score(table)

c('Accuracy:', accuracy.SVM, 'F1 Score:', f1.SVM)


## MODEL COMPARISONS ##

install.packages("tidyverse")
library(tidyverse)


df <- data.frame(Names = c("Logistic Regression", "Random Forest", "Random Forest-300 trees", 
                           "Random Forest-400 trees", "KNN", "SVM"), 
                 Accuracy = c(accuracy.log, accuracy.rf, accuracy.rf.300, accuracy.rf.400, accuracy.knn, accuracy.SVM),
                 F1_Score = c(f1.log, f1.rf, f1.rf.300, f1.rf.400, f1.knn, f1.SVM),
                 Precision = c(precise.log*100, precise.rf*100, precise.rf.300*100, precise.rf.400*100, precise.knn*100, precise.SVM*100),
                 Recall = c(recall.log*100, recall.rf*100, recall.rf.300*100, recall.rf.400*100, recall.knn*100, recall.SVM*100))

df = df %>%
  arrange(Accuracy) %>%
  mutate(Names=factor(Names, Names))

ggplot(data = df, aes(x = Names)) +
  geom_point(aes(y = F1_Score), color = "green", size = 2) +
  annotate("text", x = 6, y = 88, label = "F1_Score", color = "green") +
  geom_point(aes(y = Accuracy), color = "red", size = 2) +
  annotate("text", x = 5, y = 96, label = "Accuracy", color = "red") + 
  geom_point(aes(y = Precision), color = "purple", size = 2) +
  annotate("text", x = 3, y = 92, label = "Precision", color = "purple") +
  geom_point(aes(y = Recall), color = "blue", size = 2) +
  annotate("text", x = 1, y = 82.5, label = "Recall", color = "blue") +
  coord_flip() + 
  xlab("Models") +
  ylab("Performances")

df






