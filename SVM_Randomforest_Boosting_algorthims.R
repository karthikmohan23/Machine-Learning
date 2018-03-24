library(ggplot2)
library(e1071)
library(rpart)
library(rpart.plot)
library(caret)
set.seed(1234)

car_data = read.csv('C:/AML - BUAN 6341/car_data.csv',  header=TRUE)
str(car_data)
qplot(persons,safety, data=car_data, color = car)
car_train = car_data[1:round(0.7*nrow(car_data)),]
car_test = car_data[(round(0.7*nrow(car_data))+1):nrow(car_data),]
car_train_split = list(car_data[1:round(0.2*nrow(car_data)),],car_data[1:round(0.3*nrow(car_data)),],car_data[1:round(0.4*nrow(car_data)),],car_data[1:round(0.5*nrow(car_data)),],car_data[1:round(0.6*nrow(car_data)),],car_data[1:round(0.7*nrow(car_data)),])
car_test_split = list(car_data[round(0.9*nrow(car_data)):nrow(car_data),],car_data[round(0.85*nrow(car_data)):nrow(car_data),],car_data[round(0.80*nrow(car_data)):nrow(car_data),],car_data[round(0.75*nrow(car_data)):nrow(car_data),],car_data[round(0.70*nrow(car_data)):nrow(car_data),],car_data[round(0.65*nrow(car_data)):nrow(car_data),])
x_ax = c(20,30,40,50,60,70)

############################# SVM LINEAR #######################################################

#Learning Curve
err_rates = c()
for (d in car_train_split){
  svm_model = svm(car~.,data = d, kernel = "linear")
  pred1 = predict(svm_model, d)
  tab1 = table(Predicted = pred1, Actual = d$car)
  err_rates = c(err_rates,1-sum(diag(tab1))/sum(tab1))
}

test_err_rates = c()
for (d in car_test_split){
  svm_model = svm(car~.,data = d, kernel = "linear")
  pred1 = predict(svm_model, d)
  tab1 = table(Predicted = pred1, Actual = d$car)
  test_err_rates = c(test_err_rates,1-sum(diag(tab1))/sum(tab1))
}
plot(x_ax, err_rates, ylim=c(0.00,0.1), type="l", col="green", ylab="error rate", xlab="training data size(in %)", main="Learning Curve:SVM-Linear;Train=G;Test=R")
lines(x_ax,test_err_rates,col="red")
svm_model = svm(car~.,data = car_train, kernel = "linear")
summary(svm_model)

#Prediction on Train Data
pred = predict(svm_model, car_train)
tab = table(Predicted = pred, Actual = car_train$car)
tab
train_err = 1-sum(diag(tab))/sum(tab)
sprintf("Kernel: Linear, Train error: %f",train_err)

#Prediction on Test Data
pred = predict(svm_model, car_test)
tab = table(Predicted = pred, Actual = car_test$car)
tab
test_err = 1-sum(diag(tab))/sum(tab)
sprintf("Kernel: Linear, Test error: %f",test_err)


############################# SVM RADIAL #######################################################

#Learning Curve
err_rates = c()
for (d in car_train_split){
  svm_model = svm(car~.,data = d, kernel = "radial")
  pred1 = predict(svm_model, d)
  tab1 = table(Predicted = pred1, Actual = d$car)
  err_rates = c(err_rates,1-sum(diag(tab1))/sum(tab1))
}
test_err_rates = c()
for (d in car_test_split){
  svm_model = svm(car~.,data = d, kernel = "radial")
  pred1 = predict(svm_model, d)
  tab1 = table(Predicted = pred1, Actual = d$car)
  test_err_rates = c(test_err_rates,1-sum(diag(tab1))/sum(tab1))
}
plot(x_ax, err_rates, ylim=c(0.00,0.12), type="l", col="green", ylab="error rate", xlab="training data size(in %)", main="Learning Curve:SVM-Radial;Train=G;Test=R")
lines(x_ax,test_err_rates,col="red")
svm_model = svm(car~.,data = car_train, kernel = "radial")
summary(svm_model)

#Prediction on Train Data
pred = predict(svm_model, car_train)
tab = table(Predicted = pred, Actual = car_train$car)
tab
train_err = 1-sum(diag(tab))/sum(tab)
sprintf("Kernel: Radial, Train error: %f",train_err)

#Prediction on Test Data
pred = predict(svm_model, car_test)
tab = table(Predicted = pred, Actual = car_test$car)
tab
test_err = 1-sum(diag(tab))/sum(tab)
sprintf("Kernel: Radial, Test error: %f",test_err)

############################# SVM POLYNOMIAL #######################################################

#Learning Curve
err_rates = c()
for (d in car_train_split){
  svm_model = svm(car~.,data = d, kernel = "polynomial")
  pred1 = predict(svm_model, d)
  tab1 = table(Predicted = pred1, Actual = d$car)
  err_rates = c(err_rates,1-sum(diag(tab1))/sum(tab1))
}
test_err_rates = c()
for (d in car_test_split){
  svm_model = svm(car~.,data = d, kernel = "polynomial")
  pred1 = predict(svm_model, d)
  tab1 = table(Predicted = pred1, Actual = d$car)
  test_err_rates = c(test_err_rates,1-sum(diag(tab1))/sum(tab1))
}
plot(x_ax, err_rates, ylim=c(0.10,0.50), type="l", col="green", ylab="error rate", xlab="training data size(in %)", main="Learning Curve:SVM-Polynomial;Train=G;Test=R")
lines(x_ax,test_err_rates,col="red")
svm_model = svm(car~.,data = car_train, kernel = "polynomial")
summary(svm_model)

#Prediction on Train Data
pred = predict(svm_model, car_train)
tab = table(Predicted = pred, Actual = car_train$car)
tab
train_err = 1-sum(diag(tab))/sum(tab)
sprintf("Kernel: Polynomial, Train error: %f",train_err)

#Prediction on Test Data
pred = predict(svm_model, car_test)
tab = table(Predicted = pred, Actual = car_test$car)
tab
test_err = 1-sum(diag(tab))/sum(tab)
sprintf("Kernel: Polynomial, Test error: %f",test_err)

############################# SVM SIGMOID #######################################################

#Learning Curve
err_rates = c()
for (d in car_train_split){
  svm_model = svm(car~.,data = d, kernel = "sigmoid")
  pred1 = predict(svm_model, d)
  tab1 = table(Predicted = pred1, Actual = d$car)
  err_rates = c(err_rates,1-sum(diag(tab1))/sum(tab1))
}
test_err_rates = c()
for (d in car_test_split){
  svm_model = svm(car~.,data = d, kernel = "sigmoid")
  pred1 = predict(svm_model, d)
  tab1 = table(Predicted = pred1, Actual = d$car)
  test_err_rates = c(test_err_rates,1-sum(diag(tab1))/sum(tab1))
}
plot(x_ax, err_rates, ylim=c(0.06,0.45), type="l", col="green", ylab="error rate", xlab="training data size(in %)", main="Learning Curve:SVM-Sigmoid;Train=G;Test=R")
lines(x_ax,test_err_rates,col="red")
svm_model = svm(car~.,data = car_train, kernel = "sigmoid")
summary(svm_model)

#Prediction on Train Data
pred = predict(svm_model, car_train)
tab = table(Predicted = pred, Actual = car_train$car)
tab
train_err = 1-sum(diag(tab))/sum(tab)
sprintf("Kernel: Sigmoid, Train error: %f",train_err)

#Prediction on Test Data
pred = predict(svm_model, car_test)
tab = table(Predicted = pred, Actual = car_test$car)
tab
test_err = 1-sum(diag(tab))/sum(tab)
sprintf("Kernel: Sigmoid, Test error: %f",test_err)
 
######################## ID3 Decision TREE ######################################

ctrl <- trainControl(method = "cv", number = 5)

#Information Gain

#Learning Curve
err_rates = c()
for (d in car_train_split){
  projecttree <- rpart(car~.,data=d,method="class",parms = list(split = "information"))
  res <- predict(projecttree,d,type="class")
  tab1 = table(Predicted = res, Actual = d$car)
  err_rates = c(err_rates,1-sum(diag(tab1))/sum(tab1))
}
plot(x_ax, err_rates, ylim=c(0.010,0.035), type="l", col="green", ylab="error rate", xlab="training data size(in %)", main="Learning Curve for Decision Tree-Information Gain")

projecttree <- rpart(car~.,data=car_train,method="class",parms = list(split = "information"))
projecttree
printcp(projecttree)
plotcp(projecttree)
prp(projecttree, box.palette = "Reds", tweak = 1.2)

#Prediction on Train Data
res <- predict(projecttree,car_train,type="class")
confusionMatrix(res,car_train$car)
tab = table(Predicted = res, Actual = car_train$car)
tab
train_err = 1-sum(diag(tab))/sum(tab)
sprintf("Decision Tree: Information Gain, Train error: %f",train_err)

#Prediction on Test Data
res = predict(projecttree,car_test,type="class")
confusionMatrix(res,car_test$car)
tab = table(Predicted = res, Actual = car_test$car)
tab
test_err = 1-sum(diag(tab))/sum(tab)
sprintf("Decision Tree: Information Gain, Test error: %f",test_err)

#Gini Index

#Learning Curve

err_rates = c()
for (d in car_train_split){
  projecttree <- rpart(car~.,data=d,method="class",parms = list(split = "gini"))
  res <- predict(projecttree,d,type="class")
  tab1 = table(Predicted = res, Actual = d$car)
  err_rates = c(err_rates,1-sum(diag(tab1))/sum(tab1))
}
plot(x_ax, err_rates, ylim=c(0.010,0.040), type="l", col="green", ylab="error rate", xlab="training data size(in %)", main="Learning Curve for Decision Tree-Gini Index")


projecttree <- rpart(car~.,data=car_train,method="class",parms = list(split = "gini"))
projecttree
printcp(projecttree)
plotcp(projecttree)
prp(projecttree, box.palette = "Reds", tweak = 1.2)

#Prediction on Train Data
res <- predict(projecttree,car_train,type="class")
confusionMatrix(res,car_train$car)
tab = table(Predicted = res, Actual = car_train$car)
tab
train_err = 1-sum(diag(tab))/sum(tab)
sprintf("Decision Tree: Gini Index, Train error: %f",train_err)

#Prediction on Test Data
res = predict(projecttree,car_test,type="class")
confusionMatrix(res,car_test$car)
tab = table(Predicted = res, Actual = car_test$car)
tab
test_err = 1-sum(diag(tab))/sum(tab)
sprintf("Decision Tree: Gini Index, Test error: %f",test_err)

#Pruning
ptree<- prune(projecttree,cp= projecttree$cptable[which.min(projecttree$cptable[,"xerror"]),"CP"])
printcp(ptree)
plotcp(ptree)
prp(ptree, box.palette = "Reds", tweak = 1.2)

#Prediction on Train Data
res <- predict(ptree,car_train,type="class")
confusionMatrix(res,car_train$car)
tab = table(Predicted = res, Actual = car_train$car)
tab
train_err = 1-sum(diag(tab))/sum(tab)
sprintf("Pruned Decision Tree: Information Gain, Train error: %f",train_err)

#Prediction on Test Data
res = predict(ptree,car_test,type="class")
confusionMatrix(res,car_test$car)
tab = table(Predicted = res, Actual = car_test$car)
tab
test_err = 1-sum(diag(tab))/sum(tab)
sprintf("Pruned Decision Tree: Information Gain, Test error: %f",test_err)

################################# BOOSTED DECISION TREES##########################################################

ctr <- trainControl(method = "cv", number = 10)

boost.caret <- train(car~., car_train,
                     method='xgbTree', 
                     preProc=c('center','scale'),
                     trControl=ctr)
plot(boost.caret)
boost.caret
boost.caret.pred <- predict(boost.caret, car_train)
confusionMatrix(boost.caret.pred,car_train$car)
tab = table(Predicted = boost.caret.pred, Actual = car_train$car)
tab
train_err = 1-sum(diag(tab))/sum(tab)
sprintf("Boosted Decision Tree:, Train error: %f",train_err)

boost.caret.pred <- predict(boost.caret, car_test)
confusionMatrix(boost.caret.pred,car_test$car)
tab = table(Predicted = boost.caret.pred, Actual = car_test$car)
tab
test_err = 1-sum(diag(tab))/sum(tab)
sprintf("Boosted Decision Tree:, Test error: %f",test_err)

################################ Organics Data Set ########################################################################################################



library(ggplot2)
library(e1071)
library(rpart)
library(rpart.plot)
library(caret)
set.seed(1234)

car_data = read.csv('C:/AML - BUAN 6341/Credit-data.csv',  header=TRUE)
str(car_data)
qplot(persons,safety, data=car_data, color = A16)
car_train = car_data[1:round(0.7*nrow(car_data)),]
car_test = car_data[(round(0.7*nrow(car_data))+1):nrow(car_data),]
car_train_split = list(car_data[1:round(0.2*nrow(car_data)),],car_data[1:round(0.3*nrow(car_data)),],car_data[1:round(0.4*nrow(car_data)),],car_data[1:round(0.5*nrow(car_data)),],car_data[1:round(0.6*nrow(car_data)),],car_data[1:round(0.7*nrow(car_data)),])
car_test_split = list(car_data[round(0.9*nrow(car_data)):nrow(car_data),],car_data[round(0.85*nrow(car_data)):nrow(car_data),],car_data[round(0.80*nrow(car_data)):nrow(car_data),],car_data[round(0.75*nrow(car_data)):nrow(car_data),],car_data[round(0.70*nrow(car_data)):nrow(car_data),],car_data[round(0.65*nrow(car_data)):nrow(car_data),])
x_ax = c(20,30,40,50,60,70)

############################# SVM LINEAR #######################################################

#Learning Curve
err_rates = c()
for (d in car_train_split){
  svm_model = svm(A16~.,data = d, kernel = "linear")
  pred1 = predict(svm_model, d)
  tab1 = table(Predicted = pred1, Actual = d$A16)
  err_rates = c(err_rates,1-sum(diag(tab1))/sum(tab1))
}
test_err_rates = c()
for (d in car_test_split){
  svm_model = svm(A16~.,data = d, kernel = "linear")
  pred1 = predict(svm_model, d)
  tab1 = table(Predicted = pred1, Actual = d$A16)
  test_err_rates = c(test_err_rates,1-sum(diag(tab1))/sum(tab1))
}
plot(x_ax, err_rates, ylim=c(0.00,0.07), type="l", col="green", ylab="error rate", xlab="training data size(in %)", main="Learning Curve:SVM-Linear;Train=G;Test=R")
lines(x_ax,test_err_rates,col="red")
svm_model = svm(A16~.,data = car_train, kernel = "linear")
summary(svm_model)

#Prediction on Train Data
pred = predict(svm_model, car_train)
tab = table(Predicted = pred, Actual = car_train$A16)
tab
train_err = 1-sum(diag(tab))/sum(tab)
sprintf("Kernel: Linear, Train error: %f",train_err)

#Prediction on Test Data
pred = predict(svm_model, car_test)
tab = table(Predicted = pred, Actual = car_test$A16)
tab
test_err = 1-sum(diag(tab))/sum(tab)
sprintf("Kernel: Linear, Test error: %f",test_err)


############################# SVM RADIAL #######################################################

#Learning Curve
err_rates = c()
for (d in car_train_split){
  svm_model = svm(A16~.,data = d, kernel = "radial")
  pred1 = predict(svm_model, d)
  tab1 = table(Predicted = pred1, Actual = d$A16)
  err_rates = c(err_rates,1-sum(diag(tab1))/sum(tab1))
}
test_err_rates = c()
for (d in car_test_split){
  svm_model = svm(A16~.,data = d, kernel = "radial")
  pred1 = predict(svm_model, d)
  tab1 = table(Predicted = pred1, Actual = d$A16)
  test_err_rates = c(test_err_rates,1-sum(diag(tab1))/sum(tab1))
}
plot(x_ax, err_rates, ylim=c(0.15,0.40), type="l", col="green", ylab="error rate", xlab="training data size(in %)", main="Learning Curve:SVM-Radial;Train=G;Test=R")
lines(x_ax,test_err_rates,col="red")
svm_model = svm(A16~.,data = car_train, kernel = "radial")
summary(svm_model)

#Prediction on Train Data
pred = predict(svm_model, car_train)
tab = table(Predicted = pred, Actual = car_train$A16)
tab
train_err = 1-sum(diag(tab))/sum(tab)
sprintf("Kernel: Radial, Train error: %f",train_err)

#Prediction on Test Data
pred = predict(svm_model, car_test)
tab = table(Predicted = pred, Actual = car_test$A16)
tab
test_err = 1-sum(diag(tab))/sum(tab)
sprintf("Kernel: Radial, Test error: %f",test_err)

############################# SVM POLYNOMIAL #######################################################

#Learning Curve
err_rates = c()
for (d in car_train_split){
  svm_model = svm(A16~.,data = d, kernel = "polynomial")
  pred1 = predict(svm_model, d)
  tab1 = table(Predicted = pred1, Actual = d$A16)
  err_rates = c(err_rates,1-sum(diag(tab1))/sum(tab1))
}
test_err_rates = c()
for (d in car_test_split){
  svm_model = svm(A16~.,data = d, kernel = "polynomial")
  pred1 = predict(svm_model, d)
  tab1 = table(Predicted = pred1, Actual = d$A16)
  test_err_rates = c(test_err_rates,1-sum(diag(tab1))/sum(tab1))
}
plot(x_ax, err_rates, ylim=c(0.00,0.50), type="l", col="green", ylab="error rate", xlab="training data size(in %)", main="Learning Curve:SVM-Polynomial;Train=G;Test=R")
lines(x_ax,test_err_rates,col="red")
svm_model = svm(A16~.,data = car_train, kernel = "polynomial")
summary(svm_model)

#Prediction on Train Data
pred = predict(svm_model, car_train)
tab = table(Predicted = pred, Actual = car_train$A16)
tab
train_err = 1-sum(diag(tab))/sum(tab)
sprintf("Kernel: Polynomial, Train error: %f",train_err)

#Prediction on Test Data
pred = predict(svm_model, car_test)
tab = table(Predicted = pred, Actual = car_test$A16)
tab
test_err = 1-sum(diag(tab))/sum(tab)
sprintf("Kernel: Polynomial, Test error: %f",test_err)

############################# SVM SIGMOID #######################################################

#Learning Curve
err_rates = c()
for (d in car_train_split){
  svm_model = svm(A16~.,data = d, kernel = "sigmoid")
  pred1 = predict(svm_model, d)
  tab1 = table(Predicted = pred1, Actual = d$A16)
  err_rates = c(err_rates,1-sum(diag(tab1))/sum(tab1))
}
test_err_rates = c()
for (d in car_test_split){
  svm_model = svm(A16~.,data = d, kernel = "sigmoid")
  pred1 = predict(svm_model, d)
  tab1 = table(Predicted = pred1, Actual = d$A16)
  test_err_rates = c(test_err_rates,1-sum(diag(tab1))/sum(tab1))
}
plot(x_ax, err_rates, ylim=c(0.00,0.45), type="l", col="green", ylab="error rate", xlab="training data size(in %)", main="Learning Curve:SVM-Sigmoid;Train=G;Test=R")
lines(x_ax,test_err_rates,col="red")
svm_model = svm(A16~.,data = car_train, kernel = "sigmoid")
summary(svm_model)

#Prediction on Train Data
pred = predict(svm_model, car_train)
tab = table(Predicted = pred, Actual = car_train$A16)
tab
train_err = 1-sum(diag(tab))/sum(tab)
sprintf("Kernel: Sigmoid, Train error: %f",train_err)

#Prediction on Test Data
pred = predict(svm_model, car_test)
tab = table(Predicted = pred, Actual = car_test$A16)
tab
test_err = 1-sum(diag(tab))/sum(tab)
sprintf("Kernel: Sigmoid, Test error: %f",test_err)

######################## ID3 Decision TREE ######################################

ctrl <- trainControl(method = "cv", number = 5)

#Information Gain

#Learning Curve
err_rates = c()
for (d in car_train_split){
  projecttree <- rpart(A16~.,data=d,method="class",parms = list(split = "information"))
  res <- predict(projecttree,d,type="class")
  tab1 = table(Predicted = res, Actual = d$A16)
  err_rates = c(err_rates,1-sum(diag(tab1))/sum(tab1))
}
plot(x_ax, err_rates, ylim=c(0.010,0.035), type="l", col="green", ylab="error rate", xlab="training data size(in %)", main="Learning Curve for Decision Tree-Information Gain")

projecttree <- rpart(A16~.,data=car_train,method="class",parms = list(split = "information"))
projecttree
printcp(projecttree)
plotcp(projecttree)
prp(projecttree, box.palette = "Reds", tweak = 1.2)

#Prediction on Train Data
res <- predict(projecttree,car_train,type="class")
confusionMatrix(res,car_train$A16)
tab = table(Predicted = res, Actual = car_train$A16)
tab
train_err = 1-sum(diag(tab))/sum(tab)
sprintf("Decision Tree: Information Gain, Train error: %f",train_err)

#Prediction on Test Data
res = predict(projecttree,car_test,type="class")
confusionMatrix(res,car_test$A16)
tab = table(Predicted = res, Actual = car_test$A16)
tab
test_err = 1-sum(diag(tab))/sum(tab)
sprintf("Decision Tree: Information Gain, Test error: %f",test_err)

#Gini Index

#Learning Curve

err_rates = c()
for (d in car_train_split){
  projecttree <- rpart(A16~.,data=d,method="class",parms = list(split = "gini"))
  res <- predict(projecttree,d,type="class")
  tab1 = table(Predicted = res, Actual = d$A16)
  err_rates = c(err_rates,1-sum(diag(tab1))/sum(tab1))
}
plot(x_ax, err_rates, ylim=c(0.010,0.040), type="l", col="green", ylab="error rate", xlab="training data size(in %)", main="Learning Curve for Decision Tree-Gini Index")


projecttree <- rpart(A16~.,data=car_train,method="class",parms = list(split = "gini"))
projecttree
printcp(projecttree)
plotcp(projecttree)
prp(projecttree, box.palette = "Reds", tweak = 1.2)

#Prediction on Train Data
res <- predict(projecttree,car_train,type="class")
confusionMatrix(res,car_train$A16)
tab = table(Predicted = res, Actual = car_train$A16)
tab
train_err = 1-sum(diag(tab))/sum(tab)
sprintf("Decision Tree: Gini Index, Train error: %f",train_err)

#Prediction on Test Data
res = predict(projecttree,car_test,type="class")
confusionMatrix(res,car_test$A16)
tab = table(Predicted = res, Actual = car_test$A16)
tab
test_err = 1-sum(diag(tab))/sum(tab)
sprintf("Decision Tree: Gini Index, Test error: %f",test_err)

#Pruning
ptree<- prune(projecttree,cp= projecttree$cptable[which.min(projecttree$cptable[,"xerror"]),"CP"])
printcp(ptree)
plotcp(ptree)
prp(ptree, box.palette = "Reds", tweak = 1.2)

#Prediction on Train Data
res <- predict(ptree,car_train,type="class")
confusionMatrix(res,car_train$A16)
tab = table(Predicted = res, Actual = car_train$A16)
tab
train_err = 1-sum(diag(tab))/sum(tab)
sprintf("Pruned Decision Tree: Information Gain, Train error: %f",train_err)

#Prediction on Test Data
res = predict(ptree,car_test,type="class")
confusionMatrix(res,car_test$A16)
tab = table(Predicted = res, Actual = car_test$A16)
tab
test_err = 1-sum(diag(tab))/sum(tab)
sprintf("Pruned Decision Tree: Information Gain, Test error: %f",test_err)

################################# BOOSTED DECISION TREES##########################################################

ctr <- trainControl(method = "cv", number = 10)

boost.caret <- train(A16~., car_train,
                     method='xgbTree', 
                     preProc=c('center','scale'),
                     trControl=ctr)
plot(boost.caret)
boost.caret
boost.caret.pred <- predict(boost.caret, car_train)
confusionMatrix(boost.caret.pred,car_train$A16)
tab = table(Predicted = boost.caret.pred, Actual = car_train$A16)
tab
train_err = 1-sum(diag(tab))/sum(tab)
sprintf("Boosted Decision Tree:, Train error: %f",train_err)

boost.caret.pred <- predict(boost.caret, car_test)
confusionMatrix(boost.caret.pred,car_test$A16)
tab = table(Predicted = boost.caret.pred, Actual = car_test$A16)
tab
test_err = 1-sum(diag(tab))/sum(tab)
sprintf("Boosted Decision Tree:, Test error: %f",test_err)