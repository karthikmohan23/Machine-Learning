library(caret)
library(onehot)
library(pROC)
set.seed(1234)
data <- read.csv("C:/AML - BUAN 6341/HR_comma_sep.csv", 
                 header=TRUE)
head(data)
typeof(data$left)
#One HotEncoding
data_ <- onehot(data,stringsAsFactors = FALSE,addNA = FALSE,max_levels = 10)
pr <- as.data.frame(predict(data_,data))
head(pr)

#Scaling data to [0,1] range
maxs <- apply(pr, 2, max)    
mins <- apply(pr, 2, min)
scaled <- as.data.frame(scale(pr, center = mins, scale = maxs - mins))
scaled$left <- factor(scaled$left, labels = c( "remained","left"))
head(scaled)

# splitting data into train, test 
set.seed(1234)
index <- createDataPartition(scaled$left, p = .70,list = FALSE, times = 1)
train_knn <- scaled[ index,]
test_knn <- scaled[-index,]
head(test_knn)
#Multiple split for getting different training data sizes
train_split = list(scaled[sample(nrow(scaled), nrow(scaled)*.2),],
                   scaled[sample(nrow(scaled), nrow(scaled)*.3),],
                   scaled[sample(nrow(scaled), nrow(scaled)*.4),],
                   scaled[sample(nrow(scaled), nrow(scaled)*.5),],
                   scaled[sample(nrow(scaled), nrow(scaled)*.6),],
                   scaled[sample(nrow(scaled), nrow(scaled)*.7),])
#Store training data size in %
x_ax = c(20,30,40,50,60,70)

#KNN model using caret
set.seed(3331)
head(train_knn)
trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

knn_fit <- train(left ~., data = train_knn, method = "knn",
                 trControl=trctrl,
                 preProcess = c("center", "scale"),
                 tuneLength = 15)
knn_fit
train_accuracy_knn <- knn_fit$results$Accuracy
train_error_knn <- 1-knn_fit$results$Accuracy
#training error for best fit K
1-knn_fit$results$Accuracy[knn_fit$results$k == knn_fit$bestTune$k]

#plot train error versus K
plot(train_error_knn,  main='Training Error vs K',xlab="K",ylab='Training Error Rate',col='blue')
knnPred <- predict(knn_fit,newdata = test_knn)
confusionMatrix(knnPred, test_knn$left)
test_error_knn = 1 -mean(knnPred == test_knn$left)
test_error_knn

##ROC curve
auc_knn_test <- roc(as.numeric(test_knn$left), as.numeric(knnPred))
print(auc_knn_test)
plot(auc_knn_test, print.thres=TRUE, col = 'blue',main = 'ROC curve for HR data')

#KNN training and test error for multiple training data size
set.seed(3333)
train_error_split <- c()
test_error_split <- c()
for (i in train_split)
{
  knnfit <- train(left ~., data = i, method = "knn",tuneLength = 15)
  knnPred <- predict(knnfit,newdata = i)
  error<-1 -mean(knnPred == i$left)
  train_error_split <- c(train_error_split,error)
  TestPred <- predict(knnfit,newdata = test_knn)
  errortest<-1 -mean(TestPred == test_knn$left)
  test_error_split <- c(test_error_split,errortest)
  print(train_error_split)
  print(test_error_split)
}
train_error_split
test_error_split

#Learning Curves
#Plot training error vs training data size
plot(x_ax,train_error_split,  main='Training Error vs Training data size',
     xlab="Training data size",ylab='Train Error',col='red',type='l')

#Plot test error vs training data size
plot(x_ax,test_error_split,  main='Test Error vs Training data size',
     xlab="Training data size",ylab='Test Error',col='red',type='l')

#PLOT BOTH THE TRAIN AND TEST ERROR CURVES WITHIN THE SAME PLOT
plot(x_ax, train_error_split, ylim=c(0.044,0.077), type="l", col="green", ylab="error rate", 
     xlab="training data size(in %)", main="Train and Test Error")
lines(x_ax,test_error_split,col="red")



#----------------------------------------------------------------
### second dataset - Car Evaluation Dataset

car <- read.csv("C:/AML - BUAN 6341/car_data.csv", header=TRUE)
cardata <- car
levels(cardata$car)[levels(cardata$car)%in%c("acc","good","vgood")] <- 0
levels(cardata$car)[levels(cardata$car)%in%c("unacc")] <- 1
head(cardata)

#One HotEncoding
cardata_ <- onehot(cardata[,c(-7)],stringsAsFactors = FALSE,addNA = FALSE,max_levels = 10)
car_pr <- as.data.frame(predict(cardata_,cardata[,c(-7)]))

#scaling to [0,1] range. dummies wont get affected
maxs <- apply(car_pr, 2, max)    
mins <- apply(car_pr, 2, min)
scaled_car <- as.data.frame(scale(car_pr, center = mins, scale = maxs - mins))
head(scaled_car)

#output column is appended again as it was dropped from original dataset
scaled_car$car <- car$car
head(scaled_car)
levels(scaled_car$car)[levels(scaled_car$car)%in%c("acc","good","vgood")] <- 0
levels(scaled_car$car)[levels(scaled_car$car)%in%c("unacc")] <- 1

# splitting data into train, test
set.seed(1234)
index <- sample(1:nrow(scaled_car),round(0.70*nrow(scaled_car)))
train_car <- scaled_car[index,]
test_car <- scaled_car[-index,] 

#Multiple split for getting different training data sizes
train_car_split = list(scaled_car[1:round(0.2*nrow(scaled_car)),],
                       scaled_car[1:round(0.3*nrow(scaled_car)),],
                       scaled_car[1:round(0.4*nrow(scaled_car)),],
                       scaled_car[1:round(0.5*nrow(scaled_car)),],
                       scaled_car[1:round(0.6*nrow(scaled_car)),],
                       scaled_car[1:round(0.7*nrow(scaled_car)),])
#KNN models
set.seed(3331)
head(train_car)
trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

knn_car <- train(car ~., data = train_car, method = "knn",
                 trControl=trctrl,
                 preProcess = c("center", "scale"),
                 tuneLength = 15)
knn_car
train_accuracy_car <- knn_car$results$Accuracy
train_error_car <- 1-knn_car$results$Accuracy
#training error for best fit K
1-knn_car$results$Accuracy[knn_car$results$k == knn_car$bestTune$k]

#Plotting train error vs K
plot(train_error_car,  main='Traing Error vs K',xlab="K",ylab='Train Error',col='red')
carPred <- predict(knn_car,newdata = test_car)
confusionMatrix(carPred, test_car$car)
test_error_car = 1 -mean(carPred == test_car$car)
test_error_car

#ROC Curve
auc_car_test <- roc(as.numeric(test_car$car), as.numeric(carPred))
print(auc_car_test)
plot(auc_car_test, print.thres=TRUE, col = 'blue',main='ROC for Car data')

#KNN - train test error for multiple training data size
set.seed(3333)
train_error_car_split <- c()
test_error_car_split <- c()
for (i in train_car_split)
{
  print(i)
  knnfit <- train(car ~., data = i, method = "knn",tuneLength = 15)
  knnPred <- predict(knnfit,newdata = i)
  error<-1 -mean(knnPred == i$car)
  train_error_car_split <- c(train_error_car_split,error)
  print(train_error_car_split)
  testPred <- predict(knnfit,newdata = test_car)
  error1<-1 -mean(testPred == test_car$car)
  test_error_car_split <- c(test_error_car_split,error1)
  print(test_error_car_split)
}
train_error_car_split
test_error_car_split

#Testing for particular training data split with the test error from loop to make sure loop is correct
set.seed(3333)
knnfit <- train(car ~., data = train_car_split[[4]], method = "knn",tuneLength = 15)
testPred <- predict(knnfit,newdata = test_car)
1 -mean(testPred == test_car$car)

#Learning Curves
#Plot training error vs training data size
plot(x_ax,train_error_car_split,  main='Training Error vs Training data size',
     xlab="Training data size",ylab='Train Error',col='red',type='l')

#Plot test error vs training data size
plot(x_ax,test_error_car_split,  main='Test Error vs Training data size',
     xlab="Training data size",ylab='Test Error',col='red',type='l')

#PLOT BOTH THE TRAIN AND TEST ERROR CURVES WITHIN THE SAME PLOT
plot(x_ax, train_error_car_split, ylim=c(0.0049,0.25), type="l", col="green", ylab="error rate", 
     xlab="training data size(in %)", main="Train and Test Error")
lines(x_ax,test_error_car_split,col="red")


