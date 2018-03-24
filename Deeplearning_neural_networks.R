library(h2o)
library(plyr)
set.seed(1234)

############################# CAR EVALUATION ################################################################

h2o.init(ip = "localhost", port = 54321, max_mem_size = "4000m")
data <- read.csv("C:/AML - BUAN 6341/car_data.csv", header=TRUE)



data$car <- as.factor(data$car)
splits <- h2o.splitFrame(as.h2o(data), c(0.6,0.19,0.2), seed=1234)
train  <- h2o.assign(splits[[1]], "train.hex") # 60%
valid  <- h2o.assign(splits[[2]], "valid.hex") # 19%
test   <- h2o.assign(splits[[3]], "test.hex")  # 20%

response <- "car"
predictors <- setdiff(names(data), response)
predictors

#Tanh activation function
m1 <- h2o.deeplearning( 
  training_frame=train, 
  validation_frame=valid,   ## validation dataset: used for scoring and early stopping
  x=predictors,
  y=response,
  activation="Tanh",  
  hidden=c(200,200,200),       
  epochs=10,
  nfolds = 5,
  variable_importances=T,    
  l2 = 6e-4,
  loss = "CrossEntropy",
  distribution = "bernoulli",
  stopping_metric = "misclassification",
  seed = 123
  
)
pred = h2o.predict(m1,train)
accuracy = pred$predict == train$car
train_err_rates_main = 1 - mean(accuracy)
sprintf("Neural Network (Tanh) Train Error rate: %f",train_err_rates_main)
h2o.confusionMatrix(m1,newdata=train)
pred = h2o.predict(m1,test)
accuracy = pred$predict == test$car
test_err_rates_main = 1 - mean(accuracy)
sprintf("Neural Network (Tanh) Test Error rate: %f",test_err_rates_main)
h2o.confusionMatrix(m1,newdata=test)



#Rectifier activation function
m1 <- h2o.deeplearning( 
  training_frame=train, 
  validation_frame=valid,   ## validation dataset: used for scoring and early stopping
  x=predictors,
  y=response,
  activation="Rectifier",  
  hidden=c(200,200,200),       
  epochs=10,
  nfolds = 5,
  variable_importances=T,    
  l2 = 6e-4,
  loss = "CrossEntropy",
  distribution = "bernoulli",
  stopping_metric = "misclassification",
  seed = 123
  )

pred = h2o.predict(m1,train)
accuracy = pred$predict == train$car
train_err_rates_main = 1 - mean(accuracy)
sprintf("Neural Network (Rectifier) Train Error rate: %f",train_err_rates_main)
h2o.confusionMatrix(m1,newdata=train)
pred = h2o.predict(m1,test)
accuracy = pred$predict == test$car
test_err_rates_main = 1 - mean(accuracy)
sprintf("Neural Network (Rectifier) Test Error rate: %f",test_err_rates_main)
h2o.confusionMatrix(m1,newdata=test)


#Comparing this two models we see that the Tanh activation function has less error, so we will experiment various things with the Tanh activation function

#Experimentating the number of layers
hidden_layer = list(c(200),c(200,200),c(200,200,200),c(200,200,200,200),c(200,200,200,200,200))





test_err_rates = c()
for (d in hidden_layer){
  m1 <- h2o.deeplearning( 
    training_frame=train, 
    validation_frame=valid,   ## validation dataset: used for scoring and early stopping
    x=predictors,
    y=response,
    activation="Rectifier",  
    hidden=d,       
    epochs=1,
    nfolds = 5,
    variable_importances=T,    
    l2 = 6e-4,
    loss = "CrossEntropy",
    distribution = "bernoulli",
    stopping_metric = "misclassification",
    seed = 123
  )
  pred = h2o.predict(m1,test)
  accuracy = pred$predict == test$car
  test_err_rates = c(test_err_rates,1 - mean(accuracy))
  print(test_err_rates)
  
}

plot(list(1,2,3,4,5), test_err_rates, type="l", col="green",ylim=c(min(test_err_rates),max(test_err_rates)), ylab="error rate", xlab="Number of hidden layers", main="Error rates vs Layer count")

#Experimentating the number of Nodes in each layer
hidden_nodes = list(c(100,100,100),c(200,200,200),c(300,300,300),c(400,400,400),c(500,500,500))





test_err_rates = c()
for (d in hidden_nodes){
  m1 <- h2o.deeplearning( 
    training_frame=train, 
    validation_frame=valid,   ## validation dataset: used for scoring and early stopping
    x=predictors,
    y=response,
    activation="Rectifier",  
    hidden=d,       
    epochs=1,
    nfolds = 5,
    variable_importances=T,    
    l2 = 6e-4,
    loss = "CrossEntropy",
    distribution = "bernoulli",
    stopping_metric = "misclassification",
    seed = 123
  )
  pred = h2o.predict(m1,test)
  accuracy = pred$predict == test$car
  test_err_rates = c(test_err_rates,1 - mean(accuracy))
  print(test_err_rates)
  
}

plot(list(100,200,300,400,500), test_err_rates, type="l", col="green",ylim=c(min(test_err_rates),max(test_err_rates)), ylab="error rate", xlab="Number of hidden layer Nodes", main="Error rates vs Node count")


#after choosing the number of layers as 3 and Nodes as 200 each, now we will experiment with the epoch function

epoch = list(1,5,10,20)


test_err_rates = c()
for (d in epoch){
  m1 <- h2o.deeplearning( 
    training_frame=train, 
    validation_frame=valid,   ## validation dataset: used for scoring and early stopping
    x=predictors,
    y=response,
    activation="Tanh",  
    hidden=c(200,200,200),       
    epochs=d,
    nfolds = 5,
    variable_importances=T,    
    l2 = 6e-4,
    loss = "CrossEntropy",
    distribution = "bernoulli",
    stopping_metric = "misclassification",
    seed = 123
  )
  pred = h2o.predict(m1,test)
  accuracy = pred$predict == test$car
  test_err_rates = c(test_err_rates,1 - mean(accuracy))
  print(test_err_rates)
  
}

plot(list(1,5,10,20), test_err_rates, type="l", col="green",ylim=c(min(test_err_rates),max(test_err_rates)), ylab="error rate", xlab="Number of epochs" ,main="Error rates vs Epoch")





#summary(m1)
#plot(h2o.performance(m1))
#Plotting the error rate for different splits of traninng data

hr_train_split = list(train[1:round(0.2*nrow(train)),],train[1:round(0.4*nrow(train)),],train[1:round(0.6*nrow(train)),],train[1:round(0.8*nrow(train)),],train[1:round(1*nrow(train)),])

x_ax = c(20,40,60,80,100)




err_rates = c()
test_err_rates = c()
for (d in hr_train_split){
  m1 <- h2o.deeplearning( #model_id="dl_model_first",
    training_frame=d, 
    validation_frame=valid,   ## validation dataset: used for scoring and early stopping
    x=predictors,
    y=response,
    activation="Rectifier",  ## default
    hidden=c(200,200,200),       ## default: 2 hidden layers with 200 neurons each
    epochs=10,
    nfolds = 5,
    seed = 123,
    variable_importances=T,    ## not enabled by default
    l2 = 6e-4,
    loss = "CrossEntropy",
    distribution = "bernoulli",
    stopping_metric = "misclassification"
  )
  pred = h2o.predict(m1,d)
  accuracy = pred$predict == d$car
  err_rates = c(err_rates,1 - mean(accuracy))
  pred = h2o.predict(m1,test)
  accuracy = pred$predict == test$car
  test_err_rates = c(test_err_rates,1 - mean(accuracy))
  print(test_err_rates)
}

plot(x_ax, err_rates, type="l", col="green",ylim=c(min(c(err_rates,test_err_rates)),max(c(err_rates,test_err_rates))), ylab="error rate", xlab="training data size(in %)", main="Learning Curve:NN (Rectifier);Train=G;Test=R")
lines(x_ax,test_err_rates,col="red")


############################# HR ANALYTICS ################################################################

library(h2o)
library(plyr)
set.seed(1234)
h2o.init(ip = "localhost", port = 54321, max_mem_size = "4000m")
data <- read.csv("C:/AML - BUAN 6341/HR_comma_sep.csv", header=TRUE)

for(level in unique(data$salary)){
  data[paste("salary", level, sep = "_")] <- ifelse(data$salary == level, 1, 0)
}
for(level in unique(data$sales)){
  data[paste("sales", level, sep = "_")] <- ifelse(data$sales == level, 1, 0)
}
data = subset(data, select = -c(sales,salary) )


maxs <- apply(data, 2, max) 
mins <- apply(data, 2, min)
scaled <- as.data.frame(scale(data, center = mins, scale = maxs - mins))
scaled$left <- as.factor(scaled$left)
splits <- h2o.splitFrame(as.h2o(scaled), c(0.6,0.19,0.2), seed=1234)
train  <- h2o.assign(splits[[1]], "train.hex") 
valid  <- h2o.assign(splits[[2]], "valid.hex")
test   <- h2o.assign(splits[[3]], "test.hex")

response <- "left"
predictors <- setdiff(names(data), response)
predictors

#Tanh activation function
m1 <- h2o.deeplearning( 
  training_frame=train, 
  validation_frame=valid,   ## validation dataset: used for scoring and early stopping
  x=predictors,
  y=response,
  activation="Tanh",  
  hidden=c(300,300,300),       
  epochs=5,
  nfolds = 5,
  variable_importances=T,    
  l2 = 6e-4,
  loss = "CrossEntropy",
  distribution = "bernoulli",
  stopping_metric = "misclassification",
  seed = 123,
  export_weights_and_biases = TRUE,
  initial_weights = NULL,
  initial_biases = NULL
  
)

pred = h2o.predict(m1,train)
accuracy = pred$predict == train$left
train_err_rates_main = 1 - mean(accuracy)
sprintf("Neural Network (Tanh) Train Error rate: %f",train_err_rates_main)
h2o.confusionMatrix(m1,newdata=train)
pred = h2o.predict(m1,test)
accuracy = pred$predict == test$left
test_err_rates_main = 1 - mean(accuracy)
sprintf("Neural Network (Tanh) Test Error rate: %f",test_err_rates_main)
h2o.confusionMatrix(m1,newdata=test)



#Rectifier activation function
m1 <- h2o.deeplearning( 
  training_frame=train, 
  validation_frame=valid,   ## validation dataset: used for scoring and early stopping
  x=predictors,
  y=response,
  activation="Rectifier",  
  hidden=c(300,300,230),       
  epochs=5,
  nfolds = 5,
  variable_importances=T,    
  l2 = 6e-4,
  loss = "CrossEntropy",
  distribution = "bernoulli",
  stopping_metric = "misclassification",
  seed = 123,
  export_weights_and_biases = TRUE,
  initial_weights = NULL,
  initial_biases = NULL
)
pred = h2o.predict(m1,train)
accuracy = pred$predict == train$left
train_err_rates_main = 1 - mean(accuracy)
sprintf("Neural Network (RFectifier) Train Error rate: %f",train_err_rates_main)
h2o.confusionMatrix(m1,newdata=train)
pred = h2o.predict(m1,test)
accuracy = pred$predict == test$left
test_err_rates_main = 1 - mean(accuracy)
sprintf("Neural Network (Rectifier) Test Error rate: %f",test_err_rates_main)
h2o.confusionMatrix(m1,newdata=test)


#Comparing this two models we see that the Tanh activation function has less error, so we will experiment various things with the Tanh activation function

#Experimentating the number of layers
hidden_layer = list(c(200),c(200,200),c(200,200,200),c(200,200,200,200),c(200,200,200,200,200))





test_err_rates = c()
for (d in hidden_layer){
  m1 <- h2o.deeplearning( 
    training_frame=train, 
    validation_frame=valid,   ## validation dataset: used for scoring and early stopping
    x=predictors,
    y=response,
    activation="Tanh",  
    hidden=d,       
    epochs=1,
    nfolds = 5,
    variable_importances=T,    
    l2 = 6e-4,
    loss = "CrossEntropy",
    distribution = "bernoulli",
    stopping_metric = "misclassification",
    seed = 123,
    export_weights_and_biases = TRUE,
    initial_weights = NULL,
    initial_biases = NULL
  )
  pred = h2o.predict(m1,test)
  accuracy = pred$predict == test$left
  test_err_rates = c(test_err_rates,1 - mean(accuracy))
  print(test_err_rates)
  
}

plot(list(1,2,3,4,5), test_err_rates, type="l", col="green",ylim=c(min(test_err_rates),max(test_err_rates)), ylab="error rate", xlab="Number of hidden layers", main="Error rates vs Layer count")


#after choosing the number of layers as 3 , now we will experiment with the epoch function

epoch = list(1,5,10,20)


test_err_rates = c()
for (d in epoch){
  m1 <- h2o.deeplearning( 
    training_frame=train, 
    validation_frame=valid,   ## validation dataset: used for scoring and early stopping
    x=predictors,
    y=response,
    activation="Tanh",  
    hidden=c(200,200,200),       
    epochs=d,
    nfolds = 5,
    variable_importances=T,    
    l2 = 6e-4,
    loss = "CrossEntropy",
    distribution = "bernoulli",
    stopping_metric = "misclassification",
    seed = 123,
    export_weights_and_biases = TRUE,
    initial_weights = NULL,
    initial_biases = NULL
  )
  pred = h2o.predict(m1,test)
  accuracy = pred$predict == test$left
  test_err_rates = c(test_err_rates,1 - mean(accuracy))
  print(test_err_rates)
  
}

plot(list(1,5,10,20), test_err_rates, type="l", col="green",ylim=c(min(test_err_rates),max(test_err_rates)), ylab="error rate", xlab="Number of epochs" ,main="Error rates vs Epoch")

#after choosing the epoch as 20 , now we will experiment with the number of nodes

hidden_layer = list(c(100,100,100),c(200,200,200),c(300,300,300))





test_err_rates = c()
for (d in hidden_layer){
  m1 <- h2o.deeplearning( 
    training_frame=train, 
    validation_frame=valid,   ## validation dataset: used for scoring and early stopping
    x=predictors,
    y=response,
    activation="Tanh",  
    hidden=d,       
    epochs=20,
    nfolds = 5,
    variable_importances=T,    
    l2 = 6e-4,
    loss = "CrossEntropy",
    distribution = "bernoulli",
    stopping_metric = "misclassification",
    seed = 123,
    export_weights_and_biases = TRUE,
    initial_weights = NULL,
    initial_biases = NULL
  )
  pred = h2o.predict(m1,test)
  accuracy = pred$predict == test$left
  test_err_rates = c(test_err_rates,1 - mean(accuracy))
  print(test_err_rates)
  
}

plot(list(100,200,300), test_err_rates, type="l", col="green",ylim=c(min(test_err_rates),max(test_err_rates)), ylab="error rate", xlab="Number of nodes", main="Error rates vs nodes ")



#Plotting the error rate for different splits of training data

hr_train_split = list(train[1:round(0.2*nrow(train)),],train[1:round(0.4*nrow(train)),],train[1:round(0.6*nrow(train)),],train[1:round(0.8*nrow(train)),],train[1:round(1*nrow(train)),])

x_ax = c(20,40,60,80,100)




err_rates = c()
test_err_rates = c()
for (d in hr_train_split){
  m1 <- h2o.deeplearning( 
    training_frame=d, 
    validation_frame=valid,   ## validation dataset: used for scoring and early stopping
    x=predictors,
    y=response,
    activation="Tanh",  
    hidden=c(300,300,300),       
    epochs=20,
    nfolds = 5,
    seed = 123,
    variable_importances=T,    
    l2 = 6e-4,
    loss = "CrossEntropy",
    distribution = "bernoulli",
    stopping_metric = "misclassification")
  pred = h2o.predict(m1,d)  
  accuracy = pred$predict == d$left
  err_rates = c(err_rates,1 - mean(accuracy))
  pred = h2o.predict(m1,test)
  accuracy = pred$predict == test$left
  test_err_rates = c(test_err_rates,1 - mean(accuracy))
  print(test_err_rates)
}

plot(x_ax, err_rates, type="l", col="green",ylim=c(min(c(err_rates,test_err_rates)),max(c(err_rates,test_err_rates))), ylab="error rate", xlab="training data size(in %)", main="Learning Curve:Neural-Net;Train=G;Test=R")
lines(x_ax,test_err_rates,col="red")



hr_train_split = list(train[1:round(0.2*nrow(train)),],train[1:round(0.4*nrow(train)),],train[1:round(0.6*nrow(train)),],train[1:round(0.8*nrow(train)),],train[1:round(1*nrow(train)),])

x_ax = c(20,40,60,80,100)




err_rates = c()
test_err_rates = c()
for (d in hr_train_split){
  m1 <- h2o.deeplearning( 
    training_frame=d, 
    validation_frame=valid,   ## validation dataset: used for scoring and early stopping
    x=predictors,
    y=response,
    activation="Rectifier",  
    hidden=c(300,300,300),       
    epochs=20,
    nfolds = 5,
    seed = 123,
    variable_importances=T,    
    l2 = 6e-4,
    loss = "CrossEntropy",
    distribution = "bernoulli",
    stopping_metric = "misclassification")
  pred = h2o.predict(m1,d)  
  accuracy = pred$predict == d$left
  err_rates = c(err_rates,1 - mean(accuracy))
  pred = h2o.predict(m1,test)
  accuracy = pred$predict == test$left
  test_err_rates = c(test_err_rates,1 - mean(accuracy))
  print(test_err_rates)
}

plot(x_ax, err_rates, type="l", col="green",ylim=c(min(c(err_rates,test_err_rates)),max(c(err_rates,test_err_rates))), ylab="error rate", xlab="training data size(in %)", main="Learning Curve:Neural-Net;Train=G;Test=R")
lines(x_ax,test_err_rates,col="red")






