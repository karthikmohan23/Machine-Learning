library(onehot)
library(caret)
library(rpart)
library(rpart.plot)
library(fastICA)
library(h2o)
library(plyr)
HR <- read.csv('D:/UTD MSBA/Fall 2017 (Sem 3)/Machine Learning/Project 4/HR_comma_sep.csv',header = TRUE)
HR.f <- HR
HR.f$left <- NULL
#one  hot encoding
data_ <- onehot(HR.f,stringsAsFactors = FALSE,addNA = FALSE,max_levels = 20)
data <- as.data.frame(predict(data_,HR.f))
head(data)
#scaling
maxs <- apply(data, 2, max)    
mins <- apply(data, 2, min)
scaled <- as.data.frame(scale(data, center = mins, scale = maxs - mins))
dim(scaled)
summary(scaled)
set.seed(123)

# Compute and plot within cluster Sum of squares for k = 1 to k = 15.
k <- 15
wss <- sapply(2:k,function(k){kmeans(scaled, k, nstart=10,iter.max = 15 )$tot.withinss})
#tot.withinss or total within-cluster sum of square should be as small as possible
plot(2:k, wss,type="b", pch = 19, frame = FALSE,xlab="Number of clusters K",ylab="Total within-clusters sum of squares")
res = kmeans(scaled,9)
table(HR$left,res$cluster)
#(between_SS / total_SS) should be high - it explains the total variation 

#--------------------------Car----------------
set.seed(123)
cardata <- read.csv("D:/UTD MSBA/Fall 2017 (Sem 3)/Machine Learning/Project 4/car_data.csv", header=TRUE)
cardata.f <- cardata
cardata.f$car <- NULL

#one  hot encoding
one_hot_encoding = function(dat){
  t = onehot(dat,stringsAsFactors = FALSE,addNA = FALSE,max_levels = 20)
  t = as.data.frame(predict(t,dat))
  return(t)
}

#Scaling
scaling_data = function(dat){
  maxs <- apply(dat, 2, max)    
  mins <- apply(dat, 2, min)
  dat <- as.data.frame(scale(dat, center = mins, scale = maxs - mins))
  return(dat)
}

#Elbow Graph
get_elbow_graph <- function(dat){
  k <- 15
  wss_car <- sapply(1:k,function(k){kmeans(dat, k)$tot.withinss})
  wss_car
  plot(1:k, wss_car,type="b", pch = 19, frame = FALSE,xlab="Number of clusters K",
       ylab="Total within-clusters sum of squares")
}

data = one_hot_encoding(cardata.f)
scaled_car = scaling_data(data)

set.seed(123)
sprintf("Initial K-Means")
get_elbow_graph(scaled_car)
k_means_org_features = kmeans(scaled_car,3)
k_means_org_features

#Nueral Networks with Cluster features
clust_features = one_hot_encoding(as.data.frame(as.factor(k_means_org_features$cluster)))
colnames(clust_features) = c("cluster 1","cluster 2","cluster 3","cluster 4", "cluster 5", "cluster 6", "cluster 7","cluster 8")
h2o.init(ip = "localhost", port = 54321, max_mem_size = "4000m")
clust_features$car <- as.factor(cardata$car)
splits <- h2o.splitFrame(as.h2o(clust_features), c(0.6,0.19,0.2), seed=1234)
train  <- h2o.assign(splits[[1]], "train.hex") # 60%
valid  <- h2o.assign(splits[[2]], "valid.hex") # 19%
test   <- h2o.assign(splits[[3]], "test.hex")  # 20%

response <- "car"
predictors <- setdiff(names(clust_features), response)
predictors

m1 <- h2o.deeplearning( 
  training_frame=train, 
  validation_frame=valid,   ## validation dataset: used for scoring and early stopping
  x=predictors,
  y=response,
  activation="Rectifier",   
  hidden=c(100,100,100),       
  epochs=10,
  nfolds = 5,
  seed = 123,
  variable_importances=T,    
  l2 = 6e-4,
  loss = "CrossEntropy",
  distribution = "bernoulli",
  stopping_metric = "misclassification"
)
pred = h2o.predict(m1,train)
accuracy = pred$predict == train$car
err_rates = 1 - mean(accuracy)
sprintf("Train Error: %f",err_rates)
pred = h2o.predict(m1,test)
accuracy = pred$predict == test$car
test_err_rates = 1 - mean(accuracy)
sprintf("Test Error: %f",test_err_rates)


#PCA
set.seed(123)
pr = princomp(scaled_car,scores = TRUE)
pr_cardata = pr$scores[,1:14]
sprintf("After PCA")
get_elbow_graph(pr_cardata)
k_means_pca_features = kmeans(pr_cardata,3)
k_means_pca_features


#ICA
set.seed(123)
ic = fastICA(scaled_car, n.comp = 10, alg.typ = "parallel", fun = "logcosh", alpha = 1,
        method = "R", row.norm = FALSE, maxit = 200,
        tol = 0.0001, verbose = FALSE)
ica_data = ic$S
sprintf("After ICA")
get_elbow_graph(ica_data)
k_means_ica_features = kmeans(ica_data,12)
k_means_ica_features

#RCA
random.component.selection <- function(d=2, d.original=10)     {
  selected.features <- numeric(d);
  n.feat <- d.original+1;
  feat <- floor(runif(1,1,n.feat));
  selected.features[1] <- feat;
  for (i in 2:d) {
    present <- TRUE;
    while(present)  {
      feat <- floor(runif(1,1,n.feat));
      for (j in 1:(i-1)) {
        if (selected.features[j] == feat)
          break;
      }
      if ((j==i-1) && (selected.features[j] != feat)) {
        present<-FALSE;
        selected.features[i] <- feat;    
      }    
    }
  } 
  selected.features
} 

random_projection <- function(d, m, scaling=FALSE){
  d.original <- nrow(m);
  if (d >= d.original)
    stop("random.subspace: subspace dimension must be lower than space dimension", call.=FALSE);
  # generation of the vector selected.features  containing the indices randomly selected
  selected.features <- random.component.selection(d, d.original);
  
  # random data projection
  if (scaling == TRUE)
    reduced.m <- sqrt(d.original/d) * m[selected.features,]
  else 
    reduced.m <- m[selected.features,];
  reduced.m 
}
m = as.matrix(t(scaled_car))
rp_features = as.data.frame(t(random_projection(20, m)))
sprintf("After RCA")
get_elbow_graph(rp_features)
k_means_rp_features = kmeans(rp_features,8)
k_means_rp_features

#Feature Selection (Decision Tree)
projecttree <- rpart(car~.,data=cardata,method="class",parms = list(split = "information"))
sig_vars = names(projecttree$variable.importance)
data = one_hot_encoding(cardata[,sig_vars])
scaled_car = scaling_data(data)
get_elbow_graph(scaled_car)
k_means_f_select = kmeans(scaled_car,6)
k_means_f_select




#output column is appended again as it was dropped from original dataset
#scaled_car$car <- car$car
#head(scaled_car)
#levels(scaled_car$car)[levels(scaled_car$car)%in%c("acc","good","vgood")] <- 0
#levels(scaled_car$car)[levels(scaled_car$car)%in%c("unacc")] <- 1

# Compute and plot wss for k = 1 to k = 15.




