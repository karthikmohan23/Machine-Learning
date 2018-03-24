#dat = read.csv("D:/UTD MSBA/Fall 2017 (Sem 3)/Machine Learning/Project 1/Bike-Sharing-Dataset/hour.csv",header = TRUE)
dat = read.csv('E:/AML - BUAN 6341/hour.csv',  header=TRUE)
dat = subset(dat, select = -c(instant,dteday,casual,registered) )

############################################# ADDING DUMMY VARIABLES ############################################################################

for(level in unique(dat$season)){
  dat[paste("season", level, sep = "_")] <- ifelse(dat$season == level, 1, 0)
}
for(level in unique(dat$mnth)){
  dat[paste("mnth", level, sep = "_")] <- ifelse(dat$mnth == level, 1, 0)
}
for(level in unique(dat$hr)){
  dat[paste("hr", level, sep = "_")] <- ifelse(dat$hr == level, 1, 0)
}
for(level in unique(dat$weekday)){
  dat[paste("weekday", level, sep = "_")] <- ifelse(dat$weekday == level, 1, 0)
}
for(level in unique(dat$weathersit)){
  dat[paste("weathersit", level, sep = "_")] <- ifelse(dat$weathersit == level, 1, 0)
}
dat = subset(dat, select = -c(season,mnth,hr,weekday,weathersit) )


####################################### SPLITTING THE DATA INTO TRAINING AND TESTING SETS ##########################################################

train_data = dat[1:round(0.7*nrow(dat)),]
train_data_y = train_data['cnt']
train_data_x = subset(train_data, select = -c(cnt) )
train_data_x = data.frame(append(train_data_x, list(Intercept=1), after=0))
train_x_arr = t(as.matrix(train_data_x))
train_y_arr = t(as.matrix(train_data_y))
test_data = dat[(round(0.7*nrow(dat))+1):nrow(dat),]
test_data_y = test_data['cnt']
test_data_x = subset(test_data, select = -c(cnt) )
test_data_x = data.frame(append(test_data_x, list(Intercept=1), after=0))
test_x_arr = t(as.matrix(test_data_x))
test_y_arr = t(as.matrix(test_data_y))
sprintf("Train data length: %i and Test data length: %i", nrow(train_data_x), nrow(test_data_x))

betas = c(0.06832330454686208, 0.20881985487546728, 0.4764368259286519, 0.3951073425073658, 0.9399067315613835, 0.6856495119480731, 0.5042290334473788, 0.7340348476352466, 0.7740242791744092, 0.9945409696921729, 0.8847207037900368, 0.23366490392845052, 0.5328858468958018, 0.5469309686110189, 0.2117765140107074, 0.7205458435825078, 0.3198904581768499, 0.4492404262455927, 0.950649946249515, 0.41820922962978757, 0.3363343985480124, 0.47325315975578053, 0.3880419621717299, 0.584215796890573, 0.9458242256342042, 0.8249130910787242, 0.29927367219246404, 0.7102358446354647, 0.12387825793366336, 0.08794026119906961, 0.3196009624932792, 0.7860688889074261, 0.834180239489888, 0.8224944404045879, 0.2616294631651652, 0.9767864233685642, 0.3363199802201926, 0.2892692538045438, 0.7513911415916674, 0.9158861822755658, 0.17382942208798047, 0.6337268642484924, 0.42183340183519, 0.09499605327928662, 0.3272177777124635, 0.34285493372039544, 0.035147125356191466, 0.9569217929519969, 0.09819079552357646, 0.6571978011341697, 0.33407585977948184, 0.534272609172775, 0.573187842599701, 0.9621638491195679, 0.6120418779301108, 0.6023080849924535, 0.2033676880537726, 0.9207136838174814, 0.15998400112797317)

theta_arr   = matrix(betas,nrow=1,ncol=59)

############################################### COST FUNCTION ########################################################################################

cost_func = function(theta, x, y){
  return(1/(2*ncol(x)) * sum(((theta %*% x) - y)^2))
}

############################################### GRADIENT DESCENT ########################################################################################

gradient_descent = function(alpha=0.01, j_treshold=0.1, x=c(), y=c()){
  vec = c()
  iter = c()
  cost = cost_func(theta_arr, x, y)
  temp = theta_arr - t(alpha * (x %*% t((theta_arr %*% x )-y)) / ncol(x))
  assign('theta_arr',temp,envir=.GlobalEnv)
  new_cost = cost_func(theta_arr, x, y)
  delta_cost = abs((cost - new_cost) / cost)
  vec = c(cost,new_cost)
  cost = new_cost
  iter = c(0,1)
  i = 1
  while(is.finite(new_cost) && (delta_cost>j_treshold)){
    te = theta_arr - t(alpha * (x %*% t((theta_arr %*% x )-y)) / ncol(x))
    assign('theta_arr',te,envir=.GlobalEnv)
    new_cost = cost_func(theta_arr, x, y)
    delta_cost = abs((cost - new_cost) / cost)
    i = i+1
    vec = c(vec,new_cost)
    iter = c(iter,i)
    cost = new_cost
    
  }
  assign('coefficients',t(theta_arr),envir=.GlobalEnv)
  assign('cost_values',t(vec),envir=.GlobalEnv)
  assign('iterations',t(iter),envir=.GlobalEnv)
  print(cost)
}

######################### Cost Func vs Iterations (Train and Test sets) ######################

gradient_descent(alpha = 0.56,j_treshold = 0.000001,train_x_arr,train_y_arr)
train_cost = cost_values
train_iter = iterations
train_coeff = coefficients

rm(coefficients)
rm(cost_values)
rm(iterations)

gradient_descent(alpha = 0.42,j_treshold = 0.0000001,x = test_x_arr,y = test_y_arr)
test_cost = cost_values
test_iter = iterations
test_coeff = coefficients
plot(train_iter, train_cost,type="l",col="green", ylab="Cost Function",xlim=c(0,1200), xlab="Iterations", main="Red - Test; Green - Train")
lines(test_iter,test_cost,col="red")

########################## Cost values vs alphas (Train set) ##########################

a_train = c(0.01,0.03,0.06,0.1,0.13,0.16,0.2,0.23,0.26,0.3,0.33,0.36,0.4,0.43,0.46,0.5,0.53,0.56)
ct_train = c()
for (i in a_train){
  gradient_descent(alpha = i,j_treshold = 0.000001,x = train_x_arr, y = train_y_arr)
  ct_train = c(ct_train, cost_values[length(cost_values)])
}
plot(a_train, ct_train, ylab="Cost Function", xlab="alpha", main="Cost Func over different alphas for train-set")

########################## Cost values vs alphas (Test set) ##########################

a_test = c(0.01,0.02,0.04,0.06,0.08,0.1,0.12,0.14,0.16,0.18,0.2,0.22,0.24,0.26,0.28,0.3,0.32,0.34,0.36,0.38,0.4,0.42)
ct_test = c()
for (i in a_test){
  gradient_descent(alpha = i,j_treshold = 0.0000001,x = test_x_arr,y = test_y_arr)
  ct_test = c(ct_test, cost_values[length(cost_values)])
}
plot(a_test, ct_test, ylab="Cost Function", xlab="alpha", main="Cost Func over different alphas for test-set")


########################## Cost values vs threshold (Train set) ##########################

thres_train = c(0.1, 0.01,0.001,0.0001,0.00001,0.000001,0.0000001,0.00000001,0.000000001,0.0000000001)
ct_train = c()
for (i in thres_train){
  gradient_descent(alpha = 0.56,j_treshold = i,x = train_x_arr,y = train_y_arr)
  ct_train = c(ct_train, cost_values[length(cost_values)])
}
plot(thres_train, ct_train, ylab="Cost Function", xlab="threshold",xlim = c(0.1,0.0000000001),log = "x", main="Cost Func over different threshold for train-set")

########################## Cost values vs threshold (Test set) ##########################

thres_test = c(0.1, 0.01,0.001,0.0001,0.00001,0.000001,0.0000001,0.00000001,0.000000001,0.0000000001)
ct_test = c()
for (i in thres_test){
  gradient_descent(alpha = 0.42,j_treshold = i,x = test_x_arr,y = test_y_arr)
  ct_test = c(ct_test, cost_values[length(cost_values)])
}
plot(thres_test, ct_test, ylab="Cost Function", xlab="threshold",xlim = c(0.1,0.0000000001),log = "x", main="Cost Func over different threshold for test-set")

##############################################################################################

print("Train Set:")
sprintf("Cost Function: %s",train_cost[length(train_cost)] )
print("Coefficients:")
print(train_coeff)

print("Test Set:")
sprintf("Cost Function: %s",test_cost[length(test_cost)] )
print("Coefficients:")
print(test_coeff)


############################ USING BUILT-IN PACKAGE ########################################################################################

model <- lm(cnt~yr + holiday+ workingday+ temp+  atemp+  hum+ windspeed+ season_1+ season_2+ season_3+ season_4+ mnth_1+ mnth_2+ mnth_3+ mnth_4+ mnth_5+ mnth_6+ mnth_7+ mnth_8+ mnth_9+ mnth_10+ mnth_11+ mnth_12+ hr_0+ hr_1+ hr_2+ hr_3+ hr_4+ hr_5+ hr_6+ hr_7+ hr_8+ hr_9+ hr_10+ hr_11+ hr_12+ hr_13+ hr_14+ hr_15+ hr_16+ hr_17+ hr_18+ hr_19+ hr_20+ hr_21+ hr_22+ hr_23+ weekday_6+ weekday_0+ weekday_1+ weekday_2+ weekday_3+ weekday_4+ weekday_5+weathersit_1+ weathersit_2+ weathersit_3+ weathersit_4, data = train_data)
print("Built-in Package")
print(model)
