library(data.table) #reading in the data
library(dplyr) #dataframe manipulation
library(ggplot2) #viz
library(ranger) #the random forest implementation
library(plotly) #3D plotting
library(tidyr) #dataframe manipulation
library(FNN) #k nearest neighbors algorithm
library(xgboost)


fb <- fread(file="train.csv", integer64 = "character")

fb %>% filter(x >1, x <1.25, y >1, y < 1.25) -> fb_new

head(fb_new, 3)

fb_new$hour = (fb_new$time/60) %% 24
fb_new$weekday = (fb_new$time/(60*24)) %% 7
fb_new$month = (fb_new$time/(60*24*30)) %% 12 #month-ish
fb_new$year = fb_new$time/(60*24*365)
fb_new$day = (fb_new$time/(60*24)) %% 365


#fb_new %>% count(place_id) %>% filter(n > 1000) -> ids
#smallData = fb_new[fb_new$place_id %in% ids$place_id,]

small_train = fb_new[1:20000,]
small_test = fb_new[20000:24795,]

ggplot(fb_new, aes(x, y )) +
  geom_point(aes(color = place_id)) + 
  theme_minimal() +
  theme(legend.position = "none") +
  ggtitle("Check-ins colored by myid")

plot_ly(data = fb_new, x = fb_new$x , y = fb_new$y, z = fb_new$day, 
        color = fb_new$place_id,  type = "scatter3d", mode = "markers", 
        marker=list(size= 3))

ggplot(small_train, aes(x, y )) +
  geom_point(aes(color = place_id)) + 
  theme_minimal() +
  theme(legend.position = "none") +
  ggtitle("Check-ins colored by myid")

plot_ly(data = small_train, x = small_train$x , y = small_train$y, z = small_train$hour, 
        color = small_train$place_id,  type = "scatter3d", mode = "markers", 
        marker=list(size= 3))

# small_train %>% count(place_id) %>% filter(n > 400) -> ids
# small_train = small_train[small_train$place_id %in% ids$place_id,]

# plot_ly(data = small_train, x = small_train$x , y = small_train$y, z = small_train$hour, 
#         color = small_train$place_id,  type = "scatter3d", mode = "markers", 
#         marker=list(size= 3))

s = 2
l = 100
w = 500

create_matrix = function(train) {
  cbind(s*train$y,
        train$x,
        (train$hour)/l,
        (train$weekday)/w,
        (train$year)/w,
        (train$month)/w,
        (train$time)/w, 
        (train$accuracy)/w)
}

ptm <- proc.time()
myTrain=create_matrix(small_train)
myTest=create_matrix(small_test)
mycl=small_train$place_id
trueResult=small_test$place_id

model_knn=knn(train = myTrain,test=myTest,cl=mycl,k=10,algorithm = c("kd_tree"))

preds <- as.character(model_knn)
truth <- as.character(small_test$place_id)

myResult={}
myResult$preds=preds
myResult$truth=truth

myResult = function(myResult) {
  cbind(s*train$y,
        train$x,
        (train$hour)/l,
        (train$weekday)/w,
        (train$year)/w,
        (train$month)/w,
        (train$time)/w, 
        (train$accuracy)/w)
}

myTrain$Result <- apply(myResult,1,function(x){return '1' if (myResult$preds==myResult$truth))})
  
mean(truth == preds)
proc.time() - ptm



#---------------------------------------------------
#random forest
set.seed(131L)

small_train$place_id <- as.factor(small_train$place_id) # ranger needs factors for classification
model_rf <- ranger(place_id ~ x + y + hour + weekday + month + accuracy + time + year,
                   small_train,
                   num.trees = 100,
                   write.forest = TRUE,
                   importance = "impurity")


pred = predict(model_rf, small_test[-6])
prediction = pred$predictions
accuracy = mean(prediction == small_test$place_id)
accuracy

data.frame(as.list(model_rf$variable.importance)) %>% gather() %>% 
  ggplot(aes(x = reorder(key, value), y = value)) +
  geom_bar(stat = "identity", width = 0.6, fill = "black") +
  coord_flip() +
  theme_minimal() +
  ggtitle("Variable Importance (Gini Index)") +
  theme(axis.title.y = element_blank()) 

#-------------------------------------------------
#b=10


#myX=myTrain


#maxX=apply(abs(myTrain),2,max)

myX=myTrain
myY=matrix(mycl)
#scale the data

#Initial
b=100
numObj=dim(myX)[2]
b_new=matrix(0,numObj,1)
stepsize=b/(norm(t(myX)%*%myX,c("2")))

n=1
delta=1000

g=norm(myY-myX%*%b_new,c("2"))^2/(2*numObj)

g_sto=matrix(g,1,1)

rand=sample(1:numObj,b,replace = TRUE)

myX_new=myX[rand,]
myY_new=myY[rand,1]


#Iteration
while (delta>1) {
  
  grad_g=(-t(myX_new))%*%(myY_new-myX_new%*%b_new)/b
  
  b_old = b_new
  b_new=b_old-stepsize*grad_g

  g=norm(myY_new-myX_new%*%b_new,c("2"))^2/(2*b)
  
  g_sto=c(g_sto,g)
  
  n=n+1
  
  delta=abs(g_sto[n]-g_sto[n-1])
  #delta=norm(grad_g,type = c("2"))
  print(n+delta)
  rand=sample(1:numObj,b,replace=TRUE)
  
  myX_new=myX[rand,]
  myY_new=matrix(myY[rand,1])
}


dev.new(width=8,height=6)
df <- data.frame(matrix(2:n),matrix(g_sto[2:n]))
ggplot(df, aes(x = matrix(2:n), y = matrix(g_sto[2:n]))) +geom_point()+labs(title = "Stochasitc Gradient Descent for Multiple Linear Regression (b=10)")+xlab("Number of iterations")+ylab(expression(paste(g(beta))))

#accuracy
myResult=myTest%*%b_new
error=myResult-trueResult




