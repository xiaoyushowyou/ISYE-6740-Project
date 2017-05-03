rm(list = ls())
#------------ Library to be used ------------#
library(data.table)
library(dplyr)
library(ggplot2)
library(ranger)
library(plotly)
library(tidyr)
library(FNN)
library(xgboost)

#------------ Read and Filter ------------#
fb <- fread("train6.csv", integer64 = "character")
fb %>% filter(x >1, x <1.25, y >1, y < 1.25) -> fb1
head(fb, 3)

#------ filter for 3d vis ------#
fb1 %>% count(place_id) %>% filter(n > 500) -> ids
fb2 <- fb1[fb1$place_id %in% ids$place_id,]
length(unique(fb2$place_id))

#------ 2D Visualize ------#
ggplot(fb1, aes(x, y)) + geom_point(aes(color = place_id)) + theme_minimal() + theme(legend.position = "none") + ggtitle("Check-ins colored by place_id")

#------ 3D Visualize ------#
plot_ly(data = fb2, x = fb2$x , y = fb2$y, z = fb2$hour, color = fb2$place_id,  type = "scatter3d", mode = "markers", marker=list(size= 5)) %>% layout(title = "Place_id's by position and Time of Day")
plot_ly(data = fb2, x = fb2$x , y = fb2$y, z = fb2$weekday, color = fb2$place_id,  type = "scatter3d", mode = "markers", marker=list(size= 5)) %>% layout(title = "Place_id's by position and Day of Week")
plot_ly(data = fb2, x = fb2$x , y = fb2$y, z = fb2$accuracy, color = fb2$place_id,  type = "scatter3d", mode = "markers", marker=list(size= 5)) %>% layout(title = "Place_id's by position and Day of Week")


#============ kNN ============#

s = 2
l = 100
w = 500
fb_train <- fb1[1:22315, ]
fb_test  <- fb1[22316:24795, ]

#------ to choose appropriate k (and/or s, l, w) ------#
rate <- c()
for(kk in seq(1, 30, 1)) {
  create_matrix = function(train) {
    cbind(train$x,
          s*train$y,
          train$accuracy/w,
          train$hour/l,
          train$weekday/w,
          train$month/w)
  }
  X = create_matrix(fb_train)
  X_val = create_matrix(fb_test)
  model_knn = FNN::knn(train = X, test = X_val, cl = fb_train$place_id, k = kk)
  preds <- as.character(model_knn)
  truth <- as.character(fb_test$place_id)
  rate <- c(rate, mean(truth == preds))
}
plot(seq(1, 30, 1),rate, xlab = "Number of neighbours", ylab = "testing accuracy")
title("Exploration of perfect number of neighbours")


#----- kNN with k=10 ------#
create_matrix = function(train) {
  cbind(train$x,
        s*train$y,
        train$accuracy/w,
        train$hour/l,
        train$weekday/w,
        train$month/w)
}
X = create_matrix(fb_train)
X_val = create_matrix(fb_test)
# startTime = proc.time()
model_knn = FNN::knn(train = X, test = X_val, cl = fb_train$place_id, k = 10)
# endTime = proc.time()
# endTime - startTime
preds <- as.character(model_knn)
truth <- as.character(fb_test$place_id)
mean(truth == preds)

compare <- c()
for(i in c(1:2480)) {
  if(preds[i] == truth[i]) {
    compare <- c(compare, fb_test$place_id[i])
  } else {
    compare <- c(compare, 0)
  }
}
fb_test$correction <- compare

#------ 2d Vis Correction ------#
ggplot(fb_test, aes(x, y)) + geom_point(aes(color = correction)) + theme_minimal() + theme(legend.position = "none") + ggtitle("Check-ins colored by correction")

#------ 3d Vis Correction ------#
plot_ly(data = fb_test, x = fb_test$x , y = fb_test$y, z = fb_test$hour, color = fb_test$correction,  type = "scatter3d", mode = "markers", marker=list(size= 4)) %>% layout(title = "Place_id's by correction and Time of Day")

#------ 3d Vis Correction w/ filters ------#
fb_test %>% count(place_id) %>% filter(n > 50) -> ids
fb_test2 <- fb_test[fb_test$place_id %in% ids$place_id,]
plot_ly(data = fb_test2, x = fb_test2$x , y = fb_test2$y, z = fb_test2$hour, color = fb_test2$correction,  type = "scatter3d", mode = "markers", marker=list(size= 4)) %>% layout(title = "Place_id's by correction and Time of Day")

