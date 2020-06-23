library(readr)
library(dplyr)
library(keras)
library(tidyr)
library(beepr)
library(rlist)
library(audio)
library(fs)

# functions ####
quick_model <- function(dropout = FALSE){
# using default activation--hyperbolic tangent, tanh
  if (dropout){
    model <- keras_model_sequential() %>%
      layer_lstm(10, batch_input_shape = c(64,32,1),stateful = TRUE, return_sequences = TRUE) %>%
      layer_dropout(0.2) %>%
      layer_lstm(10, batch_input_shape = c(64,32,1),stateful = TRUE, return_sequences = TRUE) %>%
      layer_dropout(0.2) %>%
      layer_dense(1)
  }
  else{
    model <- keras_model_sequential() %>%
      layer_lstm(10, batch_input_shape = c(64,32,1),stateful = TRUE, return_sequences = TRUE) %>%
      layer_lstm(10, batch_input_shape = c(64,32,1),stateful = TRUE, return_sequences = TRUE) %>%
      layer_dense(1)
  }
  return(model)
}

break_iteration <- function ( model, train_X, train_y, val_batches = 0, batch_size = 64, epochs = 60, noisy = FALSE, ...){
  # model for tensorflow is effectively a pointer, and fit has side effects.  Model will be changed without returning it
  hist<-model %>%
    compile(
      loss = 'mse',
      optimizer = 'adam'
    ) %>%
    fit(
      epochs = epochs,
      batch_size = batch_size,
      x = train_X,
      y = train_y,
      validation_split = (val_batches / train_batches),
      ...
    )
  if (noisy){
    beep(sound = 11)
  }
  return(hist)
}



# load_and_shape_data ####
oil <- read_csv('C:\\Users\\Christopher\\Desktop\\Data\\oil_prices.csv')

oil <- oil %>%
  rename(price = DCOILBRENTEU) %>%
  mutate(price = as.numeric(price)) %>%
  mutate(change64_days = lag(price, 64) ) %>%
  drop_na() %>%
  mutate (change64_days = price - change64_days )
oil <- oil %>%
  mutate (id = 1:nrow(oil) )

# create past/future ####
timesteps = 32
batch_size = 64
epochs = 60
past <- NULL
future <- NULL
# indexing to avoid NAs in past data; could do same for future, but that would eliminate data for real predictions
for (i in 1:timesteps){
  past <- c(past,unlist(lag( oil$price, i)))
  future <- c(future,unlist(lead( oil$price, i-1)))
}
past <-   array( data =   past, dim = c(NROW(oil), 32, 1) )
future <- array( data = future, dim = c(NROW(oil), 32, 1) )

clip = timesteps + 1 + ((dim(future)[1] - timesteps) %% batch_size)

future <- future[clip:dim(future)[1],,1]
past   <- past  [clip:dim(  past)[1],,1]

# scaling ####
if ( max(past) >1 || min(past) < 0 ){
  future <- ( future - min ( oil$price ) ) / ( max( oil$price - min ( oil$price ) ) )
  past <- ( past - min ( oil$price ) ) / (max( oil$price - min( oil$price ) ) )
}
future <- array(future,c(dim(future)[1], dim(future)[2], 1))
past <- array(past,c(dim(past)[1], dim(past)[2], 1))

# batch_parameters ####
train_share <- 0.90
batches <- dim(future)[1] / batch_size
train_batches <- as.integer((batches - 1) * 0.9)
test_batches <- batches - 1 - train_batches
train_end <- train_batches * batch_size
test_start <- train_end + 1
test_end <- train_end + test_batches * batch_size
future_start <- test_end +1
future_end <- dim(future)[2]

# create models
# model_11v <- quick_model(dropout = TRUE)
# model_2v <- quick_model(dropout = TRUE)
# simple_model <- quick_model()
# simple_model_2v <- quick_model()
# 
# # create_training_data ####
# train_X <- past[1:train_end,,]
# train_X <- array(train_X,c(dim(train_X)[1], dim(train_X)[2], 1))
# train_y <- future[1:train_end,,]
# train_y <- array(train_y,c(dim(train_y)[1], dim(train_y)[2], 1))
# 
# # train_models ####
# # setting verbose = 0, as otherwise the output is excessive.  Using beepr to give audible progress update
# m11v_hist<-model_11v %>% 
#   compile(
#     loss = 'mse',
#     optimizer = 'adam'
#   ) %>%
#   fit(
#     epochs = epochs,
#     batch_size = batch_size,
#     x = train_X,
#     y = train_y,
#     validation_split = (11 / 111),
#     verbose = 0,
#   )
# beep(11)
# m2v_hist<-model_2v %>%
#   compile(
#     loss = 'mse',
#     optimizer = 'adam'
#   ) %>%
#   fit(
#     epochs = epochs,
#     batch_size = batch_size,
#     x = train_X,
#     y = train_y,
#     validation_split = (2 / 111),
#     verbose = 0
#     #    validation_data = c(test_X, test_y)
#   )
# beep(11)
# sm_hist<-simple_model %>% 
#   compile(
#     loss = 'mse',
#     optimizer = 'adam'
#   ) %>%
#   fit(
#     epochs = epochs,
#     batch_size = batch_size,
#     x = train_X,
#     y = train_y,
#     verbose = 0
#   )
# beep(11)
# sm2v_hist<-simple_model_2v %>%
#   compile(
#     loss = 'mse',
#     optimizer = 'adam'
#   ) %>%
#   fit(
#     epochs = epochs,
#     batch_size = batch_size,
#     x = train_X,
#     y = train_y,
#     validation_split = 2 / 111,
#     verbose = 0
#   )
# beep(8)
# # add_center_to_history_lists ####
# m11v_hist <- c(m11v_hist, center = test_start)
# m2v_hist <- c(m2v_hist, center = test_start)
# sm_hist <- c(sm_hist, center = test_start)
# sm2v_hist <- c(sm2v_hist, center = test_start)

# find_worst_model ####
divide = 5665
step = 64
train_batches = as.integer(divide / batch_size)
test_batches = 100 - train_batches

dr = divide + step # divide right, or in the first case, center

start <- dr - batch_size * train_batches
end <- dr + test_batches * batch_size - 1

train_X <- past[start:(dr - 1),,]
train_X <- array(train_X,c(dim(train_X)[1], dim(train_X)[2], 1))

train_y <- future[start:(dr - 1),,]
train_y <- array(train_y,c(dim(train_y)[1], dim(train_y)[2], 1))

test_X <- past[dr: end,,]
test_X <- array(test_X,c(dim(test_X)[1], dim(test_X)[2], 1))

test_y <- future[dr:end,,]
test_y <- array(test_y,c(dim(test_y)[1], dim(test_y)[2], 1))


mr <- quick_model()
mr_hist <- mr %>%
  break_iteration( train_X, train_y, verbose = 0, noisy = TRUE)
center_loss <- mr %>%
  evaluate(batch_size = 64, test_X, test_y, verbose = 0)
worst <- mr

mr_hist <- c(mr_hist, center = dr)
worst_hist <- mr_hist

old_divide <- dr
step <- step / 2
while (step >= 1){
  # set divide left and divide right
  dl <- old_divide - step
  dr <- old_divide + step
  
  # check left
  start <- dl - batch_size * train_batches
  end <- dl + test_batches * batch_size -1
  
  train_X <- past[start:(dl - 1),,]
  train_X <- array(train_X,c(dim(train_X)[1], dim(train_X)[2], 1))
  
  train_y <- future[start:(dl - 1),,]
  train_y <- array(train_y,c(dim(train_y)[1], dim(train_y)[2], 1))
  
  test_X <- past[dl: end,,]
  test_X <- array(test_X,c(dim(test_X)[1], dim(test_X)[2], 1))
  
  test_y <- future[dl:end,,]
  test_y <- array(test_y,c(dim(test_y)[1], dim(test_y)[2], 1))
  ml <- quick_model()
  ml_hist <- ml %>%
    break_iteration( train_X, train_y, verbose = 0, noisy = TRUE )
  ml_hist <- c(ml_hist, center = dl)
  left_loss <- ml %>%
    evaluate(batch_size = 64,test_X, test_y)
  
  # check right
  start <- dr - batch_size * train_batches
  end <- dr + test_batches * batch_size -1
  
  train_X <- past[start:(dr - 1),,]
  train_X <- array(train_X,c(dim(train_X)[1], dim(train_X)[2], 1))
  
  train_y <- future[start:(dr - 1),,]
  train_y <- array(train_y,c(dim(train_y)[1], dim(train_y)[2], 1))
  
  test_X <- past[dr: end,,]
  test_X <- array(test_X,c(dim(test_X)[1], dim(test_X)[2], 1))
  
  test_y <- future[dr:end,,]
  test_y <- array(test_y,c(dim(test_y)[1], dim(test_y)[2], 1))
  
  mr <- quick_model()
  mr_hist <- mr %>%
    break_iteration( train_X, train_y, verbose = 0, noisy = TRUE )
  mr_hist <- c(mr_hist, center = dr)
  right_loss <- mr %>%
    evaluate(batch_size = 64,test_X, test_y)
  
  # compare and updade
  losses <- c(center_loss, left_loss, right_loss)
  if (  left_loss == max(losses)  ){
    old_divide <- dl
    center_loss <- left_loss
    worst <- ml
    worst_hist <- ml_hist
  }
  else if (  right_loss == max(losses)  ){
    old_divide <- dr
    center_loss <- right_loss
    worst <- mr
    worst_hist <- mr_hist
  }
  step <- step / 2
}
beep(sound = 3)
wait(3)
beep(sound = 4)


# save models ####
# target_dir <- path_wd('R','Breaking_ML_w_R','Data','model_11v', ext = 'hdf5')
# model_11v %>% save_model_hdf5(target_dir)
# target_dir <- path_wd('R','Breaking_ML_w_R','Data','model_2v', ext = 'hdf5')
# model_2v %>% save_model_hdf5(target_dir)
# target_dir <- path_wd('R','Breaking_ML_w_R','Data','simple_model', ext = 'hdf5')
# simple_model %>% save_model_hdf5(target_dir)
# target_dir <- path_wd('R','Breaking_ML_w_R','Data','simple_model_2v', ext = 'hdf5')
# simple_model_2v %>% save_model_hdf5(target_dir)

target_dir <- path_wd('R','Breaking_ML_w_R','Data','worst_model', ext = 'hdf5')
worst %>% save_model_hdf5(target_dir)


# save histories ####
# target_dir <- path_wd('R','Breaking_ML_w_R','Data','m11v_hist', ext = 'rdata')
# list.save(m11v_hist, target_dir)
# target_dir <- path_wd('R','Breaking_ML_w_R','Data','m2v_hist', ext = 'rdata')
# list.save(m2v_hist, target_dir)
# target_dir <- path_wd('R','Breaking_ML_w_R','Data','sm_hist', ext = 'rdata')
# list.save(sm_hist, target_dir)
# target_dir <- path_wd('R','Breaking_ML_w_R','Data','sm2v_hist', ext = 'rdata')
# list.save(sm2v_hist, target_dir)

target_dir <- path_wd('R','Breaking_ML_w_R','Data','wm_hist', ext = 'rdata')
list.save(worst_hist, target_dir)
