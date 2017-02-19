## ==============================
## Load packages
## ==============================

## These packages are available from CRAN

library(jpeg)
library(randomForest)
library(readr)
rm(list = ls())
gc()

## ==============================
## Main variables
## ==============================
setwd('E:/02.DS competitions/Yelp/R')
## Set the path to your data here
data_dir <- "../input"
train_data_dir <- paste(data_dir,"/train_photos", sep="")
test_data_dir <- paste(data_dir,"/test_photos", sep="")

## ==============================
## Define Functions
## ==============================

## Handy function to display a greyscale image of the plankton
im <- function(image) image(image, col = grey.colors(32))


## Function to extract some simple statistics about the image
## UPDATE THIS to do any calculations that you think are useful
extract_stats <- function(working_image = working_image){
  im_length <- nrow(working_image)
  im_width <- ncol(working_image)
  im_density <- mean(working_image)
  im_ratio <- im_length / im_width 
  im_df = as.data.frame(cbind(length=im_length,width=im_width,density=im_density,ratio=im_ratio))
  return(im_df)
#   #Invert the image to calculate density
#   statsR = summary(as.vector(working_image[,,1]))
#   names(statsR) = paste0(names(statsR),'R')
#   statsG = summary(as.vector(working_image[,,2]))
#   names(statsG) = paste0(names(statsG),'G')
#   statsB = summary(as.vector(working_image[,,3]))
#   names(statsB) = paste0(names(statsB),'B')
#   return(as.data.frame(c(im_df,statsR,statsG,statsB)))
  
#   nBins = 10
#   histR =hist(working_image[,,1],breaks=seq(0,1,1/nBins),plot = F)$density/nBins
#   names(histR) = paste0('histR_',1:nBins)
#   histG =hist(working_image[,,2],breaks=seq(0,1,1/nBins),plot = F)$density/nBins
#   names(histG) = paste0('histG_',1:nBins)
#   histB =hist(working_image[,,3],breaks=seq(0,1,1/nBins),plot = F)$density/nBins
#   names(histB) = paste0('histB_',1:nBins)
#   return(as.data.frame(c(im_df,statsR,statsG,statsB,histR,histG,histB)))

}

## Function to calculate multi-class loss on train data
mcloss <- function (y_actual, y_pred) {
  dat <- rep(0, length(y_actual))
  for(i in 1:length(y_actual)){
    dat_x <- y_pred[i,y_actual[i]]
    dat[i] <- min(1-1e-15,max(1e-15,dat_x))
  }
  return(-sum(log(dat))/length(y_actual))
}

## ==============================
## Read training data
## ==============================

## Read all the image files and calculate training statistics
# Get list of all the examples of this classID
# train_file_list <- list.files(train_data_dir)
# test_file_list <- list.files(test_data_dir)
# save(train_file_list,test_file_list, file = '../input/file_list.RData')
load('../input/file_list.RData')

feature_engineering <- function(feature_file_name,file_dir, file_list){
  feature_data <- c()
  idx <- 1
  startT <- Sys.time()
  #Read and process each image
  cat('Read and process each image\n')
  for(fileID in file_list){
    working_file <- paste(file_dir,"/",fileID,sep="")
    working_image <- readJPEG(working_file)
    
    #View image
  #   if (exists("rasterImage")) { # can plot only in R 2.11.0 and higher
  #     temp = max(ncol(working_image),nrow(working_image))
  #     plot(0:temp,0:temp, type='n')
  #     rasterImage(working_image, 0, 0, nrow(working_image), ncol(working_image))
  #   }
    
    # Calculate model statistics
    ## YOUR CODE HERE ##
    working_stats <- extract_stats(working_image)
    working_summary <- data.frame(fileName = fileID,working_stats)
    if(is.null(feature_data)){
      feature_data <- data.frame(temp = rep("",length(file_list)), working_summary, stringsAsFactors = FALSE)
      feature_data$temp = NULL
    }else{
      feature_data[idx,] <- working_summary
    }
    idx <- idx + 1
    if(idx %% 1000 ==0){
      cat('Processed ', idx, ', time: ',Sys.time() - startT,'\n')
      if(idx %% 10000 ==0){
        save(feature_data,file = paste0('../input/', feature_file_name,'.RData'))
      }
    }
  }
  save(feature_data,file = paste0('../input/', feature_file_name,'.RData'))
  cat("Finished processing train data", '\n')
  return(feature_data)
}
feature_data = feature_engineering('train_feature',train_data_dir,train_file_list)
feature_data = feature_engineering('test_feature',test_data_dir,test_file_list)

## Get labels
train_photo_to_biz_id = read_csv('../input/train_photo_to_biz_ids.csv')
test_photo_to_biz_id = read_csv('../input/test_photo_to_biz_ids.csv')
length(unique(train_photo_to_biz_id$business_id))
length(unique(test_photo_to_biz_id$business_id))
train = read_csv('../input/train.csv')

train_file_num = as.numeric(gsub("([0-9]*).*","\\1",train_file_list))
test_file_num = as.numeric(gsub("([0-9]*).*","\\1",test_file_list))
length(intersect(train_file_num,train_photo_to_biz_id$photo_id))
length(intersect(test_file_num,test_photo_to_biz_id$photo_id))
test_photo_to_biz_id = test_photo_to_biz_id[which(test_photo_to_biz_id$photo_id %in% test_file_num),]








## ==============================
## Create Model
## ==============================

## We need to convert class to a factor for randomForest
## so we might as well get subsets of x and y data for easy model building
y_dat <- as.factor(train_data$class)
x_dat <- train_data[,3:6]

plankton_model <- randomForest(y = y_dat, x = x_dat)



# Compare importance of the variables
importance(plankton_model)


## Check overall accuracy... 24%, not very good but not bad for a simplistic model
table(plankton_model$predicted==y_dat)
#  FALSE  TRUE 
#  22959  7377

## Make predictions and calculate log loss
y_predictions <- predict(plankton_model, type="prob")

ymin <- 1/1000
y_predictions[y_predictions<ymin] <- ymin

mcloss(y_actual = y_dat, y_pred = y_predictions)
# 3.362268


## ==============================
## Read test data and make predictions
## ==============================



## Read all the image files and calculate training statistics
## This should take about 10 minutes, with speed limited by IO of the thousands of files

# Get list of all the examples of this classID
test_file_list <- list.files(paste(test_data_dir,sep=""))
test_cnt <- length(test_file_list)
test_data <- data.frame(image = rep("a",test_cnt), lenght=0,width=0,density=0,ratio=0, stringsAsFactors = FALSE)
idx <- 1
#Read and process each image
for(fileID in test_file_list){
  working_file <- paste(test_data_dir,"/",fileID,sep="")
  working_image <- readJPEG(working_file)
  
  # Calculate model statistics
  
  ## YOUR CODE HERE ##
  working_stats <- extract_stats(working_image)
  working_summary <- array(c(fileID,working_stats))
  test_data[idx,] <- working_summary
  idx <- idx + 1
  if(idx %% 10000 == 0) cat('Finished processing', idx, 'of', test_cnt, 'test images', '\n')
}


## Make predictions with class probabilities
test_pred <- predict(plankton_model, test_data, type="prob")
test_pred[test_pred<ymin] <- ymin

## ==============================
## Save Submission File
## ==============================

## Combine image filename and class predictions, then save as csv
submission <- cbind(image = test_data$image, test_pred)
submission_filename <- paste(data_dir,"/submission03.csv",sep="")
write.csv(submission, submission_filename, row.names = FALSE)