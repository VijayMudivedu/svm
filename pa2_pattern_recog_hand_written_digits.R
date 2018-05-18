#-------------------------
# Business Understanding
#-------------------------
# A classic problem in the field of pattern recognition is that of handwritten digit recognition. Suppose that you have an image of a digit submitted by a user via a scanner, a tablet, or other digital devices. The goal is to develop a model that can correctly identify the digit (between 0-9) written in an image. 
# 
# 
#-------------------------
# Objective
#-------------------------
# To develop a model using Support Vector Machine which should correctly classify the handwritten digits based on the pixel values given as features.
# 
# 
# install.packages(“caret”)
# install.packages(“kernlab”)
# install.packages(“tidyverse”)
# install.packages(“doParallel”)
# install.packages(“parallel”)
# install.packages(“gridExtra”)
# 
library(caret)
library(kernlab)
library(tidyverse)
library(reshape2)
library(doParallel)
library(parallel)

# physical cores and clusters
phy_cores <- detectCores(logical = FALSE)
phy_clust <- makeCluster(phy_cores)

doParallel::registerDoParallel(cl = phy_clust,cores = phy_cores)

# 
setwd("~/OneDrive/OneDrive - Atimi Software Inc/Upgrad/_Upgrad/PA II/Assignment")

# Reading the training dataset
train_digit_pattern <-
  read.csv(
    file = file.choose(),
    header = F,
    blank.lines.skip = T,
    skipNul = T,
    stringsAsFactors = F,
    check.names = T
  )

# reading the test dataset
test_digit_pattern <-
  read.csv(
    file.choose(),
    header = F,
    blank.lines.skip = T,
    skipNul = T,
    stringsAsFactors = F,
    check.names = T
  )

# checking the dimenions of data
dim(train_digit_pattern)
dim(test_digit_pattern)

# Images of each of Digits are converted into 28*28 pixels. Each row representing value of one each pixel. Thus there are 784 and digit label.

# renaming the column names
colnames(train_digit_pattern)[1] <- "label"

# converting the label to factor
train_digit_pattern$label <- factor(train_digit_pattern$label)
str(train_digit_pattern$label)

# summary of each of the characters of the in the digit pattern
summary(train_digit_pattern$label)

# creating the sampled dataset of 10000 samples with first 5000 (1:5000) samples and last 5000 samples.
sampled_digits_first_last_set <- train_digit_pattern[c(1:5000,55001:60000),]

dim(sampled_digits_first_last_set)


# looking the summary of digits
summary(sampled_digits_first_last_set$label)

########################################
# DATA PREPARATION
########################################

# check NAs
sum(is.na(train_digit_pattern))

# check duplicates
sum(duplicated(train_digit_pattern))

# extracting labels in the digit recognition
labels_dig_recog <- sampled_digits_first_last_set[,1]

# Training sample data without labels out the labels and finding the overiable
train_without_labels <- sampled_digits_first_last_set[,-1]

# transforming the varibles in the dataset using melt function
melted_sample_data <- sampled_digits_first_last_set %>%
  melt(id.vars = "label") %>%
  rename(pixel = variable) %>%
  extract(pixel, "pixel", "(\\d+)", convert = TRUE) %>%
  mutate(pixel = pixel - 2, id = 1:n(),
         x = pixel %% 28,
         y = 28 - pixel %% 28, dotproduct = x*y) %>% filter((dotproduct > 0))# & (x == y))
head(melted_sample_data)  

# list of values in melted sample values
unique(melted_sample_data$value)
unique(melted_sample_data$dotproduct)
unique(melted_sample_data$y)
unique(melted_sample_data$pixel)

temp_df <- as.data.frame(melted_sample_data[,c(1,3,4,5,7)])
head(temp_df)

transformed_data <-  temp_df %>% dcast(formula = label + id ~ x,value.var = "value",fill = 0 )
dim(transformed_data)
head(transformed_data)

# taking the the unique records to 
new_transformed <- transformed_data[,-2] %>% unique()

plot(colSums(x = as.data.frame(new_transformed[,-1])))
# most of the data is concetrated between pixels 5 and 20 along x and y axis.

# creating a new train by appending the labels with train dataset
train <- cbind(labels_dig_recog,train)


#------------------------------
# Exploratory Data analysis 
#------------------------------

# converting the factor to character as melt fucntions is based on factors
sampled_digits_first_last_set$label <- as.character(sampled_digits_first_last_set$label)


# dividing the melting the dataframe and 
# creating one row for each pixel
# dividing the 784 pixels into 28 X 28 pixels. 
# reshaping the dataframe into melted sampled data

head(melted_sample_data)
dim(melted_sample_data)

# Exploring the distribution of pixel data in each digit. 
melted_sample_data %>% 
  ggplot(aes(value)) +
  geom_density() +
  facet_wrap( ~ label,scales = "free") 
# exploring th
train %>% melt(id.vars = "labels_dig_recog") %>%
  ggplot(aes(value)) +
  geom_density() +
  facet_wrap( ~ labels_dig_recog,scales = "free") 


# exploring the distribution of data in each digit using
melted_sample_data %>% 
  ggplot(aes(x = pixel, y = value)) +
  geom_point() +
  facet_wrap( ~ label,scales = "free") 

# Comments - most of the data is concentrated in 0 indicating the grey matter
# part of the data is spread in 1 indicating the value of the position

# 
melted_sample_data %>% #filter(row_num <= 120000) %>%
  ggplot(aes(x,y, fill = label)) + geom_tile() +
  facet_wrap( ~ label)

# mean value distribution of pixel data for each character.
melted_sample_data_mean <- 
  melted_sample_data %>%
  group_by(label, x, y) %>%
  summarise(mean_value = mean(value)) %>% 
  ungroup()

head(melted_sample_data_mean)

# plotting the mean value of data
melted_sample_data_mean %>% 
  ggplot(aes(x, mean_value, fill=y)) + geom_point() +
  facet_wrap(~label, scales = "free_y",ncol = 2)

# Comments:  Data is normally distributed across the each digit and the data is unevenly distributed and non_linear  

melted_sample_data_mean %>% 
  ggplot(aes(x = x,y = mean_value,group = -1)) + 
  geom_boxplot(outlier.colour = "red") +
  facet_wrap(~label, scales = "free",nrow = 3)

# Comment: mean_value in label 1 has an atypical distribution indicating near zero median with several mean value outliers. Indicating the several people have distnct ways of writing a 1.


#############################################
# Dimensionality reduction techniques 
# using Principle Component Analysis (prcomp) and singular vector decomposition (svd)
#############################################
# principal component analysis
# creating a covariance matrix

principle_comp_sample <- prcomp(x = as.matrix(train_without_labels))

# checking the transformed columns and rows
head(principle_comp_sample$rotation)
head(principle_comp_sample$x)

# Computing the std deviation of each column from the PCA analysis of the training set
std_data <- principle_comp_sample$sdev

# computing the variance in the data
pr_data_var <- std_data ^ 2

# proporotion of variance in the dataset
prop_pr_varex <- pr_data_var/sum(pr_data_var)
prop_pr_varex

# plotting the cumsum of Proportionate variance and plotting the bubble plot
df = data.frame(cumulative_variance = cumsum(prop_pr_varex))

ggplot(df, aes(x = c(1:ncol(train_without_labels)), y = cumulative_variance)) + geom_point() +
  xlab(label = "Pixels") + ylab(label = "Cumulative Variance") +
  labs(title = "Cumulative variance in Pixels") +
  scale_y_continuous(breaks = round(seq(min(df$cumulative_variance),
                                        max(df$cumulative_variance),
                                        by = 0.1),1)) +
  geom_vline(xintercept = 140,linetype = "dotted") + 
  geom_text(aes(x = 140,y = 0,label = 140,vjust = 1, size = 3),show.legend = F) +
  theme(plot.title = element_text(hjust = 0.5))

# Comments: From plot we can infer that first 140 rows capture 95% of Cumulative vairance 

# picking the columns that cover the 95% of data variability
which(round(cumsum(prop_pr_varex),digits = 3) == 0.948)

#-----------------------------
# dimensionality reduction using svd() - singular value decomposition
#-----------------------------
# compute the covariance and 
cov_samp <- cov(train_without_labels)

# evaluating the svd of the covariace
samp_svd <- svd(cov_samp)

View(samp_svd)
# svd generates the number of left singluar vectors. Thus extracting only the 140 columns that capture 95% of the data
u <- samp_svd$u[,1:140]

#-----------------------------
# computing the dot.product of the train
#-----------------------------

train <- as.matrix(train_without_labels) %*% u

head(train)
# converting the train matrix to a dataframe
train <- as.data.frame(train)

# assigning the columns names of train dataframe
colnames(train) <- colnames(train_without_labels[,1:140])

# combining the labels and transformed variables
train <- cbind(labels_dig_recog,train)


#-----------------------------------
# Model Creation
#-----------------------------------

# Exploratory data analysis of the dataset demonstrates non_linear relation of 28*28 pixels for each label
model_svm_digit_recog <- ksvm(labels_dig_recog ~ . ,data = train, kernel = "polydot",C = 1)

# similar to the way how train datset was transformed, now transforming the test data
test <- as.matrix(test_digit_pattern[,-1])

# # evaluating the cov_samp and svd of test data
# cov_samp_test <- cov(test)
# svd_test <- svd(cov_samp_test)
# test_u <- svd_test$u[,1:140]

# performing dotProduct  the t(Xj) * Xi.
test <- test %*% u
# combining the records
test <- cbind(test_digit_pattern[,1],test)
test <- as.data.frame(test)
colnames(test) <- colnames(train)
head(test)

# predicted data from the model
digit_recog_pred <- predict(model_svm_digit_recog,test)

confusionMatrix(digit_recog_pred,test$labels_dig_recog)

# Overall Statistics
# 
# Accuracy : 0.9107          
# 95% CI : (0.8969, 0.9086)

####################################################
# using "rbf" kernel to verify the model accuracy
####################################################
model_svm_digi_reco_rbf <- ksvm(labels_dig_recog ~., data = train, kernel = "rbfdot")

# predicting the rbf parameters
digit_recog_pred_rbf <- predict(model_svm_digi_reco_rbf,test)

# checking the accuracy of model using
confusionMatrix(digit_recog_pred_rbf,test$labels_dig_recog)


# Overall Statistics
# 
# Accuracy : 0.9504          
# 95% CI : (0.9486, 0.9521)

# thus the accuracy of the model has considerably increased indicating non_linearity in data.


#---------------------------
# K-Cross validation to compute the hyperparameters
#---------------------------

set.seed(100)

# using k-cross validation to identify the sigma and hyper parameter cost function c
train_control_digit_recog <- trainControl(method = "cv",number = 3) # using 3 fold cross validata

# using the cost function dataframe. tuning grid
# tuning grid of different values of sigma and hyperparamet c
tuning_grid_digit_recog <- expand.grid(.sigma = seq(0,0.03,0.01),.C = seq(1,3,1))
tuning_grid_digit_recog


# bringing the number of samples down from 10000 to 1000 random samples to reduce the computations to get a feel of hyper parameters

train_sample <- train[sample(1:nrow(train),0.1*nrow(train)),]

# running the non-linear train function to identify optimized C the values for which
non_linear_fit_digit_recog <-  train(labels_dig_recog ~ ., 
                                     data = train_sample, 
                                     method = "svmRadial",
                                     trControl = train_control_digit_recog,
                                     tuneGrid = tuning_grid_digit_recog,
                                     metric = "Accuracy")

non_linear_fit_digit_recog

# Comments: 
# Accuracy was used to select the optimal model using the largest value of Accuracy and Kappa.
# The final values used for the model were sigma = 0.01 and C = 2.

# sigma  C  Accuracy   Kappa    
# 0.00   1  0.1120009  0.0000000
# 0.00   2  0.1120009  0.0000000
# 0.00   3  0.1120009  0.0000000
# 0.01   1  0.8240478  0.8042497
# 0.01   2  0.8330391  0.8142702
# 0.01   3  0.8330391  0.8142702
# 0.02   1  0.6499708  0.6096625


# re-running the svm with hyper parameters
model_svm_digi_reco_rbf_sigma <-
  ksvm(
    labels_dig_recog ~ .,
    data = train,
    kernel = "rbfdot",
    kpar = list(sigma = 0.01),
    C = 2
  )

# evaluating the predictions on the test data
predictions_digit_test <- predict(model_svm_digi_reco_rbf_sigma,test)

# Evaluating the accuracy of the model
confusionMatrix(predictions_digit_test,test$labels_dig_recog)

# Accuracy : 0.9532          
# 95% CI : (0.9489, 0.9573)

# Class: 0 Class: 1 Class: 2 Class: 3 Class: 4 Class: 5 Class: 6 Class: 7
# Sensitivity            0.9755   0.9850   0.9545   0.9337   0.9664   0.9406   0.9645   0.9329
# Specificity            0.9975   0.9986   0.9897   0.9944   0.9950   0.9962   0.9966   0.9970

# Class: 8 Class: 9
# Sensitivity            0.9548   0.9207
# Specificity            0.9878   0.9952

##################
# CONCLUSIONS
#####################

# Thus the pattern optimises with an Accuracy = 95% and a small non_linearity identified by radial basis function kernel, fits the test data accurately, Identifying a character from the unseen data with an accuracy of 95.5%




