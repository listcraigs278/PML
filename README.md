# PML

# Introduction
This project aims to predict the activity of subjects wearing activity band devices such as Jawbone Up, Nike FuelBand, and Fitbit. Specifically, we exploit various machine learning algorithms in order to predict subject performing weight-lifting exercises.

The data utilized consists of accelerometer readings corresponding to six participants. These readings are taken from sensors that are worn on the belt, forearm, and on the dumbbell. Our goal is to predict, given a set of readings the manner in which an individual is performing the exercise. 

# Methodology
Preliminary data collection and preparation steps are described in the appendices, below. We employ four machine learning algorithms on our training data then compare the performance of these algorithms in order to select the best algorithm for this task. For our chosen algorithm, we further perform cross validation and evaluate the out of sample error. The algorithms employed are: 1) Classification Trees; 2) Bootstrap Aggregation; Random Forests; and Boosting.

The model fitting for these algorithms is presented in Appendix VI.


# Conclusion
In this project, a predicting model was created using Random Forests classification method. Our model was considerably accurate **98.9%** with a CI of 98.59% and 99.15%. Our out of sample error rate was: **0.64%**. When our model was applied to the given testing data, we achieved a prediction accuracy of 100% on the 20 problems.

# Appendix I: List of packages used
library(knitr)
library(caret)
library(randomForest)
library(rpart)
library(rattle)
library(Hmisc)
library(plyr)

# Appendix II: Loading the data

```r
# Create and/or set directory 
setwd("C:/")
if(!file.exists("C:/PMLProject")){
  dir.create("C:/PMLProject")
}
setwd("C:/PMLProject")

# download training data...
if(!file.exists("./training.csv")){
  url.training <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
  download.file(url.training, destfile = "./training.csv")
}

# download testing data...
if(!file.exists("./testing.csv")){
  url.testing <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
  download.file(url.testing, destfile = "./testing.csv")
}

# load data
training <- read.csv("./training.csv", na.strings=c("NA",""),stringsAsFactors = TRUE)
testing <- read.csv("./testing.csv", na.strings=c("NA",""),stringsAsFactors = TRUE)
```
# Appendix III: Data Cleaning
```r
# We remove any columns that contain calculated/computed variables...
variables <- names(training)
excluded <- c( grep("X", variables),
                     grep("new_window", variables),
                     grep("num_window", variables),
                     grep("avg", variables),
                     grep("min", variables),
                     grep("max", variables),
                     grep("stddev", variables),
                     grep("var", variables),
                     grep("total", variables),
                     grep("timestamp", variables),
                     grep("user_name", variables))
training <- training[ , variables[-excluded]]

variables <- names(testing)

testing <- testing[ , variables[-excluded] ]

# filter out columns that contain too many NA values...
NAColumns <- apply(!is.na(training), 2, sum)>3000
training <- training[,NAColumns]
testing <- testing[,NAColumns]
```
# Appendix V: Data Splitting
Our clean training data set is further split into a **training** and **testing** set with the ratio of (70/30).
```r
set.seed(12345) # For reproducibile purpose
inTrain <- createDataPartition(training$classe,p=0.7, list=F)
subTraining <- training[inTrain, ]
subTesting <- training[-inTrain, ]
```

# Appendix VI: Model Fitting for our Chosen Algorithms
In this Appendix, we describe the steps we took to fit our training data to the various Machine Learning Algorithms.

# Classification trees
```r
class_tree_fit<-train(classe ~ ., method="rpart",data=subTraining)
print(class_tree_fit$finalModel)
class_tree_fit

class_tree_results<- confusionMatrix(subTesting$classe, predict(class_tree_fit, newdata=subTesting))
class_tree_results

Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 1494   21  128    0   31
         B  470  380  289    0    0
         C  467   29  530    0    0
         D  438  184  342    0    0
         E  141  147  277    0  517

Overall Statistics
                                          
               Accuracy : 0.4963          
                 95% CI : (0.4835, 0.5092)
    No Information Rate : 0.5115          
    P-Value [Acc > NIR] : 0.9902          
                                          
                  Kappa : 0.3425          
 Mcnemar's Test P-Value : NA              

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            0.4963  0.49934  0.33844       NA  0.94343
Specificity            0.9374  0.85187  0.88516   0.8362  0.89414
Pos Pred Value         0.8925  0.33363  0.51657       NA  0.47782
Neg Pred Value         0.6400  0.91972  0.78679       NA  0.99355
Prevalence             0.5115  0.12931  0.26610   0.0000  0.09312
Detection Rate         0.2539  0.06457  0.09006   0.0000  0.08785
Detection Prevalence   0.2845  0.19354  0.17434   0.1638  0.18386
Balanced Accuracy      0.7169  0.67561  0.61180       NA  0.91878
```
# Bootstrap Aggregating
```r
bootstrap_aggregating_fit<-train(classe ~ ., method="treebag",data=subTraining)
print(bootstrap_aggregating_fit$finalModel)
bootstrap_aggregating_fit

boostrap_results<- confusionMatrix(subTesting$classe, predict(bootstrap_aggregating_fit, newdata=subTesting))
boostrap_results

Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 1668    5    1    0    0
         B   16 1114    9    0    0
         C    1    7 1004   11    3
         D    1    0   15  946    2
         E    0    2    3    7 1070

Overall Statistics
                                          
               Accuracy : 0.9859          
                 95% CI : (0.9825, 0.9888)
    No Information Rate : 0.2865          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.9822          
 Mcnemar's Test P-Value : NA              

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            0.9893   0.9876   0.9729   0.9813   0.9953
Specificity            0.9986   0.9947   0.9955   0.9963   0.9975
Pos Pred Value         0.9964   0.9781   0.9786   0.9813   0.9889
Neg Pred Value         0.9957   0.9971   0.9942   0.9963   0.9990
Prevalence             0.2865   0.1917   0.1754   0.1638   0.1827
Detection Rate         0.2834   0.1893   0.1706   0.1607   0.1818
Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
Balanced Accuracy      0.9939   0.9912   0.9842   0.9888   0.9964
```
## Random Forests
```r
random_forest_fit<-train(classe ~ ., method="rf",data=subTraining, prox=TRUE)
print(random_forest_fit$finalModel)
random_forest_fit

random_forest_results<- confusionMatrix(subTesting$classe, predict(random_forest_fit, newdata=subTesting))
random_forest_results

Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 1673    1    0    0    0
         B   11 1123    5    0    0
         C    0   16 1006    4    0
         D    0    0   24  940    0
         E    0    0    0    4 1078

Overall Statistics
                                          
               Accuracy : 0.989           
                 95% CI : (0.9859, 0.9915)
    No Information Rate : 0.2862          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.986           
 Mcnemar's Test P-Value : NA              

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            0.9935   0.9851   0.9720   0.9916   1.0000
Specificity            0.9998   0.9966   0.9959   0.9951   0.9992
Pos Pred Value         0.9994   0.9860   0.9805   0.9751   0.9963
Neg Pred Value         0.9974   0.9964   0.9940   0.9984   1.0000
Prevalence             0.2862   0.1937   0.1759   0.1611   0.1832
Detection Rate         0.2843   0.1908   0.1709   0.1597   0.1832
Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
Balanced Accuracy      0.9966   0.9909   0.9839   0.9933   0.9996
```
## Boosting
```r
boosting_fit<-train(classe ~ ., method="gbm",data=subTraining, verbose=FALSE)
print(boosting_fit$finalModel)
boosting_fit

boosting_results<- confusionMatrix(subTesting$classe, predict(boosting_fit, newdata=subTesting))
boosting_results

Stochastic Gradient Boosting 

13737 samples
   48 predictor
    5 classes: 'A', 'B', 'C', 'D', 'E' 

No pre-processing
Resampling: Bootstrapped (25 reps) 
Summary of sample sizes: 13737, 13737, 13737, 13737, 13737, 13737, ... 
Resampling results across tuning parameters:

  interaction.depth  n.trees  Accuracy   Kappa      Accuracy SD  Kappa SD   
  1                   50      0.7539884  0.6880180  0.005014978  0.006321792
  1                  100      0.8186104  0.7704377  0.005998331  0.007505360
  1                  150      0.8491977  0.8091544  0.005906796  0.007385037
  2                   50      0.8529400  0.8136475  0.004343612  0.005432992
  2                  100      0.9030051  0.8772119  0.003939703  0.004931615
  2                  150      0.9276492  0.9084192  0.003479400  0.004361568
  3                   50      0.8948549  0.8668563  0.004217731  0.005267138
  3                  100      0.9380099  0.9215398  0.004283749  0.005373514
  3                  150      0.9578001  0.9465937  0.003150472  0.003958739

Tuning parameter 'shrinkage' was held constant at a value of 0.1
Tuning parameter 'n.minobsinnode' was held constant at a value of 10
Accuracy was used to select the optimal model using  the largest value.
The final values used for the model were n.trees = 150, interaction.depth = 3, shrinkage = 0.1 and n.minobsinnode = 10.
```
