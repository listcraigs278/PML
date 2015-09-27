# PML


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
  download.file(url.training, destfile = "./testing.csv")
}

# load data
training <- read.csv("./training.csv", na.strings=c("NA",""),stringsAsFactors = TRUE)
testing <- read.csv("./pml-testing.csv", na.strings=c("NA",""),stringsAsFactors = TRUE)
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

