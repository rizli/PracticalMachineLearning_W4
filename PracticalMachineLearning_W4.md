---
title: "Predictive Model Assignment"
author: "Rizli Anshari"
date: "August 18, 2019"
output:
  html_document:
    keep_md: yes
---



## Synopsis
The goal of this project is to predict the manner in which type of exercise they performed. This is the "classe" variable in the training set. 

## Loading and Processing the Data

```r
train_url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
test_url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
training <- read.csv(url(train_url),header=TRUE)
testing  <- read.csv(url(test_url),header=TRUE)
```

## Build a split between Training and Testing data set

```r
inTrain  <- createDataPartition(training$classe, p=0.7, list=FALSE)
TrainingSet <- training[inTrain, ]
TestingSet  <- training[-inTrain, ]
dim(TrainingSet)
```

```
## [1] 13737   160
```

```r
dim(TestingSet)
```

```
## [1] 5885  160
```

## Cleaning the data
# remove variables with Nearly Zero Variance

```r
NZV1 <- nearZeroVar(TrainingSet)
NZV2 <- nearZeroVar(TestingSet)
TrainingSet <- TrainingSet[, -NZV1]
TestingSet  <- TestingSet[, -NZV2]
dim(TrainingSet)
```

```
## [1] 13737   104
```

```r
dim(TestingSet)
```

```
## [1] 5885  117
```

```r
# remove variables with mostly NA
AllNA1    <- sapply(TrainingSet, function(x) mean(is.na(x))) > 0.95
AllNA2    <- sapply(TestingSet, function(x) mean(is.na(x))) > 0.95
TrainingSet <- TrainingSet[, AllNA1==FALSE]
TestingSet  <- TestingSet[, AllNA2==FALSE]
dim(TrainingSet)
```

```
## [1] 13737    59
```

```r
dim(TestingSet)
```

```
## [1] 5885   59
```

## Exclude Column 1 to 5

```r
TrainingSet <- TrainingSet[, -(1:5)]
TestingSet  <- TestingSet[, -(1:5)]
dim(TrainingSet)
```

```
## [1] 13737    54
```

```r
dim(TestingSet)
```

```
## [1] 5885   54
```

## Prediction Model
## Method 1: Random Forest
Cross-validation is a resampling procedure used to evaluate machine learning models on a limited data sample. In this project we use k-fold = 3.

```r
controlRF <- trainControl(method="cv", number=3, verboseIter=FALSE)
modelRandomForest <- train(classe ~ ., data=TrainingSet, method="rf", trControl=controlRF)
modelRandomForest$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 0.23%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3904    1    0    0    1 0.0005120328
## B    7 2649    2    0    0 0.0033860045
## C    0    5 2391    0    0 0.0020868114
## D    0    0   11 2241    0 0.0048845471
## E    0    0    0    4 2521 0.0015841584
```

```r
# prediction on Test dataset
predictRandomForest <- predict(modelRandomForest, newdata=TestingSet)
confusionMatrixRandomForest <- confusionMatrix(predictRandomForest, TestingSet$classe)
confusionMatrixRandomForest
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    3    0    0    0
##          B    0 1135    4    0    0
##          C    0    0 1022    4    0
##          D    0    1    0  959    0
##          E    0    0    0    1 1082
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9978          
##                  95% CI : (0.9962, 0.9988)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9972          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9965   0.9961   0.9948   1.0000
## Specificity            0.9993   0.9992   0.9992   0.9998   0.9998
## Pos Pred Value         0.9982   0.9965   0.9961   0.9990   0.9991
## Neg Pred Value         1.0000   0.9992   0.9992   0.9990   1.0000
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2845   0.1929   0.1737   0.1630   0.1839
## Detection Prevalence   0.2850   0.1935   0.1743   0.1631   0.1840
## Balanced Accuracy      0.9996   0.9978   0.9976   0.9973   0.9999
```

## Method 2: Decision Tree

```r
modelDecisionTree <- rpart(classe ~ ., data=TrainingSet, method="class")
# prediction on Test dataset
predictDecisionTree <- predict(modelDecisionTree, newdata=TestingSet, type="class")
confusionMatrixDecisionTree <- confusionMatrix(predictDecisionTree, TestingSet$classe)
confusionMatrixDecisionTree
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1496  114    3   18    7
##          B   74  843   55   71   44
##          C    0   57  830   37    3
##          D   84   51  124  778   68
##          E   20   74   14   60  960
## 
## Overall Statistics
##                                           
##                Accuracy : 0.8338          
##                  95% CI : (0.8241, 0.8432)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.7901          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8937   0.7401   0.8090   0.8071   0.8872
## Specificity            0.9663   0.9486   0.9800   0.9336   0.9650
## Pos Pred Value         0.9133   0.7755   0.8954   0.7041   0.8511
## Neg Pred Value         0.9581   0.9383   0.9605   0.9611   0.9744
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2542   0.1432   0.1410   0.1322   0.1631
## Detection Prevalence   0.2783   0.1847   0.1575   0.1878   0.1917
## Balanced Accuracy      0.9300   0.8444   0.8945   0.8703   0.9261
```

## Method 3: Gradient Boosting Model

```r
controlGBM <- trainControl(method = "repeatedcv", number = 5, repeats = 1)
modelGBM  <- train(classe ~ ., data=TrainingSet, method = "gbm",trControl = controlGBM, verbose = FALSE)
modelGBM$finalModel
```

```
## A gradient boosted model with multinomial loss function.
## 150 iterations were performed.
## There were 53 predictors of which 53 had non-zero influence.
```

```r
# prediction on Test dataset
predictGBM <- predict(modelGBM, newdata=TestingSet)
confusionMatrixGBM <- confusionMatrix(predictGBM, TestingSet$classe)
confusionMatrixGBM
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1673   13    0    0    1
##          B    1 1117   13    3    1
##          C    0    9 1010   18    4
##          D    0    0    3  943    8
##          E    0    0    0    0 1068
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9874          
##                  95% CI : (0.9842, 0.9901)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9841          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9994   0.9807   0.9844   0.9782   0.9871
## Specificity            0.9967   0.9962   0.9936   0.9978   1.0000
## Pos Pred Value         0.9917   0.9841   0.9702   0.9885   1.0000
## Neg Pred Value         0.9998   0.9954   0.9967   0.9957   0.9971
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2843   0.1898   0.1716   0.1602   0.1815
## Detection Prevalence   0.2867   0.1929   0.1769   0.1621   0.1815
## Balanced Accuracy      0.9980   0.9884   0.9890   0.9880   0.9935
```

## Prediction with RandomForest
In this model, RandomForest is choosen because it has the highest accuracy rate compare with other two models, Decision Tree and Gradient Boosting Model

## Expected out-of-sample error
The expected out-of-sample error is very low, estimated at 0.5%. The expected out-of-sample error is calculated as 1 - accuracy for predictions made. Test data set is consisted with 20 cases. The accuracy is very high >99%, our expectation is that almost none of the test samples would be missclassified.


```r
predictRFtest <- predict(modelRandomForest, newdata=testing)
predictRFtest
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

