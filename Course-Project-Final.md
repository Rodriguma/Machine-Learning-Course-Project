---
title: "Qualitative activity recognition"
author: "Marta.R"
date: "23 de enero de 2016"
output: html_document
---

## SUMMARY
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit makes now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. 

**In this project, the course-project for the Practical Machine Learning" course at coursera, our goal will be to use data from UMIT (Project description says only accelerometers, but I guess we should also include gyroscope and magnetometer) on the belt, forearm, arm, and dumbbell of 6 participants to predict how well the exercise is done.**

The original study and addittional information are available from the website: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

## THE QUESTION
The question we want to address; **is it possible to predict how well barbell lifts are done from the information of four sensors in the belt, forearm, arm, and dumbbell of the person doing the exercise?**

We will see through this report we are going to be able to train a model to get the accurate predictions needed for the quiz, as we have records of all the people we intend to predict about. 
However, as indicated in the original report, our classifier/model trained for some people/subjects isn't very useful for predicting a new subject, we will check this behaviour in the [Appendix](#id1) of this report. 


## THE DATA
Six participants in the experiment were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in 5 different ways: according to the specification (Class A) and with four different common mistakes (Class B, C, D & E). The data we will work on was recorded from four sensors on the belt, forearm, arm, and dumbbell of 6 participants. These sensors provided three-axes acceleration, gyroscope and magnetometer data.

The training data is available in http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv , and the data to predict in http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv.

To start the study we will download the data and load it in to R. 


```r
## getting the data and reading it in R
OrigData_URL<-"http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
predictions_URL<-"http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
if(!file.exists("./Data")) {dir.create("./Data")}
if(!file.exists("./Data/pml-training.csv")) {
download.file(OrigData_URL,destfile = "./Data/pml-training.csv")}
if(!file.exists("./Data/pml-testing.csv")) {
download.file(predictions_URL,destfile = "./Data/pml-testing.csv")}
OrigData<-read.csv("./Data/pml-training.csv",na.strings=c("NA","#DIV/0!",""))
predictions<-read.csv("./Data/pml-testing.csv",na.strings=c("NA","#DIV/0!",""))
```
The data we get from the file is not the original measurements from the sensors, in fact it contains some of the transformations of the original experiments that could lead us to the final results doing nothing; all the records with the same value in the num_window column will have the same classification. 
So to go ahead with the study we will remove the seven first columns and the columns that have NA's in all the rows with new_window=="no" (this columns contain the mean, variance, standard deviation, max, min, amplitude, kurtosis and skewness, generating in total 96 derived feature sets, for the Euler angles of each of the four sensors by window).


```r
##cleaning the data
which(sapply(OrigData[OrigData$new_window=="no",], function(x)all(is.na(x))))->quitarcolumn
OrigData[,-quitarcolumn]->OrigData1
OrigData1[,8:length(OrigData1)]->OrigData1
predictions[,-quitarcolumn]->predictions1
predictions1[,8:length(predictions1)]->predictions1
predictions1->predictions2
```

Now we will split our dataset into the training and test sets to go ahead with features and model selection. 

```r
## Creating training and test subsets
set.seed(1974)
inTrain = createDataPartition(OrigData1$classe, p = 0.6, list = FALSE)
training = OrigData1[ inTrain,]
testing = OrigData1[-inTrain,]
```
From now on we will only work with training set until the validation phase when we will calculate the out of sample error in the test set.

## FEATURES
I am not an expert in Kinematics, but looking at the information in the original report it seems that from the initial measurements they calculated Euler Angles for the four sensors. The Euler angles are rotational coordinates to describe the orientation of a rigid body (to get Euler angles we have to use specific nonlinear transformations of the measurements). So I think it is a good start to use the Euler angles as our features to train the algorithms.  

These Euler angles are recorded in columns named "euler_angle"_"sensor_position", where possible "euler_angle" are roll, pitch or yaw and possible "Sensor_position" are belt, arm, dumbbell or forearm. So I will subset the data to get just those 12 columns plus the classe one.


```r
grep("yaw|pitch|roll",names(training))->selectcolumn
c(selectcolumn,53)->selectcolumn
training[,selectcolumn]->training2
testing[,selectcolumn]->testing2
predictions1[,selectcolumn]->predictions1
```
## ALGORITHM
Now we have chosen the features we think more suitable, we will train different algorithms to compare them and select the best for our predictions. 
I will start from the simplest one, a Classification tree algorithm, then I will train a Bagging Classification tree, later a Radom forest and finally a Gradient Boosting Algorithm. 
We will implement cross validation for all of them using 8 K-folds to try to avoid overfitting. 

First we will set the train control options so that all the algorithms trained later use cross validation with 5 K-folds.


```r
trainControl(method="cv", number=8)->TCTR
```

And now we will train each algorithm according to the options below.

### Classification Tree


```
## train.formula(form = classe ~ ., data = training2, method = "rpart", 
##     trControl = TCTR)
```

### Classification Tree with Bagging


```
## train.formula(form = classe ~ ., data = training2, method = "treebag", 
##     trControl = TCTR)
```

### Random Forest (500 trees)


```
## train.formula(form = classe ~ ., data = training2, method = "rf", 
##     trControl = TCTR)
```

### Gradient Boosting 


```
## train.formula(form = classe ~ ., data = training2, method = "gbm", 
##     verbose = FALSE, trControl = TCTR)
```

## VALIDATION
Now we will calculate accuracy and error rates (out and in sample) for the different models to compare the results;  

```r
methods<-list(ModeltreeCV,ModeltreeBAGCV,ModelRFCV,ModelGBMCV)
predict(methods, newdata=testing2)-> tests
predict(methods, newdata=training2)-> tests2
createMatrix <- function(x) {
  confusionMatrix(x,testing2$classe)
}
createMatrix2 <- function(x) {
  confusionMatrix(x,training2$classe)
}
lapply(tests,createMatrix)->confusions
lapply(tests2,createMatrix2)->confusions2
```

Now here we have the different results we got from all the algorithms.


|Algorithm                    |Accuracy in training Sample |In Sample error |Accuracy in testing Sample |Out of Sample error |Time to train the Model |
|:----------------------------|:---------------------------|:---------------|:--------------------------|:-------------------|:-----------------------|
|CART                         |47.02%                      |52.98%          |47.21%                     |52.79%              |15S                     |
|Bagged CART                  |100.00%                     |0.00%           |98.09%                     |1.91%               |4M 36S                  |
|Random Forest                |100.00%                     |0.00%           |98.66%                     |1.34%               |8M 57S                  |
|Stochastic Gradient Boosting |93.94%                      |6.06%           |92.75%                     |7.25%               |5M 41S                  |

## CONCLUSIONS
From the information in the previous table the most accurate model is Random Forest, however "Classification trees with Bagging Algorithm" is performing quite well, its accuracy is near to RF's and the time needed to train the algorithm is half the time needed to train RF. It is also true that the file generated after saving the Bagging method is ten times the size of the RF's one.
As my PC is an old one with a slow two-cores CPU and 2 GB of memory, and time is quite important, I will select the second algorithm, just in case I got more data to train :). 

## Final prediction of the 20 records in "pml-testing"

Now we will predict the class of the records in pml-testing file, using the model we selected in the previous Step,**"Classification trees with Bagging Algorithm"**


|R.1 |R.2 |R.3 |R.4 |R.5 |R.6 |R.7 |R.8 |R.9 |R.10 |R.11 |R.12 |R.13 |R.14 |R.15 |R.16 |R.17 |R.18 |R.19 |R.20 |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|
|B   |A   |B   |A   |A   |E   |D   |B   |A   |A    |B    |C    |B    |A    |E    |E    |A    |B    |B    |B    |

Double checking the results we get the predictions are equal using Random Forest Algorithm.

|R.1 |R.2 |R.3 |R.4 |R.5 |R.6 |R.7 |R.8 |R.9 |R.10 |R.11 |R.12 |R.13 |R.14 |R.15 |R.16 |R.17 |R.18 |R.19 |R.20 |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|
|B   |A   |B   |A   |A   |E   |D   |B   |A   |A    |B    |C    |B    |A    |E    |E    |A    |B    |B    |B    |

Predictions have some differences in the other models.


```
## [1] "Classification Tree Algorithm"
```



|R.1 |R.2 |R.3 |R.4 |R.5 |R.6 |R.7 |R.8 |R.9 |R.10 |R.11 |R.12 |R.13 |R.14 |R.15 |R.16 |R.17 |R.18 |R.19 |R.20 |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|
|C   |A   |C   |A   |A   |C   |C   |A   |A   |A    |C    |C    |C    |A    |C    |A    |A    |A    |A    |C    |

```
## [1] "Gradient Boosting Algorithm"
```



|R.1 |R.2 |R.3 |R.4 |R.5 |R.6 |R.7 |R.8 |R.9 |R.10 |R.11 |R.12 |R.13 |R.14 |R.15 |R.16 |R.17 |R.18 |R.19 |R.20 |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|
|B   |A   |B   |A   |A   |E   |D   |E   |A   |A    |B    |C    |B    |A    |E    |E    |A    |B    |B    |B    |

## APPENDIX

<a id="id1"></a>

### Leaving one subject out of the training set.

As we said before the classifier we trained is very accurate for predicting the exercise of people whose data is used to train the algorithm. However it is not very useful for predicting new people's exercise.

The original study suggest the use of this approach requires a lot of data from more subjects, in order to reach a result that can be generalized for a new user without the need of training the classifier. So this is another good reason to select Bagging against Random Forest, as Bagging needs less computing resources.

Below you can find the results obtained training the best two algorithms leaving one subject OUT of the training set. As you can see overall recognition performance (out of sample accuracy) is quite poor, no matter who you leave out or the algorithm you use. 




|subject left Out |Algorithm     |Accuracy in training Sample |In Sample error |Accuracy in testing Sample |Out of Sample error |Time to train the Model |
|:----------------|:-------------|:---------------------------|:---------------|:--------------------------|:-------------------|:-----------------------|
|adelmo           |Random Forest |99.99%                      |0.01%           |17.39%                     |82.61%              |11M 7S                  |
|adelmo           |Bagged CART   |99.99%                      |0.01%           |17.37%                     |82.63%              |5M 54S                  |
|carlitos         |Random Forest |100.00%                     |0.00%           |32.97%                     |67.03%              |11M 40S                 |
|carlitos         |Bagged CART   |99.94%                      |0.06%           |32.84%                     |67.16%              |6M 8S                   |
|charles          |Random Forest |100.00%                     |0.00%           |43.75%                     |56.25%              |11M 57S                 |
|charles          |Bagged CART   |99.98%                      |0.02%           |34.90%                     |65.10%              |6M 1S                   |
|eurico           |Random Forest |99.99%                      |0.01%           |26.68%                     |73.32%              |11M 19S                 |
|eurico           |Bagged CART   |99.98%                      |0.02%           |20.78%                     |79.22%              |6M 5S                   |
|jeremy           |Random Forest |100.00%                     |0.00%           |37.04%                     |62.96%              |13M 3S                  |
|jeremy           |Bagged CART   |99.99%                      |0.01%           |36.07%                     |63.93%              |6M 11S                  |
|pedro            |Random Forest |99.98%                      |0.02%           |37.97%                     |62.03%              |12M 27S                 |
|pedro            |Bagged CART   |99.96%                      |0.04%           |17.97%                     |82.03%              |6M 23S                  |

### NOTES
Training each model in this report takes quite long, so all the algorithms trained in this report are saved in files available in the github repo. The markdown file checks the existence of these files before starting to train the algorithm.
All the code is available in the associated Rmd file in the Github Repo. 
