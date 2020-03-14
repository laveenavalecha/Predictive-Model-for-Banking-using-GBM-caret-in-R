# Author - Laveena Valecha - valechalaveena21@gmail.com

library(dplyr)
options(stringsAsFactors =FALSE,scipen = 99)

bank_train = read.csv("R/data/bank-full_train.csv", stringsAsFactors = FALSE)
bank_test = read.csv("R/data/bank-full_test.csv", stringsAsFactors = FALSE)

setdiff(names(bank_train), names(bank_test))
bank_test$y = NA

bank_train$data = "train"
bank_test$data = "test"

bank = rbind(bank_train, bank_test)

sapply(bank[,names(bank)], function(x)sum(is.na(x)))

## Removing the garbage columns which contains a lot of categories or missing data
bank=bank %>%
  select(-ID)

str(bank)

#bank$y=(bank$y=="yes")+0

CreateDummies=function(data,var,freq_cutoff=0){
  t=table(data[,var]) ## getting the table for the variable(any categorical variable)
  t=t[t>freq_cutoff] ## cutoff is the frequency of occurance of a variable default is 0 , but ideally it should be atleast 15-20% of actual data,
  ## so here whatever categories which are less than that cut off frequencies are dropped(no dummes are created for them)
  t=sort(t) ## sort the data
  categories=names(t)[-1] ## pick everything but exclude the first as it has lowest frequency: REmember its n-1
  
  for( cat in categories){
    name=paste(var,cat,sep="_") ## Inside the for loop create a name separated by name of variable and category separeted by "_" underscore
    name=gsub(" ","",name) ## replace any spaces if there is found in categoreis of variables
    name=gsub("-","_",name) ## replace any dash if found in categories to underscropes: e.g. 'Cat-1', 'Cat-2' will be 'Cat_1', 'Cat_2'
    name=gsub("\\?","Q",name) ## any question mark is converted to 'Q'
    name=gsub("<","LT_",name) ## Less than sign is converted to LT_
    name=gsub("\\+","",name) ## + sign is removed
    name=gsub("\\/","_",name) ## "/" is replaced with "_"
    name=gsub(">","GT_",name) ## ">" is replaced with 'GT_'
    name=gsub("=","EQ_",name) ## '=' is replaced with 'EQ_'
    name=gsub(",","",name) ##  ',' is replaced with ''
    data[,name]=as.numeric(data[,var]==cat) ## changing to numeric type
  }
  
  data[,var]=NULL
  return(data)
}

#bank  %>% group_by(bank$month) %>%  summarise('count'= n()) %>% View()
dim(bank)
## picking all the character columns and creating dummies
bank=CreateDummies(bank,"job",2000) ##job
bank=CreateDummies(bank,"marital") ##marital
bank=CreateDummies(bank,"education") ##education
bank=CreateDummies(bank,"default") ##default
bank=CreateDummies(bank,"housing") ##housing
bank=CreateDummies(bank,"loan") ##loan
bank=CreateDummies(bank,"contact") ##contact
bank=CreateDummies(bank,"poutcome") ##poutcome

bank1=bank
bank=bank1

## converting the age band to numeric
bank=bank %>%
  mutate(week1=ifelse(bank$day>=1 & bank$day<=7,1,0)) 

bank=bank %>%
  mutate(week2=ifelse(bank$day>=8 & bank$day<=14,1,0)) 

bank=bank %>%  
  mutate(week3=ifelse(bank$day>=15 & bank$day<=21,1,0))

bank=bank %>%
  mutate(week4=ifelse(bank$day>=22 & bank$day<=28,1,0)) 

bank=bank %>%  
  mutate(weekRem=ifelse(bank$day>=29,1,0)) 

names(bank)

##converting months into quarters
bank= bank %>% 
  mutate(Q1=ifelse(bank$month=="jan" | bank$month=="feb" | bank$month=="mar",1,0)) 
bank= bank %>% 
  mutate(Q2=ifelse(bank$month=="apr" | bank$month=="may" | bank$month=="jun",1,0))
bank= bank %>% 
  mutate(Q3=ifelse(bank$month=="jul" | bank$month=="aug" | bank$month=="sep",1,0))
bank= bank %>% 
  mutate(Q4=ifelse(bank$month=="oct" | bank$month=="nov" | bank$month=="dec",1,0))


names(bank)    
bank=bank %>% select(-day,-month)


bank_train=bank %>% filter(data=='train') %>% select(-data)
bank_test=bank %>% filter(data=='test') %>% select (-data,-y)

## ------------------------------------------------------------------------


## splitting the data 80%-20%
set.seed(21)
s=sample(1:nrow(bank_train),0.8*nrow(bank_train))
bank_train1=bank_train[s,]
bank_train2=bank_train[-s,]


library(car)
library(gbm)
library(cvTools)
library(caret)

fitControl <- trainControl(## 10-fold CV
  method = "cv",
  # repeats = 5,
  sampling = 'up',
  number = 5,
  summaryFunction = twoClassSummary,
  classProbs = TRUE)

set.seed(2)
gbmGrid <- expand.grid(interaction.depth = c(1, 5, 9), 
                       n.trees = (1:30)*50, 
                       shrinkage = 0.01,
                       n.minobsinnode = 20)
gbm_samp_grid <- sample(1:nrow(gbmGrid),.50*nrow(gbmGrid))
grid <- gbmGrid[gbm_samp_grid,]


gbmFit1 <- train(y ~ ., data = bank_train1, 
                 method = "gbm", 
                 trControl = fitControl,
                 metric = 'ROC',
                 tuneGrid = grid,
                 ## This last option is actually one
                 ## for gbm() that passes through
                 verbose = TRUE)

train.pred=predict(gbmFit1,newdata=bank_train1,type = 'prob')
View(train.pred)
val.pred=predict(gbmFit1,newdata=bank_train2)
test.pred=predict(gbmFit1,newdata=bank_test)
yesdata=round(train.pred$yes)

length(test.pred[test.pred=="no"])
write.csv(test.pred,"Laveena_Valecha_P5_part2.csv",row.names = F)





train1values=(train.pred=="yes")+0
train2values=(val.pred=="yes")+0
caTools::colAUC(train1values, bank_train1$y)
caTools::colAUC(train2values, bank_train2$y)
finalvalues=(test.pred=="yes")+0

## For K-S we use below code
## get the real value using y of bank_train1
real=bank_train1$y
length(real)
## get 999 values of probabilities score for which you want to test TP, FP, FN and TN
cutoffs=seq(0.001,0.999,0.001)
length(cutoffs)

## Create a data frame with initialised garbage values
cutoff_data=data.frame(cutoff=99,Sn=99,Sp=99,KS=99,F5=99,F.1=99,M=99)

## iterating the loop for all the 999 probabilities
for(cutoff in cutoffs){
  ## determine the prediction for each cut off here
  predicted=as.numeric(yesdata>cutoff)
  
  ## fill the value of TP, FP, FN and TN
  TP=sum(real==1 & predicted==1)
  TN=sum(real==0 & predicted==0)
  FP=sum(real==0 & predicted==1)
  FN=sum(real==1 & predicted==0)
  
  P=TP+FN
  N=TN+FP
  
  Sn=TP/P
  Sp=TN/N
  precision=TP/(TP+FP)
  recall=Sn
  ## KS is the cutoff
  KS=(TP/P)-(FP/N)
  
  F5=(26*precision*recall)/((25*precision)+recall)
  ## F.1 score is maximum at 1 and min at 0
  ## A value of F.1 closer to 1 is good
  ## In case of low event rate model, F.1 closer to 1 is great
  ## F.1 score captures both precision and recall hence it is very useful in case of low event rate model
  F.1=(1.01*precision*recall)/((.01*precision)+recall)
  
  M=(4*FP+FN)/(5*(P+N))
  
  ## Binding the data
  cutoff_data=rbind(cutoff_data,c(cutoff,Sn,Sp,KS,F5,F.1,M))
}


## removing the garbage column
cutoff_data=cutoff_data[-1,]

## getting the row where maximum value of KS is there
cutoff_data[cutoff_data$KS == max(cutoff_data$KS),]
View(cutoff_data)







