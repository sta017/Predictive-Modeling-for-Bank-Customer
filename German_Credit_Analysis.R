library(ggplot2)
library(gridExtra)
library(dplyr)
library(repr)

credit = read.csv('German_Credit.csv')
head(credit, n=3)

# Add column names to make them more readable to people.

names(credit) = c('Customer_ID', 'checking_account_status', 'loan_duration_mo', 'credit_history', 
                  'purpose', 'loan_amount', 'savings_account_balance', 
                  'time_employed_yrs', 'payment_pcnt_income','gender_status', 
                  'other_signators', 'time_in_residence', 'property', 'age_yrs',
                  'other_credit_outstanding', 'home_ownership', 'number_loans', 
                  'job_category', 'dependents', 'telephone', 'foreign_worker', 
                  'bad_credit')
str(credit)

# Next step is the recode the categorical features as well with readable text
#instead of codes.

checking_account_status = c('< 0 DM', '0 - 200 DM', '> 200 DM or salary assignment', 'none')
names(checking_account_status) = c('A11', 'A12', 'A13', 'A14')
credit_history = c('no credit - paid', 'all loans at bank paid', 'current loans paid', 
                   'past payment delays',  'critical account - other non-bank loans')
names(credit_history) = c('A30', 'A31', 'A32', 'A33', 'A34')
purpose = c( 'car (new)', 'car (used)', 'furniture/equipment', 'radio/television', 
             'domestic appliances', 'repairs', 'education', 'vacation', 'retraining',
             'business', 'other')
names(purpose) = c('A40', 'A41', 'A42', 'A43', 'A44', 'A45', 'A46', 'A47', 'A48', 'A49', 'A410')
savings_account_balance = c('< 100 DM', '100 - 500 DM', '500 - 1000 DM', '>= 1000 DM', 'unknown/none')
names(savings_account_balance) = c('A61', 'A62', 'A63', 'A64', 'A65')
time_employed_yrs = c('unemployed', '< 1 year', '1 - 4 years', '4 - 7 years', '>= 7 years')
names(time_employed_yrs) = c('A71', 'A72', 'A73', 'A74', 'A75')
gender_status = c('male-divorced/separated', 'female-divorced/separated/married',
                  'male-single', 'male-married/widowed', 'female-single')
names(gender_status) = c('A91', 'A92', 'A93', 'A94', 'A95')
other_signators = c('none', 'co-applicant', 'guarantor')
names(other_signators) = c('A101', 'A102', 'A103')
property =  c('real estate', 'building society savings/life insurance', 'car or other', 'unknown-none')
names(property) = c('A121', 'A122', 'A123', 'A124')
other_credit_outstanding = c('bank', 'stores', 'none')
names(other_credit_outstanding) = c('A141', 'A142', 'A143')
home_ownership = c('rent', 'own', 'for free')
names(home_ownership) = c('A151', 'A152', 'A153')
job_category = c('unemployed-unskilled-non-resident', 'unskilled-resident', 'skilled', 'highly skilled')
names(job_category) =c('A171', 'A172', 'A173', 'A174')
telephone = c('none', 'yes')
names(telephone) = c('A191', 'A192')
foreign_worker = c('yes', 'no')
names(foreign_worker) = c('A201', 'A202')
bad_credit = c(1, 0)
names(bad_credit) = c(2, 1)
            
codes = c('checking_account_status' = checking_account_status,
         'credit_history' = credit_history,
         'purpose' = purpose,
         'savings_account_balance' = savings_account_balance,
         'time_employed_yrs' = time_employed_yrs,
         'gender_status' = gender_status,
         'other_signators' = other_signators,
         'property' = property,
         'other_credit_outstanding' = other_credit_outstanding,
         'home_ownership' = home_ownership,
         'job_category' = job_category,
         'telephone' = telephone,
         'foreign_worker' = foreign_worker,
         'bad_credit' = bad_credit)         

cat_cols = c('checking_account_status', 'credit_history', 'purpose', 'savings_account_balance', 
                  'time_employed_yrs','gender_status', 'other_signators', 'property',
                  'other_credit_outstanding', 'home_ownership', 'job_category', 'telephone', 'foreign_worker', 
                  'bad_credit')
for(col in cat_cols){
 credit[,col] = sapply(credit[,col], function(code){codes[[paste(col,
			'.', code,sep='')]]})
}

head(credit)             # the categorical values are now coded in readable form.



#2. Removing Duplicate entries
dim(credit)
dim(distinct(credit))  # if any duplicates, it will be deleted.

# update the 'credit' array.
credit <- distinct(credit)

# Let's save the dataframe to a csv file 
# 
#write.csv(credit, file = 'German_Credit_Preped.csv', row.names = FALSE)

##3. FEATURE ENGINEERING:

#listing the desired columns for feature engineering
columns =c('loan_duration_mo', 'loan_amount', 'age_yrs')
log_col= c('log_loan_duration_mo', 'log_loan_amount', 'log_age_yrs')

credit[,log_col]= lapply(credit[,columns], log)   # Applying log transformation

#Visualizing the change.
plot_violin= function(df, cols, col_x='bad_credit'){
 options(repr.plot.width=4, repr.plot.height=4)
 for(col in cols){
	p= ggplot(df, aes_string(col_x, col))+
		geom_violin()+
		ggtitle(paste('Violin Plot of',col_x,'vs.',col,
			'with log-transformation'))
	print(p)
	}
}

plot_violin(credit, log_col)


############# Applications of  Classification #################3

##REDO
#A classifier is a machine learning model that separates the label into
categories or classes.

library(caret)
library(ROCR)
library(pROC)


## Computing and plotting an example Logistic function.

xseq = seq(-7, 7, length.out=500)
plot.logistic= function(v) {
 options(repr.plot.width = 4, repr.plot.height= 4)
 logistic= exp(xseq- v)/(1 + exp(xseq -v))
 df = data.frame(x= xseq, y = logistic)
 ggplot(df, aes(x,y)) + 
	geom_line(size= 1, color= 'red')+                 # sigmoid func. line
 	geom_vline(xintercept = v, size = 1, color= 'blue')+    # (y-axis or, x intercept))
	geom_hline(yintercept = 0.5, size=1, color='black')+    # (x-axis or , y intercept)
	ylab('log likelihood') + xlab('Value of x') +
	ggtitle('Logistic function for \n two-class classification') +
	theme_grey(base_size= 18)
}
plot.logistic(0)

#####
#Back to data

#4. Investigating data imbalance

table(credit[, 'bad_credit'])
# Since 'caret' package uses positive cases for analysis, we use
# bad credit as the positive and good credit as the negative

credit$bad_credit <- ifelse(credit[,'bad_credit']==1,'bad','good')
credit$bad_credit <- factor(credit[, 'bad_credit'], level= c('bad','good'))
credit$bad_credit[1:5]

# 5. SPLitting the data
set.seed(1234)
#using caret package for 'createDataPartition' function
partition = createDataPartition(credit[,'bad_credit'], times = 1, p=0.7, list=FALSE)
training = credit[partition,]
test = credit[-partition,]
dim(training)

#6. Scaling the data:(numeric features only)

#NOTE : missing steps (plots to show 4 of 7 numerical features are more important)
num_cols = c('loan_duration_mo', 'loan_amount', 'payment_pcnt_income', 'age_yrs')
preProcValues <- preProcess(training[, num_cols], method = c('center','scale'))
training[,num_cols] <-  predict(preProcValues, training[,num_cols])
test[,num_cols] <- predict(preProcValues, test[,num_cols])
head(training[,num_cols])

#7. Constructing the Logistic Regression Model with glm() function as follows:
#compute a model object as follows:

# i.The formula for the label vs. the features is defined.
#ii.Since this is logistic regression, the Binomial distribution
# 	is specified for the response.

set.seed(1234)
logistic_mod = glm(bad_credit ~ loan_duration_mo + loan_amount +
						payment_pcnt_income + age_yrs+
						checking_account_status + credit_history+
						purpose + gender_status + time_in_residence+
						property,
		  	family= binomial,
			data= training)

logistic_mod$coefficients

#First of all, notice that model coefficients are similar to what you might expect
# for an linear regression model. As previously explained the logistic
# regression is indeed a linear model.

#Since logistic reg. model outputs log likelihood, the classes with 
#highest probability are taken as the score(prediction).

test$probs = predict(logistic_mod, newdata= test, type ='response')
test[1:10, c('bad_credit','probs')]

# column 'bad_credit' is the label as we know. 
# column 2 i.e 'probs' is the log likelihood of a positive score, and
#		notice several of these log likelihoods are close to "0.5"


#8. Score and EVALUATE the CLASSIFICATION model

# the computed log likelihood to be transformed into actual classes. 

score_model = function(df, threshold){
	df$score = ifelse(df$probs <threshold, 'bad','good')
 	df
}
test = score_model(test, 0.5)
training = score_model(training, 0.5)          # just checking to see errors in training set.
test[1:10, c('bad_credit','probs','score')]

#Some of the predictions agree but several do not.  
#We always use MULTIPLE METRICS to evaluate performance.
# i. CONFUSION MATRIX
# Usual convention is to call '1' positive and '0' as negative


logistic.eval <- function(df){
 #1. First step is to create confusion matrix. 
 df$conf = ifelse(df$bad_credit == 'bad' & df$score== 'bad', 'TP', 
		ifelse(df$bad_credit == 'bad' & df$score == 'good', 'FN',
			ifelse(df$bad_credit =='good' & df$score =='good','TN', 'FP' ) ))

 #2. Elements of confusion matrix
 TP = length(df[df$conf== 'TP','conf'])
 FP = length(df[df$conf== 'FP','conf'])
 TN = length(df[df$conf== 'TN','conf'])
 FN= length(df[df$conf == 'FN','conf'])

 #3. Confusion matrix as data frame
 out= data.frame(Negative = c(TN,FN), Positive=c(FP,TP))
 row.names(out) = c('Actual Negative','Actual Positive')
 print(out)

 #4. Compute and print metrics
 P= TP/(TP + FP)
 R= TP/(TP + FN)
 F1= 2*(P*R)/(P+R)
 A= (TP +TN)/(TP+TN+FP+FN)
 S= TN/(TN+FP)
 cat('\n')
 cat(paste('Accuracy =', as.character(round(A, 3)),'\n'))
 cat(paste('Precision=', as.character(round(P, 3)),'\n'))
 cat(paste('Recall   =', as.character(round(R, 3)),'\n'))
 cat(paste('Specificity=',as.character(round(S,3)),'\n'))
 cat(paste('F1	   =', as.character(round(F1, 3)), '\n'))

 roc_obj <- roc(df$bad_credit, df$probs)
 cat(paste('AUC      =', as.character(round(auc(roc_obj), 3)), '\n'))
}

logistic.eval(test)

#NOTE:
##  If the 'score' has been converted into same categorical(factor) 
##  variable, the function "confusionMatrix()" can be used. 

## same result.
confusionMatrix(factor(test$score), test$bad_credit)

#however, the columns and row names are switched compared to earlier.

## prediction, performance()   - ROCR package to compute and display ROC curve.

ROC_AUC= function(df){
 options(repr.plot.width = 4, repr.plot.height = 4)
 pred_obj = prediction(df$probs, df$bad_credit)
 perf_obj = performance(pred_obj, measure = 'tpr', x.measure= 'fpr')
 AUC = performance(pred_obj, 'auc')@y.values[[1]] 
      #accesses the AUC from the slot of the S4 object
 plot(perf_obj)
 abline(a=0, b=1, col='red')
 text(0.8,0.2, paste('AUC  =', as.character(round(AUC,3))))
}

ROC_AUC(test)

### Given that this data is very imbalanced (70:30),  the AUC value cannot 
##be readily trusted. use a NAIVE classifier
#as below that sets all cases to positive. 

test$probs = rep(0, length.out= nrow(test))
test$score = rep(0, length.out= nrow(test))
logistic.eval(test)
ROC_AUC(test)



###
# 5. Computing a WEIGHTED Model:




# NOTE: i. As the data is imbalance, it biases the training of the model. 
#    and  the Accuracy can't be trusted. 
#  ii. Weight the result to counter the imbalance


#Create a weight vector for the training case
weights = ifelse(training$bad_credit == 'bad', 0.66, 0.33)

# create logistic reg. model with "glm()" with weights
logistic_mod_w = glm(bad_credit ~ loan_duration_mo + loan_amount +
				payment_pcnt_income +age_yrs +
				checking_account_status+ credit_history+
				purpose + gender_status + time_in_residence+
				property,
			family= quasibinomial, 
			data= training,
			weights= weights)

test$probs= predict(logistic_mod_w, newdata= test, type ='response')
test[1:20, c('bad_credit','probs')]

#To find if there is any significant difference with the unweighted model, compute 
#the scores, the metrics and display as below

test= score_model(test, 0.5)
logistic.eval(test)
ROC_AUC(test)

# F1 and Recall have improved at expense of Precision. But the model 
#performance has improved to desired direction, although , the AUC value
# is about the same. (proving again, AUC, by itself can't be trusted.)



#6. FIND A BETTER THRESHOLD



##Recall that the score is determined by setting the threshold along
# the sigmoidal or logistic function. It is possible to favor either 
#positive or negative cases by changing the threshold along this curve.


test_threshold= function(df, threshold){
 df$score= predict(logistic_mod_w, newdata= df, type= 'response')
 df= score_model(df, t)
 cat('\n')
 cat(paste('For threshold =', as.character(threshold), '\n'))
 logistic.eval(df)

}

thresholds= c(0.5,0.55, 0.6,0.65)

for(t in thresholds){
	test_threshold(test, t)       # Iterate over threshold values
}


#### Here we observe above threshold 0.6, misclassified good customers
## (FN) are more than 3 times misclassified bad customers(FP)
#     The threshold to choose is purely BUSINESS DECISION.

# I prefer 0.60 as Precision, F1 and Recall are the most favorable here. 










































