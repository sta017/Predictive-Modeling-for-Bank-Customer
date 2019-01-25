import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import numpy.random as nr
import math
from sklearn import preprocessing           ## for Applying Classification/ Module 4
import sklearn.model_selection as ms
from sklearn import linear_model
import sklearn.metrics as sklm

### 1. Reading the data
credit= pd.read_csv('German_Credit.csv', header= None)
credit.columns = ['customer_id', 'checking_account_status','loan_duration_mo', 'credit_history',
                  'purpose','loan_amount', 'savings_account_balance', 'time_employed_yrs',
                  'payment_pcnt_income', 'gender_status', 'other_signators', 'time_in_residence',
                  'property', 'age_yrs','other_credit_outstanding', 'home_ownership','number_loans',
                  'job_category','dependents', 'telephone', 'foreign_worker', 'bad_credit']
##print(credit.shape)
##print(credit.head())
##
### 2.Dropping unneeded column
#credit.drop(['customer_id'], axis = 1, inplace= True)    
##print(credit.shape)
##print(credit.head())
##
#### 3. Using a list of dictionaries to recode the categorical features
#### the final dictionary recodes good and bad credit customer in binary as 0 and 1.
##
code_list = [['checking_account_status',
              {'A11' : '< 0 DM',
               'A12' : '0 - 200 DM',
               'A13' : '> 200 DM or salary assignment',
               'A14' : 'none'}],
             ['credit_history',
            {'A30' : 'no credit - paid', 
             'A31' : 'all loans at bank paid', 
             'A32' : 'current loans paid', 
             'A33' : 'past payment delays', 
             'A34' : 'critical account - other non-bank loans'}],
            ['purpose',
            {'A40' : 'car (new)', 
             'A41' : 'car (used)',
             'A42' : 'furniture/equipment',
             'A43' : 'radio/television', 
             'A44' : 'domestic appliances', 
             'A45' : 'repairs', 
             'A46' : 'education', 
             'A47' : 'vacation',
             'A48' : 'retraining',
             'A49' : 'business', 
             'A410' : 'other' }],
            ['savings_account_balance',
            {'A61' : '< 100 DM', 
             'A62' : '100 - 500 DM', 
             'A63' : '500 - 1000 DM', 
             'A64' : '>= 1000 DM',
             'A65' : 'unknown/none' }],
            ['time_employed_yrs',
            {'A71' : 'unemployed',
             'A72' : '< 1 year', 
             'A73' : '1 - 4 years', 
             'A74' : '4 - 7 years', 
             'A75' : '>= 7 years'}],
            ['gender_status',
            {'A91' : 'male-divorced/separated', 
             'A92' : 'female-divorced/separated/married',
             'A93' : 'male-single', 
             'A94' : 'male-married/widowed', 
             'A95' : 'female-single'}],
            ['other_signators',
            {'A101' : 'none', 
             'A102' : 'co-applicant', 
             'A103' : 'guarantor'}],
            ['property',
            {'A121' : 'real estate',
             'A122' : 'building society savings/life insurance', 
             'A123' : 'car or other',
             'A124' : 'unknown-none' }],
            ['other_credit_outstanding',
            {'A141' : 'bank', 
             'A142' : 'stores', 
             'A143' : 'none'}],
             ['home_ownership',
            {'A151' : 'rent', 
             'A152' : 'own', 
             'A153' : 'for free'}],
            ['job_category',
            {'A171' : 'unemployed-unskilled-non-resident', 
             'A172' : 'unskilled-resident', 
             'A173' : 'skilled',
             'A174' : 'highly skilled'}],
            ['telephone', 
            {'A191' : 'none', 
             'A192' : 'yes'}],
            ['foreign_worker',
            {'A201' : 'yes', 
             'A202' : 'no'}],
            ['bad_credit',
            {2 : 1,
             1 : 0}]]
for col_dic in code_list:
    col = col_dic[0]           # first element in the  'i'th dictionary 
    dic = col_dic[1]           #  second element
    credit[col] = [dic[x] for x in credit[col]]

    
##print(credit.head())

###Displaying the Frequency table for the label class to display CLASS IMBALANCE
credit_counts= credit['bad_credit'].value_counts()
#print(credit_counts)


##  4. Visualize class separation by Numeric features

def plot_box(credit, cols, col_x = 'bad_credit'):
    for col in cols:
        sns.set_style("whitegrid")
        sns.boxplot(col_x, col, data= credit)
        plt.xlabel(col_x)
        plt.ylabel(col)
        plt.show()
num_cols = ['loan_duration_mo', 'loan_amount', 'payment_pcnt_income',
            'age_yrs', 'number_loans', 'dependents']
#plot_box(credit, num_cols)


# 4. 2 alternatively, using violin plot.
def plot_violin(credit, cols, col_x= 'bad_credit'):
    for col in cols:
        sns.set_style('whitegrid')
        sns.violinplot(col_x, col, data= credit)
        plt.xlabel(col_x)
        plt.ylabel(col)
        plt.show()

#plot_violin(credit, num_cols)

# 5. Visualize the class separation by categorical features:

import numpy as np
cat_cols = ['checking_account_status', 'credit_history','purpose', 'savings_account_balance',
            'time_employed_yrs', 'gender_status', 'other_signators', 'property',
            'other_credit_outstanding', 'home_ownership', 'job_category', 'telephone',
            'foreign_worker']
credit['dummy']= np.ones(shape= credit.shape[0])  ## creating a unit matrix with length = nrows(credit)

for col in cat_cols:
#    print(col) 
    counts= credit[['dummy', 'bad_credit', col]].groupby(['bad_credit', col], as_index= False).count()
    temp = counts[counts['bad_credit']== 0][[col, 'dummy']]    
##    _ = plt.figure(figsize=(10,4))
##    plt.subplot(1,2,1)
    temp= counts[counts['bad_credit'] == 0][[col, 'dummy']]
##    plt.bar(temp[col], temp.dummy)
##    plt.xticks(rotation= 90)
##    plt.title('Counts for ' + col + '\n Bad Credit')
##    plt.ylabel('Count')
##    plt.subplot(1,2,2)
    temp= counts[counts['bad_credit']==1][[col, 'dummy']]
##    plt.bar(temp[col], temp.dummy)
##    plt.xticks(rotation= 90)
##    plt.title('Counts for ' + col)       #to use when superimposing good and bad credit labels.
##    plt.title('Counts for ' + col + '\n Good credit')
##    plt.ylabel('Count')
##    plt.show()
    
## Note: without plt.subplot,  the barplots for good and bad credit are superposed on each other.

#####################################################################################



### Module 3
####################################################################################
#6.   DATA PREPARATION:



# 6.i  Determining and  Removing Duplicate rows:

print(credit.shape)
print(credit.customer_id.unique().shape)


#### keeping the first instance and removing the later ones. 
credit.drop_duplicates(subset= 'customer_id', keep= 'first', inplace= True)
print(credit.shape)
print(credit.customer_id.unique().shape)

#### Saving the data to a csv file.

##credit.to_csv('German_Credit_Preped.csv', index=False, header=True)


# 6.ii:  Feature engineering:

credit[['log_loan_duration_mo', 'log_loan_amount', 'log_age_yrs']]= credit[['loan_duration_mo',
                                                                            'loan_amount',
                                                                            'age_yrs']].applymap(math.log)
num_cols = ['log_loan_duration_mo', 'log_loan_amount', 'log_age_yrs','loan_duration_mo','loan_amount','age_yrs']

for col in num_cols:
    print(col)
##    _=plt.figure(figsize = (5,4))
##    sns.violinplot(x='bad_credit', y=col, hue='bad_credit',
##                   data=credit)
##    plt.title('Plot for '+ col)
##    plt.ylabel('value')
##    plt.xlabel(col)
##    plt.show()


# though the log tranformation makes distribution more symmetric, it does not show any
#improvement in the label cases. So, these features will not be used further.
#

credit.drop(['log_loan_duration_mo', 'log_loan_amount', 'log_age_yrs'], axis = 1, inplace= True)


#ATTEMPTING TO MAKE SUBPLOT FOR ALL SIX FIG

##    for i in (1,2,3):
##        for j in (1,2):
##            plt.subplot(j,3,i)
##            plt.title('Plot for ' + col)
##            plt.ylabel('value')
##            plt.xlabel(col)
##            plt.show()
##    

##################################################################################


####   Module 4:  Applying Classification:

##################################################################################

###Examining the class imbalance

credit_counts= credit[['credit_history', 'bad_credit']].groupby('bad_credit').count()
##print(credit_counts)

### Notice only 30% of the cases are of bad_credit.

###  7. Prepare Data for   scikit-learn Model.

labels = np.array(credit['bad_credit'])

## now to create a numpy feature or model matrix, the categorical variables need to be
##  recorded as binary dummy variables.

## A. Encode the categorical string variables as integers.
## B. Transform the integer coded variables to dummy variables.
## C. Append each dummy coded categorical variable to the model matrix.

def encode_string(cat_features):
    # First encode the string s to numeric categories.
    enc = preprocessing.LabelEncoder()
    enc.fit(cat_features)
    enc_cat_features = enc.transform(cat_features)

    ##Now, apply one hot coding
    ohe = preprocessing.OneHotEncoder()
    encoded = ohe.fit(enc_cat_features.reshape(-1,1))
    return encoded.transform(enc_cat_features.reshape(-1,1)).toarray()

categorical_columns= ['credit_history','purpose','gender_status',
                      'time_in_residence', 'property']

Features = encode_string(credit['checking_account_status'])
for col in categorical_columns:
    temp = encode_string(credit[col])
    Features = np.concatenate([Features, temp], axis= 1)

#print(Features.shape)
#print(Features[:2, :])

## 7.i  Concatenate teh numeric features to the numpy array:
Features = np.concatenate([Features, np.array(credit[['loan_duration_mo', 'loan_amount',
                                                      'payment_pcnt_income', 'age_yrs']])], axis = 1)
#print(Features.shape)
#print(Features[:2, :])

## With the dummy varibales the original 6 categorical features are now 31 dummy variables,
## with 4 new numeric features, the total is 35. 

 ##  7.ii   Splitting the data

nr.seed(9988)
indx = range(Features.shape[0])
indx = ms.train_test_split(indx, test_size= 300)
X_train = Features[indx[0],:]
y_train = np.ravel(labels[indx[0]])
X_test= Features[indx[1], :]
y_test= np.ravel(labels[indx[1]])

## 7.iii     Z-score scaling the numeric features of the data:
scaler = preprocessing.StandardScaler().fit(X_train[:, 34:])
X_train[:, 34:]= scaler.transform(X_train[:, 34:])
X_test[:, 34:] = scaler.transform(X_test[:, 34:])
print(X_train[:2, ])


##8. Constructing the LOGISTIC REGRESSION:
logistic_mod = linear_model.LogisticRegression()
logistic_mod.fit(X_train, y_train)

print(logistic_mod.intercept_)
print(logistic_mod.coef_)


probabilities = logistic_mod.predict_proba(X_test)
print('Probability of scores for 0 and 1 in first and second columns: \n ' +str(probabilities[:15,:]))


### The first column is the probability of a score of 0  and the second is the probability
### of a score of 1. Notice, most, but not all, probability of a score of  0 is higher
##   than for 1.

###9.  Score and Evaluate the Classification model.

def score_model(probs, threshold):
    return np.array([1 if x > threshold else 0 for x in probs[:, 1]])
scores = score_model(probabilities, 0.5)    #Setting the threshold to 0.5
print('Printing scores: \n' + str(np.array(scores[:15])))
print('This matrix, y_test, tells if the value and the scores matched: \n' + str(y_test[:15]))

#Some of the positive(1) predictions agree with the test labels in the second row,
##   but several do not.

###  10.  Analysis Metrics for Performance:

            ### compute and display forementioned classifier performance with
            ### precision_recall_fscore_support and accuracy_score functions from
            ###   "metrics" package of SCIKIT-LEARN
def print_metrics(labels, scores):
    metrics = sklm.precision_recall_fscore_support(labels, scores)
    conf= sklm.confusion_matrix(labels, scores)
    print('                 Confusion Matrix')
    print('                 Score positive     Score negative')
    print('Actual positive   %6d' % conf[0,0] + '         %5d' % conf[0,1])
    print('Actual negative   %6d' % conf[1,0] + '         %5d' % conf[1,1])
    print('')
    print('Accuracy %0.2f'  % sklm.accuracy_score(labels, scores))
    print('')
    print('         Positive     Negative')
    print('Num case    %6d'  % metrics[3][0] + '       %6d' %metrics[3][1])   # 6d = whole numbers
    print('Precision   %6.2f'  % metrics[0][0] + '     %6.2f' %metrics[0][1]) # 6.2f = 2 decimal places
    print('Recall      %6.2f'  % metrics[1][0] + '     %6.2f' %metrics[1][1])
    print('F1          %6.2f'  % metrics[2][0] + '     %6.2f' %metrics[2][1])


print_metrics(y_test, scores)


        ###   ROC and AUC :
def plot_auc(labels, probs):
    ## Compute the false positive rate, true positive rate
    ## and threshold along with the AUC
    fpr, tpr, threshold = sklm.roc_curve(labels, probs[:, 1])
    auc = sklm.auc(fpr, tpr)

    ## plotting the results.
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, color= 'orange', label= 'AUC = %0.2f' %auc)
    plt.legend(loc = 'lower right')
    plt.plot([0,1], [0,1],'r--')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

#plot_auc(y_test, probabilities)    



        ### 11. Computing a WEIGHTED MODEL:
            ## chosing weights of 0.3 and 0.7 for good and bad credit respectively.


logistic_mod = linear_model.LogisticRegression(class_weight= {0:0.3, 1:0.7})
logistic_mod.fit(X_train, y_train)

         ##display the class probabilities for each case.
probabilities = logistic_mod.predict_proba(X_test)
print(probabilities[:15,:])

        ### Use the metrics as earlier to find if there is any significant difference with the
        ### unweighted model.

    ### COMMENT THE 3 LINES BELOW TO RUN THE LOOP FOR THRESHOLD.

scores= score_model(probabilities, 0.5)
print_metrics(y_test, scores)
plot_auc(y_test, probabilities)

        ### accuracy and precision dropped but, 'Recall' has significantly improved for the
        ## positive label (bad credit or 1-column)


##    ##12. Finding a better Threshold:
##def test_threshold(probs, labels, threshold):
##    scores = score_model(probs, threshold)
##    print('')
##    print('For threshold = ' + str(threshold))
##    print_metrics(labels, scores)
##
##
##thresholds= [0.45, 0.40, 0.35, 0.3, 0.25]
##for t in thresholds:
##    test_threshold(probabilities, y_test, t)


        ## for 0.4 and 0.35, the model improves on its classification of bad credit
            ## customer without severly missclassifying good customers as well. 














