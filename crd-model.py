#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st 
from sqlalchemy import create_engine


import joblib   # save and load ML models
import gc       # garbage collection
import os 
import sklearn

# preprocessing steps
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Import the required libraries

from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, precision_recall_curve, auc
from sklearn.feature_selection import f_classif
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import chi2_contingency



df=pd.read_csv('https://afexdataset.s3.eu-central-1.amazonaws.com/loan_data.csv',error_bad_lines=False, engine="python")
pd.options.display.max_columns = None

pd.options.display.max_columns = None
# get a list of columns that have more than 80% null values

#dropping  spme irrelevant columns from loan_loan data
df = df.drop(columns =['rejected_by_id','rejection_reason','next_approval','approval_done','approval_permissions','is_deleted','created','updated','created_offline','maturity_date','created_by_id','ln_id','approval_date','rejected_date'])
df.rename( columns = {'is_repaid' : 'Target'}, inplace = True)
df=df[['farmer_id','id','repayment_value','total_loan_value','amount_repaid','Target','insurance','crg','interest','admin_fee','equity','value_chain_management','hectare','data_identification_verification','contacted','is_approved','is_approval_completed','is_reverted','is_matured_processed','project_id','warehouse_id','is_rejected']]


import streamlit as st
pd.options.display.max_columns = 100
pd.options.display.max_rows = 900
pd.set_option('float_format', '{:f}'.format)

# Print shape and description of the data
st.write("""
# Credit Scoring App
## Data Snapshot

	""")

if st.sidebar.checkbox('Show what the dataframe looks like'):
    st.write(df.head(100))
    st.write('Shape of the dataframe: ',df.shape)

import pandas as pd
from woe_scoring import WOETransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
train, test = train_test_split(df, test_size=0.3, random_state=42, stratify= df['Target'])

special_cols = [
    "farmer_id",
    "Target",
    "project_id",
    "id",
    "warehouse_id"
]

cat_cols = [
    "contacted",
    "is_rejected",
    "is_reverted",
    "is_approved",
    "is_approval_completed",
    "is_matured_processed"
    
]

encoder = WOETransformer(
    max_bins=8,
    min_pcnt_group=0.1,
    cat_features=cat_cols,
    special_cols=special_cols
)

encoder.fit(train, train["Target"])
encoder.save("train_dict.json")

enc_train = encoder.transform(train.drop(special_cols, axis=1))
enc_test = encoder.transform(test.drop(special_cols, axis=1))

model = LogisticRegression()
model.fit(enc_train, train["Target"])
test_proba = model.predict_proba(enc_test)[:, 1]

#('Probability of Default and Non Default is :', test_proba)


if st.sidebar.checkbox('Show the probability of Loan Default and Non Default'):
    st.write('Probabilities of Default and Non Default is ')
    st.write(test_proba)
    
    
#Socrecard
# creating a new column with numeric entities
dff =df
dff['target'] =  [1 if line == True else 0 for line in dff.Target]
dff['grade'] = ['A' if line <= 4 else 'C' for line in dff.hectare]
# Dropping the former target variable
dff= dff.drop(columns = ['Target'])

# split data into 80/20 while keeping the distribution of bad loans in test set same as that in the pre-split dataset
X = dff.drop('target', axis = 1)
y = dff ['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

# specifically hard copying the training sets to avoid Pandas' SetttingWithCopyWarning when we play around with this data later on.
# as noted [here](https://github.com/scikit-learn/scikit-learn/issues/8723), this is currently an open issue between Pandas and Scikit-Learn teams
X_train, X_test = X_train.copy(), X_test.copy()


# first divide training data into categorical and numerical subsets
X_train_cat = X_train.select_dtypes(include = 'object').copy()
X_train_num = X_train.select_dtypes(include = 'number').copy()

X_train_cat = dff[["contacted",
    "is_rejected",
    "is_reverted",
    "is_approved",
    "is_approval_completed",
    "is_matured_processed"]]

# define an empty dictionary to store chi-squared test results
chi2_check = {}

# loop over each column in the training set to calculate chi-statistic with the target variable
for column in X_train_cat:
    chi, p, dof, ex = chi2_contingency(pd.crosstab(y_train, X_train_cat[column]))
    chi2_check.setdefault('Feature',[]).append(column)
    chi2_check.setdefault('p-value',[]).append(round(p, 10))

# convert the dictionary to a DF
chi2_result = pd.DataFrame(data = chi2_check)
chi2_result.sort_values(by = ['p-value'], ascending = True, ignore_index = True, inplace = True)

# since f_class_if does not accept missing values, we will do a very crude imputation of missing values
X_train_num.fillna(X_train_num.mean(), inplace = True)
# Calculate F Statistic and corresponding p values
F_statistic, p_values = f_classif(X_train_num, y_train)
# convert to a DF
ANOVA_F_table = pd.DataFrame(data = {'Numerical_Feature': X_train_num.columns.values, 'F-Score': F_statistic, 'p values': p_values.round(decimals=10)})
ANOVA_F_table.sort_values(by = ['F-Score'], ascending = False, ignore_index = True, inplace = True)


# save the top 20 numerical features in a list
top_num_features = ANOVA_F_table.iloc[:20,0].to_list()
# calculate pair-wise correlations between them
corrmat = X_train_num[top_num_features].corr()
plt.figure(figsize=(10,10))
sns.heatmap(corrmat);

# Define a helper function to drop the 4 categorical features with least p-values for chi squared test, 14 numerical features with least F-Statistic
# and 2 numerical features with high multicollinearity
drop_columns_list = ANOVA_F_table.iloc[20:, 0].to_list()
drop_columns_list.extend(chi2_result.iloc[4:, 0].to_list())
drop_columns_list.extend(['admin_fee', 'value_chain_management'])

def col_to_drop(dff, columns_list):
    dff.drop(columns = columns_list, inplace = True)

# apply to X_train
col_to_drop(X_train, drop_columns_list)

# function to create dummy variables
def dummy_creation(df, columns_list):
    df_dummies = []
    for col in columns_list:
        df_dummies.append(pd.get_dummies(df[col], prefix = col, prefix_sep = ':'))
    df_dummies = pd.concat(df_dummies, axis = 1)
    df = pd.concat([df, df_dummies], axis = 1)
    return df

# apply to our final four categorical variables
X_train = dummy_creation(X_train, ['is_approved', 'is_reverted', 'is_rejected','grade','contacted'])


col_to_drop(X_test, drop_columns_list)
X_test = dummy_creation(X_test, ['is_approved','grade', 'is_reverted','is_rejected','contated'])
# reindex the dummied test set variables to make sure all the feature columns in the train set are also available in the test set
X_test = X_test.reindex(labels=X_train.columns, axis=1, fill_value=0)

# Create copies of the 4 training sets to be preprocessed using WoE
X_train_prepr = X_train.copy()
y_train_prepr = y_train.copy()
X_test_prepr = X_test.copy()
y_test_prepr = y_test.copy()


# The function takes 3 arguments: a dataframe (X_train_prepr), a string (column name), and a dataframe (y_train_prepr).
# The function returns a dataframe as a result.
def woe_discrete(df, cat_variabe_name, y_df):
    df = pd.concat([df[cat_variabe_name], y_df], axis = 1)
    df = pd.concat([df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].count(),
                    df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].mean()], axis = 1)
    df = df.iloc[:, [0, 1, 3]]
    df.columns = [df.columns.values[0], 'n_obs', 'prop_good']
    df['prop_n_obs'] = df['n_obs'] / df['n_obs'].sum()
    df['n_good'] = df['prop_good'] * df['n_obs']
    df['n_bad'] = (1 - df['prop_good']) * df['n_obs']
    df['prop_n_good'] = df['n_good'] / df['n_good'].sum()
    df['prop_n_bad'] = df['n_bad'] / df['n_bad'].sum()
    df['WoE'] = np.log(df['prop_n_good'] / df['prop_n_bad'])
    df = df.sort_values(['WoE'])
    df = df.reset_index(drop = True)
    df['diff_prop_good'] = df['prop_good'].diff().abs()
    df['diff_WoE'] = df['WoE'].diff().abs()
    df['IV'] = (df['prop_n_good'] - df['prop_n_bad']) * df['WoE']
    df['IV'] = df['IV'].sum()
    return df


# We set the default style of the graphs to the seaborn style. 
sns.set()
# Below we define a function for plotting WoE across categories that takes 2 arguments: a dataframe and a number.
def plot_by_woe(df_WoE, rotation_of_x_axis_labels = 0):
    x = np.array(df_WoE.iloc[:, 0].apply(str))
    y = df_WoE['WoE']
    plt.figure(figsize=(18, 6))
    plt.plot(x, y, marker = 'o', linestyle = '--', color = 'k')
    plt.xlabel(df_WoE.columns[0])
    plt.ylabel('Weight of Evidence')
    plt.title(str('Weight of Evidence by ' + df_WoE.columns[0]))
    plt.xticks(rotation = rotation_of_x_axis_labels)   
    
    
df_temp = woe_discrete(X_train_prepr, 'contacted', y_train_prepr)



# We define a function to calculate WoE of continuous variables. This is same as the function we defined earlier for discrete variables.
# The only difference are the 2 commented lines of code in the function that results in the df being sorted by continuous variable values
def woe_ordered_continuous(df, continuous_variabe_name, y_df):
    df = pd.concat([df[continuous_variabe_name], y_df], axis = 1)
    df = pd.concat([df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].count(),
                    df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].mean()], axis = 1)
    df = df.iloc[:, [0, 1, 3]]
    df.columns = [df.columns.values[0], 'n_obs', 'prop_good']
    df['prop_n_obs'] = df['n_obs'] / df['n_obs'].sum()
    df['n_good'] = df['prop_good'] * df['n_obs']
    df['n_bad'] = (1 - df['prop_good']) * df['n_obs']
    df['prop_n_good'] = df['n_good'] / df['n_good'].sum()
    df['prop_n_bad'] = df['n_bad'] / df['n_bad'].sum()
    df['WoE'] = np.log(df['prop_n_good'] / df['prop_n_bad'])
    #df = df.sort_values(['WoE'])
    #df = df.reset_index(drop = True)
    df['diff_prop_good'] = df['prop_good'].diff().abs()
    df['diff_WoE'] = df['WoE'].diff().abs()
    df['IV'] = (df['prop_n_good'] - df['prop_n_bad']) * df['WoE']
    df['IV'] = df['IV'].sum()
    return df

df_temp = woe_ordered_continuous(X_train_prepr, 'crg', y_train_prepr)

# create a list of all the reference categories, i.e. one category from each of the global features
ref_categories = ['insurance:>79,780', 'equity:>7,260', 'total_loan_value:>25,000', 'repayment_value:>15,437', 'admision_fee:>875', 
                  'amount_repaid:>150K', 'interest:>20.281', 'total_identification_verification:3000', 'is_approved:True', 'contacted: True', 'grade:A']


# This custom class will create new categorical dummy features based on the cut-off points that we manually identified
# based on the WoE plots and IV above.
# Given the way it is structured, this class also allows a fit_transform method to be implemented on it, thereby allowing 
# us to use it as part of a scikit-learn Pipeline 
class WoE_Binning(BaseEstimator, TransformerMixin):
    def __init__(self, X): # no *args or *kargs
        self.X = X
    def fit(self, X, y = None):
        return self #nothing else to do
    def transform(self, X):
        X_new = X.loc[:, 'grade:A': 'grade:C']
        X_new = pd.concat([X_new, X.loc[:, 'is_approved:False':'is_approved:True']], axis = 1)
        # For the purpose of this column, we keep debt_consolidation (due to volume) and credit_card (due to unique characteristics) as separate cateogories
        # These categories have very few observations: educational, renewable_energy, vacation, house, wedding, car
        # car is the least risky so we will combine it with the other 2 least risky categories: home_improvement and major_purchase
        # educational, renewable_energy (both low observations) will be combined with small_business and moving
        # vacation, house and wedding (remaining 3 with low observations) will be combined with medical and other
        X_new['is_reverted:True'] = X.loc[:,'is_reverted:True']
        X_new['is_reverted:False'] = X.loc[:,'is_reverted:False']
        
        X_new['contacted:True'] = X.loc[:,'contacted:True']
        X_new['contacted:False'] = X.loc[:,'contacted:False']
    
        X_new['interest:<7,071'] = np.where((X['interest'] <= 7071), 1, 0)
        X_new['interest :7,071-10,374'] = np.where((X['interest'] > 7071) & (X['interest'] <= 10374), 1, 0)
        X_new['interest:10,374-13.676'] = np.where((X['interest'] > 10374) & (X['interest'] <= 13676), 1, 0)
        X_new['interest:13,676-15.74'] = np.where((X['interest'] > 13676) & (X['interest'] <= 15.74), 1, 0)
        X_new['interest:15,74-20281'] = np.where((X['interest'] > 1574) & (X['interest'] <= 20.281), 1, 0)
        X_new['interest:>20,281'] = np.where((X['interest'] > 20281), 1, 0)
        
        X_new['repayment_value :missing'] = np.where(X['repayment_value'].isnull(), 1, 0)
        X_new['repayment_value:<28,555'] = np.where((X['repayment_value'] <= 28555), 1, 0)
        X_new['repayment_value:28,555-37,440'] = np.where((X['repayment_value'] > 28555) & (X['repayment_value'] <= 37440), 1, 0)
        X_new['repayment_value:37,440-61,137'] = np.where((X['repayment_value'] > 37440) & (X['repayment_value'] <= 61137), 1, 0)
        X_new['repayment_value:61,137-81,872'] = np.where((X['repayment_value'] > 61137) & (X['repayment_value'] <= 81872), 1, 0)
        X_new['repayment_value:81,872-102,606'] = np.where((X['repayment_value'] > 81872) & (X['repayment_value'] <= 102606), 1, 0)
        X_new['repayment_value:102,606-120,379'] = np.where((X['repayment_value'] > 102606) & (X['repayment_value'] <= 120379), 1, 0)
        X_new['repayment_value:120,379-150,000'] = np.where((X['repayment_value'] > 120379) & (X['repayment_value'] <= 150000), 1, 0)
        X_new['repayment_value:>150K'] = np.where((X['repayment_value'] > 150000), 1, 0)
        
        X_new['equity:<1,286'] = np.where((X['equity'] <= 1286), 1, 0)
        X_new['equity:1,286-6,432'] = np.where((X['equity'] > 1286) & (X['equity'] <= 6432), 1, 0)
        X_new['equity:6,432-9,005'] = np.where((X['equity'] > 6432) & (X['equity'] <= 9005), 1, 0)
        X_new['equity:9,005-10,291'] = np.where((X['equity'] > 9005) & (X['equity'] <= 10291), 1, 0)
        X_new['equity:10,291-15,437'] = np.where((X['equity'] > 10291) & (X['equity'] <= 15437), 1, 0)
        X_new['equity:>15,437'] = np.where((X['equity'] > 15437), 1, 0)
        
        X_new['amount_repaid :<10,000'] = np.where((X['amount_repaid'] <= 10000), 1, 0)
        X_new['amount_repaid:10,000-15,000'] = np.where((X['amount_repaid'] > 10000) & (X['amount_repaid'] <= 15000), 1, 0)
        X_new['amount_repaid:15,000-20,000'] = np.where((X['amount_repaid'] > 15000) & (X['amount_repaid'] <= 20000), 1, 0)
        X_new['amount_repaid:20,000-25,000'] = np.where((X['amount_repaid'] > 20000) & (X['amount_repaid'] <= 25000), 1, 0)
        X_new['amount_repaid:>25,000'] = np.where((X['amount_repaid'] > 25000), 1, 0)
        
        X_new['total_loan_value:<1,089'] = np.where((X['total_loan_value'] <= 1089), 1, 0)
        X_new['total_loan_value:1,089-2,541'] = np.where((X['total_loan_value'] > 1089) & (X['total_loan_value'] <= 2541), 1, 0)
        X_new['total_loan_value:2,541-4,719'] = np.where((X['total_loan_value'] > 2541) & (X['total_loan_value'] <= 4719), 1, 0)
        X_new['total_loan_value:4,719-7,260'] = np.where((X['total_loan_value'] > 4719) & (X['total_loan_value'] <= 7260), 1, 0)
        X_new['total_loan_value:>7,260'] = np.where((X['total_loan_value'] > 7260), 1, 0)
        X_new['total_loan_value:missing'] = np.where(X['total_loan_value'].isnull(), 1, 0)
        X_new['total_loan_value:<6,381'] = np.where((X['total_loan_value'] <= 6381), 1, 0)
        X_new['total_loan_value:6,381-19,144'] = np.where((X['total_loan_value'] > 6381) & (X['total_loan_value'] <= 19144), 1, 0)
        X_new['total_loan_value:19,144-25,525'] = np.where((X['total_loan_value'] > 19144) & (X['total_loan_value'] <= 25525), 1, 0)
        X_new['total_loan_value:25,525-35,097'] = np.where((X['total_loan_value'] > 25525) & (X['total_loan_value'] <= 35097), 1, 0)
        X_new['total_loan_value:35,097-54,241'] = np.where((X['total_loan_value'] > 35097) & (X['total_loan_value'] <= 54241), 1, 0)
        X_new['total_loan_value:54,241-79,780'] = np.where((X['total_loan_value'] > 54241) & (X['total_loan_value'] <= 79780), 1, 0)
        X_new['total_loan_value:>79,780'] = np.where((X['total_loan_value'] > 79780), 1, 0)
        
        return X_new
# we could have also structured this class without the last drop statement and without creating categories out of the 
# feature categories. But doing the way we have done here allows us to keep a proper track of the categories, if required

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# define modeling pipeline
reg = LogisticRegression(max_iter=1000, class_weight = 'balanced')
woe_transform = WoE_Binning(X)
pipeline = Pipeline(steps=[('woe', woe_transform), ('model', reg)])

# define cross-validation criteria. RepeatedStratifiedKFold automatially takes care of the class imbalance while splitting
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)

# fit and evaluate the logistic regression pipeline with cross-validation as defined in cv
scores = cross_val_score(pipeline, X_train, y_train, scoring = 'roc_auc', cv = cv)
AUROC = np.mean(scores)
GINI = AUROC * 2 - 1



if st.sidebar.checkbox('Show the accuracy of the trained model'):
    st.write('## Model Evaluation')
    st.write('Mean AUROC: %.4f' % (AUROC))
    st.write('Gini: %.4f' % (GINI))
    
    
    
pipeline.fit(X_train, y_train)

# first create a transformed training set through our WoE_Binning custom class
X_train_woe_transformed = woe_transform.fit_transform(X_train)
# Store the column names in X_train as a list
feature_name = X_train_woe_transformed.columns.values
# Create a summary table of our logistic regression model
summary_table = pd.DataFrame(columns = ['Feature name'], data = feature_name)
# Create a new column in the dataframe, called 'Coefficients', with row values the transposed coefficients from the 'LogisticRegression' model
summary_table['Coefficients'] = np.transpose(pipeline['model'].coef_)
# Increase the index of every row of the dataframe with 1 to store our model intercept in 1st row
summary_table.index = summary_table.index + 1
# Assign our model intercept to this new row
summary_table.loc[0] = ['Intercept', pipeline['model'].intercept_[0]]
# Sort the dataframe by index
summary_table.sort_index(inplace = True)

# make preditions on our test set
y_hat_test = pipeline.predict(X_test)
# get the predicted probabilities
y_hat_test_proba = pipeline.predict_proba(X_test)
# select the probabilities of only the positive class (class 1 - default) 
y_hat_test_proba = y_hat_test_proba[:][: , 1]

    
# we will now create a new DF with actual classes and the predicted probabilities
# create a temp y_test DF to reset its index to allow proper concaternation with y_hat_test_proba
y_test_temp = y_test.copy()
y_test_temp.reset_index(drop = True, inplace = True)
y_test_proba = pd.concat([y_test_temp, pd.DataFrame(y_hat_test_proba)], axis = 1)
# check the shape to make sure the number of rows is same as that in y_test


# Rename the columns
y_test_proba.columns = ['y_test_class_actual', 'y_hat_test_proba']
# Makes the index of one dataframe equal to the index of another dataframe.
y_test_proba.index = X_test.index
y_test_proba.head()

# assign a threshold value to differentiate good with bad
tr = 0.5
# crate a new column for the predicted class based on predicted probabilities and threshold
# We will determine this optimat threshold later in this project
y_test_proba['y_test_class_predicted'] = np.where(y_test_proba['y_hat_test_proba'] > tr, 1, 0)
# create the confusion matrix
confusion_matrix(y_test_proba['y_test_class_actual'], y_test_proba['y_test_class_predicted'], normalize = 'all')

# get the values required to plot a ROC curve
fpr, tpr, thresholds = roc_curve(y_test_proba['y_test_class_actual'], y_test_proba['y_hat_test_proba'])
# plot the ROC curve
plt.plot(fpr, tpr)
# plot a secondary diagonal line, with dashed line style and black color to represent a no-skill classifier
plt.plot(fpr, fpr, linestyle = '--', color = 'k')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve');

# Calculate the Area Under the Receiver Operating Characteristic Curve (AUROC) on our test set
AUROC = roc_auc_score(y_test_proba['y_test_class_actual'], y_test_proba['y_hat_test_proba'])
AUROC

# calculate Gini from AUROC
Gini = AUROC * 2 - 1
Gini

# draw a PR curve
# calculate the no skill line as the proportion of the positive class
no_skill = len(y_test[y_test == 1]) / len(y)
# plot the no skill precision-recall curve
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')

# calculate inputs for the PR curve
precision, recall, thresholds = precision_recall_curve(y_test_proba['y_test_class_actual'], y_test_proba['y_hat_test_proba'])
# plot PR curve
plt.plot(recall, precision, marker='.', label='Logistic')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.title('PR curve');

# calculate PR AUC
auc_pr = auc(recall, precision)

# We create a new dataframe with one column. Its values are the values from the 'reference_categories' list. We name it 'Feature name'.
df_ref_categories = pd.DataFrame(ref_categories, columns = ['Feature name'])
# We create a second column, called 'Coefficients', which contains only 0 values.
df_ref_categories['Coefficients'] = 0


# Concatenates two dataframes.
df_scorecard = pd.concat([summary_table, df_ref_categories])
# We reset the index of a dataframe.
df_scorecard.reset_index(inplace = True)


# create a new column, called 'Original feature name', which contains the value of the 'Feature name' column, up to the column symbol.
df_scorecard['Original feature name'] = df_scorecard['Feature name'].str.split(':').str[0]

# Define the min and max threshholds for our scorecard
min_score = 300
max_score = 850


# calculate the sum of the minimum coefficients of each category within the original feature name
min_sum_coef = df_scorecard.groupby('Original feature name')['Coefficients'].min().sum()
# calculate the sum of the maximum coefficients of each category within the original feature name
max_sum_coef = df_scorecard.groupby('Original feature name')['Coefficients'].max().sum()
# create a new columns that has the imputed calculated Score based on the multiplication of the coefficient by the ratio of the differences between
# maximum & minimum score and maximum & minimum sum of cefficients.
df_scorecard['Score - Calculation'] = df_scorecard['Coefficients'] * (max_score - min_score) / (max_sum_coef - min_sum_coef)
# update the calculated score of the Intercept (i.e. the default score for each loan)
df_scorecard.loc[0, 'Score - Calculation'] = ((df_scorecard.loc[0,'Coefficients'] - min_sum_coef) / (max_sum_coef - min_sum_coef)) * (max_score - min_score) + min_score
# round the values of the 'Score - Calculation' column and store them in a new column
df_scorecard['Score - Preliminary'] = df_scorecard['Score - Calculation'].round()


# check the min and max possible scores of our scorecard
if st.sidebar.checkbox('Check the Minimum and Maximum score'):   
    min_sum_score_prel = df_scorecard.groupby('Original feature name')['Score - Preliminary'].min().sum()
    max_sum_score_prel = df_scorecard.groupby('Original feature name')['Score - Preliminary'].max().sum()
    st.write ('The minimum Score is ', min_sum_score_prel)
    st.write ('The maximum Score is ', max_sum_score_prel)


# so both our min and max scores are out by +1. we need to manually adjust this
# Which one? We'll evaluate based on the rounding differences of the minimum category within each Original Feature Name.
pd.options.display.max_rows = 102
df_scorecard['Difference'] = df_scorecard['Score - Preliminary'] - df_scorecard['Score - Calculation']



# look like we can get by deducting 1 from the Intercept

if st.sidebar.checkbox('Score the clients'):
    df_scorecard['Score - Final'] = df_scorecard['Score - Preliminary']
    df_scorecard.loc[0, 'Score - Final'] = 598
    st.write('''## Scorecard''')
    st.write(dff.join(df_scorecard['Score - Final']))

