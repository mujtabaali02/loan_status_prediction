import numpy as np
from src import config
import pandas as pd
import seaborn as sns

df=pd.read_csv(config.DATA)
df.head()

pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",None)

df=df.drop(columns=['Loan_ID'],axis=1)
df.describe()

# missing value imputation / handling it

miss=df.isnull().sum().sort_values(ascending=False)
per=df.isnull().mean().sort_values(ascending=False)
pd.concat([miss,per*100,df.dtypes],axis=1,keys=['Missing_Value','Percentage','DataType'])

types=['int64','float64']
df_num=df.select_dtypes(include=types).columns
# df_num
miss_num_cols=[col for col in df_num if df[col].isnull().sum()>0]
# miss_num_cols

sns.distplot(df['LoanAmount'])
sns.distplot(df['Loan_Amount_Term'])
sns.distplot(df['Credit_History'])

from sklearn.impute import SimpleImputer
imputer_for_num=SimpleImputer()
arr_num=imputer_for_num.fit_transform(df[miss_num_cols])
df_num=pd.DataFrame(data=arr_num,columns=miss_num_cols)
# df_num


df_cat_cols=df.select_dtypes(include="O").columns
# df_cat_cols

mis_cat_cols=[col for col in df_cat_cols if df[col].isnull().sum()>0]
# mis_cat_cols
imputer_cat=SimpleImputer(strategy='most_frequent')
arr_cat=imputer_cat.fit_transform(df[mis_cat_cols])
# arr_cat

df_cat=pd.DataFrame(data=arr_cat,columns=mis_cat_cols)
# df_cat
df_cat['Dependents']=df['Dependents'].replace("3+",4)
df.head()
df_cat['Dependents']=df_cat['Dependents'].fillna(df['Dependents'].mode()[0])
# df_cat['Dependents']

df_cat.head()
df_cat=pd.get_dummies(data=df_cat,columns=['Gender','Married','Self_Employed'],drop_first=True)
df_cat.head()
df_cat['Status']=df['Loan_Status']
df_cat.head()
df1=pd.concat([df_num,df_cat],ignore_index=True,axis=1)
df1.head()
df_cat.head()
df_clean=df1.copy()
df1.columns=['LoanAmount', 'Loan_Amount_Term', 'Credit_History','Dependents','Gender_Male','Married_Yes','Self_Employed_Yes', 'Status']
# df_num.columns
# df_cat.columns
# df_num.columns

# df_cat.columns
# df_num.columns
# df1.isnull().sum()

# df1.to_csv(config.PROCESS_DATA)
df1.to_csv('data/process_data.csv')



