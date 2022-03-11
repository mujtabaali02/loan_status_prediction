import pickle
import matplotlib.pyplot as plt
import pandas as pd
from src import config

df=pd.read_csv('data/process_data.csv')
df.head()
df=df.drop(columns=['Unnamed: 0'],axis=1)
df.head()
df.describe()

# df.shape
x=df.drop(columns=['Status'],axis=1)
y=df['Status']

from sklearn.preprocessing import StandardScaler
standard_scaler=StandardScaler()
arr=standard_scaler.fit_transform(x)
df=pd.DataFrame(data=arr,columns=x.columns)
df.head()
df.describe()


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=100)
# x_train.head()

# LogisticRegression

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)
lr.score(x_train,y_train)*100 #82.05
lr.score(x_test,y_test)*100 #77.83
pickle.dump(lr,open("models/logistic_regression",'wb'))


from sklearn.tree import DecisionTreeClassifier
dtree_clsf=DecisionTreeClassifier(ccp_alpha=.0234,random_state=100)
dtree_clsf.fit(x_train,y_train)
dtree_clsf.score(x_train,y_train)*100 # 82.2
dtree_clsf.score(x_test,y_test)*100 #77.83
pickle.dump(dtree_clsf,open('models/dtree_cls','wb'))


path=dtree_clsf.cost_complexity_pruning_path(x_train,y_train)
ccp_alphas=path.ccp_alphas
dt_model=[]
for i in ccp_alphas:
 mod=DecisionTreeClassifier(ccp_alpha=i)
 mod.fit(x_train,y_train)
 dt_model.append(mod)

train_score=[model.score(x_train,y_train) for model in dt_model]
test_score=[model.score(x_test,y_test) for model in dt_model]

fig,ax=plt.subplots()
ax.set_xlabel("ccp_alpha")
ax.set_ylabel("score/accuracy")
ax.plot(ccp_alphas,train_score,marker="o",label="train")
ax.plot(ccp_alphas,test_score,marker="o",label="test")

# not for performing it by using this algorithm
from sklearn.ensemble import RandomForestClassifier
rnd_forst=RandomForestClassifier(n_estimators=100)
rnd_forst.fit(x_train,y_train)
rnd_forst.score(x_train,y_train)
rnd_forst.score(x_test,y_test)

# Not good for this project
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(x_train,y_train)
knn.score(x_train,y_train)
knn.score(x_test,y_test)


# process and training the data because database is not supporting Y and N

df_process=pd.read_csv('data/process_data.csv')
pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",None)
print(df_process.head())
# df_process['Status']=df_process['Status'].map({0:'N',1:'Y'})
df_process=pd.get_dummies(df_process,columns=['Status'],drop_first=True)
print(df_process.head())
df_process=df_process .rename(columns={'Status_Y':'Result'})

# Convert into X and y
x=df_process.drop(columns=['Result'],axis=1)
y=df_process['Result']
print(x.shape)

# Apply the feature transformation

standard_scaler=StandardScaler()
arr=standard_scaler.fit_transform(x)
print(arr)
df1=pd.DataFrame(data=arr,columns=x.columns)
print(df1.head())