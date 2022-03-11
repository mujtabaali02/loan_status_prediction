import pickle
import numpy as np
import mysql.connector as con
import pandas as pd
from sklearn.preprocessing import StandardScaler
from flask import Flask,render_template,request
from sklearn.linear_model import LogisticRegression
app=Flask(__name__)

try:
    conn=con.connect(host="localhost",user="root",password="",database="loandb")
    mycursor=conn.cursor()
except :
    print("Error occurred while connecting to database")
# list1=[101.0,360.0,1.0,0,1,1,0,1]
# try:
#     mycursor.execute("insert into customer_entry values(NULL,{},{},{},{},{},{},{},{})".format(list1[0],list1[1],list1[2],list1[3],list1[4],list1[5],list1[6],list1[7]))
#     conn.commit()
#     conn.close()
#     mycursor.close()
# except:
#     print("Error occurred while inserting the data")
# else:
#     print("Inserted Successfully")

# app=Flask(__name__)
model=pickle.load(open('models/logistic_new1.pkl','rb'))
@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/result', methods=['GET','POST'])
def result_page():
    if request.method=="POST":
        values=[float(col) for col in request.form.values()]
        df_result = pd.DataFrame(values)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_result)
        df_result_scaled = pd.DataFrame(scaled_data)
        list1 = df_result_scaled[0].to_list()
        final=np.array([list1])
        result=model.predict(final)
        mycursor.execute(
            "insert into customer_entry values(NULL,{},{},{},{},{},{},{},{})".format(values[0], values[1], values[2], values[3],
                                                                                     values[4], values[5], values[6],
                                                                                     result[0]))
        conn.commit()
        if result[0]==1:
            value1="Wow"
            value2="You Are Selected to get the Loan Have a Greate Day a head!!"
        else:
            value1 = "Sorry"
            value2 = "You are not selected to get the Loan Tray again!!"

    return render_template('result.html',value1=value1,value2=value2)

if __name__=='__main__':
    app.run(debug=True)