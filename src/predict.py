import pickle
model=pickle.load(open('models/logistic_regression','rb'))
model.predict([[158.0,360.0,0.0,4,1,1,0]])[0]
model.predict([[95.0,360.0,1.0,0,1,1,0]])[0]