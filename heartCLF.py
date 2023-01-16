print("*********LOADING DATASET*************")
import pandas as pd

dataset=pd.read_csv('heart.csv')
# print(dataset.head(5))
print(dataset.info())
print(dataset.describe())

print('Sex: {}'.format(dataset['Sex'].unique()))
print('ChestPainType: {}'.format(dataset['ChestPainType'].unique()))
print('RestingECG: {}'.format(dataset['RestingECG'].unique()))
print('ExerciseAngina: {}'.format(dataset['ExerciseAngina'].unique()))
print('ST_Slope: {}'.format(dataset['ST_Slope'].unique()))


SexNumeric=[]
# print('Row size: {}'.format(len(dataset['HeartDisease'])))
for i in range(len(dataset['HeartDisease'])):
    if dataset['Sex'][i]=='M':
        SexNumeric.append(0)
    else:
        SexNumeric.append(1)


ChestPainTypeNumeric=[]
for i in range(len(dataset['HeartDisease'])):
    if dataset['ChestPainType'][i]=='ATP':
        ChestPainTypeNumeric.append(0)
    elif dataset['ChestPainType'][i]=='NAP':
        ChestPainTypeNumeric.append(1)
    elif dataset['ChestPainType'][i]=='ASY':
        ChestPainTypeNumeric.append(2)
    else:
        ChestPainTypeNumeric.append(3)


RestingECGNumeric=[]
for i in range(len(dataset['HeartDisease'])):
    if dataset['RestingECG'][i]=='Normal':
        RestingECGNumeric.append(0)
    elif dataset['RestingECG'][i]=='ST':
        RestingECGNumeric.append(1)
    else:
        RestingECGNumeric.append(2)


ExerciseAnginaNumeric=[]
for i in range(len(dataset['HeartDisease'])):
    if dataset['ExerciseAngina'][i]=='N':
        ExerciseAnginaNumeric.append(0)
    else:
        ExerciseAnginaNumeric.append(1)


ST_SlopeNumeric=[]
for i in range(len(dataset['HeartDisease'])):
    if dataset['ST_Slope'][i]=='Up':
        ST_SlopeNumeric.append(0)
    elif dataset['ST_Slope'][i]=='Flat':
        ST_SlopeNumeric.append(1)
    else:
        ST_SlopeNumeric.append(2)

# Dee Copy Of Dataset In Order To Avoid Changes In The Real Dataset
datasetHeart=dataset.copy(deep=True)


datasetHeart['Sex']=SexNumeric
datasetHeart['ChestPainType']=ChestPainTypeNumeric
datasetHeart['RestingECG']=RestingECGNumeric
datasetHeart['ExerciseAngina']=ExerciseAnginaNumeric
datasetHeart['ST_Slope']=ST_SlopeNumeric


# print(dataset.head(5))
# print(datasetHeart.head(5))
print(dataset.info())
print(datasetHeart.info())

# SAVE THE CLEANING DATASET>>> NUMERIC DATASET
datasetHeart.to_csv('cleaningHeartDataset.csv')


#PLOTING OF CLEANING DATASET

print('*******PLOTING DATASET*******')
import matplotlib.pyplot as plt 

# datasetHeart.plot(kind='box',subplots=True,layout=(4,4),sharex=False)
# plt.show()

# Cholesterol, Oldpeak and RestingBP

# datasetHeart.hist()
# plt.show()

# PERFORMING CLASSIFICATION TASK
print('************PERFORMING CLASSIFICATION TASK')

datasetHeart=datasetHeart.drop(['Cholesterol','Oldpeak','RestingECG'],axis=1)

x=datasetHeart.drop('HeartDisease',axis=1)
y=dataset['HeartDisease']
# print(x.info())
# print('*****')


#PREPROCESSING DATA INFO STANDAR SCALE
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
x=scaler.fit_transform(x)

#SPLITTING THE DATASET INTO TRAINING AND TESTING
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.7)


from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier() 
model.fit(x_train,y_train) 
yPredictionOnTraining=model.predict(x_train)
yPredictionOnTesting=model.predict(x_test)

from sklearn import metrics
print('TRAIN ACCURACY SCORE:{}'.format(metrics.accuracy_score(y_train,yPredictionOnTraining)))
print('TEST ACCURACY SCORE:{}'.format(metrics.accuracy_score(y_test,yPredictionOnTesting)))
print('CONFUSION MATRIX: \n{}'.format(metrics.confusion_matrix(y_test,yPredictionOnTesting)))


from sklearn.linear_model import LogisticRegression
model=LogisticRegression() 
model.fit(x_train,y_train) 
yPredictionOnTraining=model.predict(x_train)
yPredictionOnTesting=model.predict(x_test)

from sklearn import metrics
print('TRAIN ACCURACY SCORE:{}'.format(metrics.accuracy_score(y_train,yPredictionOnTraining)))
print('TEST ACCURACY SCORE:{}'.format(metrics.accuracy_score(y_test,yPredictionOnTesting)))
print('CONFUSION MATRIX: \n{}'.format(metrics.confusion_matrix(y_test,yPredictionOnTesting)))
