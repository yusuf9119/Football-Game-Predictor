import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt




import warnings
warnings.filterwarnings('ignore')

#importing the data
df = pd.read_csv('soccerdata.csv')
df['date'] = pd.to_datetime(df['date'])



#reading the dataset and selects relevant columns
rawmatchstats = df[['date',
                   'match_id',
                   'home_team_name',
                   'away_team_name',
                   'home_team_goal_count',
                   'away_team_goal_count',
                   'home_team_half_time_goal_count',
                   'away_team_half_time_goal_count',
                   'home_team_shots',
                   'away_team_shots',
                   'home_team_shots_on_target',
                   'away_team_shots_on_target',
                   'home_team_fouls',
                   'away_team_fouls',
                   'home_team_corner_count',
                   'away_team_corner_count',
                   'home_team_yellow',
                   'away_team_yellow',
                   'home_team_red',
                   'away_team_red']]


#cleaning the data
rawmatchstats = rawmatchstats.sort_values(by=['date'],ascending=False)
cleaneddata = rawmatchstats.dropna()


#splitting the dataset to training and testing

X = cleaneddata['home_team_goal_count']
y = cleaneddata['home_team_shots_on_target']
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2,random_state=12)

X_train = np.array(X_train).reshape(-1,1)
y_train = np.array(y_train).reshape(-1,1)
X_test = np.array(X_test).reshape(-1,1)
y_test = np.array(y_test).reshape(-1,1)

#building models
#Linearreg
linearmodel = LinearRegression()
linearmodel.fit(X_train,y_train)
linearpredictions = linearmodel.predict(X_test)

#KNN
knnmodel = KNeighborsRegressor()
knnmodel.fit(X_train,y_train)
knnnpredictions = knnmodel.predict(X_test)

#random forest model
rfmodel = RandomForestClassifier()
rfmodel.fit(X_train,y_train)
rfpredictions = rfmodel.predict(X_test)

svmmodel = SVR()
svmmodel.fit(X_train,y_train)
svmpredictions = svmmodel.predict(X_test)



#evauate the models using mse,r2
def evaluatemodels(predictions):
    mse = mean_squared_error(y_test,predictions)
    r2 = r2_score(y_test,predictions)
    return mse,r2


linearmse,linearr2 = evaluatemodels(linearpredictions)
knnmse,knnr2 = evaluatemodels(knnnpredictions)
rfmse,rfr2 = evaluatemodels(rfpredictions)
svmmse,svmr2 = evaluatemodels(svmpredictions)


#test predictions
print("linear regression:")
print(f"Mean Squared Error (MSE): {linearmse}")
print(f"r-squared (R2): {linearr2}")

print("nearest neightbours regressor")
print(f"mean squared error(MSE) {knnmse}")
print(f"R-Squared (R2) {knnr2}")

print("random forest regressor")
print(f"mean squared eror (MSE) {rfmse}")
print(f"R-Squared (R2) {rfr2}")

print("Support Vector Machine:")
print(f"Mean Squared Error {svmmse}")
print(f"R-Squared (R2) {svmr2}")


plt.figure(figsize=(8,6))
plt.scatter(y_test,linearpredictions,color='blue')
plt.xlabel('actual goals scored')
plt.ylabel('predicted goals from shots')
plt.title('Linear Regresssion: Premier League goals per shot predictions')
plt.show()


plt.figure(figsize=(8,6))
plt.scatter(y_test,knnnpredictions,color='red')
plt.xlabel('actual goals scored')
plt.ylabel('predicted goals from shots')
plt.title('K Nearest Neightbours: Premier League goals per shot predictions')
plt.show()

plt.figure(figsize=(8,6))
plt.scatter(y_test,rfpredictions,color='green')
plt.xlabel('actual goals scored')
plt.ylabel('predicted goals from shots')
plt.title('Random Forest Regressor: Premier League goals per shot predictions')
plt.show()


plt.figure(figsize=(8,6))
plt.scatter(y_test,svmpredictions,color='yellow')
plt.xlabel('actual goals scored')
plt.ylabel('predicted goals from shots')
plt.title('SVR : Premier League goals per shot predictions')
plt.show()


