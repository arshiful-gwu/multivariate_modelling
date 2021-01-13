import pandas as pd
import helper as helpme
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import seaborn as sns
from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import arma_estimator as my_arma
from scipy import signal
import statsmodels.api as sm


all_data=pd.read_csv("data/all_data_numeric.csv")
all_data.head()

train, test = train_test_split(all_data, train_size = 0.8, test_size = 0.2, shuffle = False)



X = train[["coffee", "corn", "cotton", "gold", "lumber", "oil", "wheat"]]
X = sm.add_constant(X)
Y=  train["snp"]

x_test=test[["coffee", "corn", "cotton", "gold", "lumber", "oil", "wheat"]]
x_test = sm.add_constant(x_test)

lin_reg_model = sm.OLS(Y,X.astype(float))

results = lin_reg_model.fit()
print (results.params)

test_pred_ols=results.predict(x_test)
test_pred_ols

print(results.summary())

plt.plot(np.concatenate((Y,test["snp"])), label="actual")
plt.plot(np.concatenate((Y,test_pred_ols)), label="prediction")
#plt.plot(Y, label="actual")
plt.show()


forecast_error=test["snp"]-test_pred_ols

RMSE = np.sqrt(np.mean(forecast_error**2))
print("The Root Mean Square of Forecast Error using OLS is "+str(RMSE))

mean_forecast_error=np.mean(forecast_error)
print("The Mean of Forecast Error using is "+str(mean_forecast_error))

def standard_error(forecast_error,num_of_predictors):
    return np.sqrt(np.sum(forecast_error**2)/(len(forecast_error)-num_of_predictors-1))

se=standard_error(forecast_error,1)
print("The standard error using OLS is "+str(se))