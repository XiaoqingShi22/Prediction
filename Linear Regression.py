import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.figsize'] = [15, 5]


#Read the data and visualize it to see how the variables are related
adv = pd.read_csv('../datas/Advertising.csv')
adv.head()

sns.pairplot(adv,vars={'sales','TV','radio','newspaper'})

#Extract values from the DataFrame into numpy arrays x and y
x = adv.iloc[:,1:2].values
y = adv.iloc[:,4].values

#Define the Model to be a linear regression model. 
model = LinearRegression(fit_intercept=True)
reg = model.fit(x,y)
print('Regression coefficients are',reg.coef_)
print('The Intercept beta_0 is',reg.intercept_)

yfit = reg.predict(x)
mse = mean_squared_error(y,yfit)
R2 = r2_score(y,yfit)
print('Mean square error = ',mse)
print('R squared = ',R2)

#plot the scatter plot and linear regression fit for Sales vs radio and Sales vs newspaper
fig,ax = plt.subplots(1,3,sharey='row')
xax = np.linspace(0,np.max(x),100).reshape(-1,1)
yfit2 = reg.predict(xax)
ax[0].scatter(x,y,color='green')
ax[0].plot(xax,yfit2,color='red')

x1 = adv.iloc[:,2:3].values
y1= adv.iloc[:,4].values
model1 = LinearRegression(fit_intercept=True)
reg1 = model1.fit(x1,y1)
print('Regression coefficients for radio are',reg1.coef_)
print('The Intercept beta_0 for radio is',reg1.intercept_)
x2 = adv.iloc[:,3:4].values
y2= adv.iloc[:,4].values
model2 = LinearRegression(fit_intercept=True)
reg2 = model2.fit(x2,y2)
print('Regression coefficients for newspaper are',reg2.coef_)
print('The Intercept beta_0 for newspaper is',reg2.intercept_)
xax1 = np.linspace(0,np.max(x1),100).reshape(-1,1)
yfit1 = reg1.predict(xax1)
ax[1].scatter(x1,y1,color='green')
ax[1].plot(xax1,yfit1,color='red')

xax2 = np.linspace(0,np.max(x2),100).reshape(-1,1)
yfit2 = reg2.predict(xax2)
ax[2].scatter(x2,y2,color='green')
ax[2].plot(xax2,yfit2,color='red')

#Compute the standard errors, t-statistic and p-value
import statsmodels.api as sm
adv['beta_0'] = 1
reg0 = sm.OLS(endog=adv['sales'], exog=adv[['beta_0', 'TV']])
type(reg0)
results = reg0.fit()
type(results)
print(results.summary())