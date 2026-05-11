# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 11/05/26



### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
data = pd.read_csv('NFLX.csv')
print(data.columns)
print(data.head())
if 'Date' in data.columns:
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
X = data['Open']

# Convert to stationary
X = X.diff().dropna()
plt.figure(figsize=(12,6))
plt.plot(X)
plt.title('Open Price (Differenced)')
plt.show()
```
<img width="1247" height="657" alt="image" src="https://github.com/user-attachments/assets/b53d5e42-4c05-4030-a731-37eabdefcd5a" />

``` 
plt.subplot(2, 1, 1)
plot_acf(X, lags=50, ax=plt.gca())
plt.title('ACF')

plt.subplot(2, 1, 2)
plot_pacf(X, lags=50, ax=plt.gca())
plt.title('PACF')

plt.tight_layout()
plt.show()
```
<img width="878" height="581" alt="image" src="https://github.com/user-attachments/assets/4cc29fb6-53e4-4586-ade7-4391fbdd0615" />

```
arma11_model = ARIMA(X, order=(1, 0, 1)).fit()

phi1 = arma11_model.params['ar.L1']
theta1 = arma11_model.params['ma.L1']
ar1 = np.array([1, -phi1])
ma1 = np.array([1, theta1])

ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=1000)

plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1)')
plt.xlim([0, 500])
plt.show()
plot_acf(ARMA_1)
plt.show()

plot_pacf(ARMA_1)
plt.show()
arma22_model = ARIMA(X, order=(2, 0, 2)).fit()

phi1 = arma22_model.params['ar.L1']
phi2 = arma22_model.params['ar.L2']
theta1 = arma22_model.params['ma.L1']
theta2 = arma22_model.params['ma.L2']
ar2 = np.array([1, -phi1, -phi2])
ma2 = np.array([1, theta1, theta2])

ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=5000)

plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2)')
plt.xlim([0, 500])
plt.show()
plot_acf(ARMA_2)
plt.show()

plot_pacf(ARMA_2)
plt.show()

```
OUTPUT:
SIMULATED ARMA(1,1) PROCESS:

<img width="702" height="527" alt="image" src="https://github.com/user-attachments/assets/72b95c48-75a0-4530-a531-109ec98a469f" />


Partial Autocorrelation
```
<img width="726" height="537" alt="image" src="https://github.com/user-attachments/assets/4d9e1095-5ef1-4251-bc62-2bddb2344b58" />
```

Autocorrelation
```
<img width="726" height="533" alt="image" src="https://github.com/user-attachments/assets/fd6f6d8d-a714-4a84-90f4-9a23d2418ebf" />
```

SIMULATED ARMA(2,2) PROCESS:
```
<img width="712" height="538" alt="image" src="https://github.com/user-attachments/assets/f25ef72e-db11-47f2-8d23-d60f7c1a6308" />
```

Partial Autocorrelation
```
<img width="786" height="541" alt="image" src="https://github.com/user-attachments/assets/e94124c6-e8ae-4630-b6d5-749709bf248e" />
```

Autocorrelation
```
<img width="773" height="523" alt="image" src="https://github.com/user-attachments/assets/b1616f14-bc6c-455a-a9ad-c39fba110fb5" />
```

RESULT:
Thus, a python program is created to fir ARMA Model successfully.
