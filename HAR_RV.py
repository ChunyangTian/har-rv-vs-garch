#pip install yfinance pandas matplotlib numpy arch

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from arch import arch_model

#=== Part 1 ===
#dowload SPY data from Yahoo Finance and clean it up
df = yf.download('SPY', start='2004-01-01', end='2026-05-01')
df.columns = df.columns.get_level_values(0)
df = df.loc[:, ['Close']]
df['Return'] = np.log(df['Close'] / df["Close"].shift(1))
#RV = relized volatility, annualized by multiplying by sqrt(252)
df['RV'] = abs(df['Return']) * np.sqrt(252)
df.dropna(inplace = True)

#=== Part 2 ===
#RV day
df['RV_d'] = df['RV'].shift(1)
#RV week
df['RV_w'] = df['RV'].rolling(window=5).mean().shift(1)
#RV month
df['RV_m'] = df['RV'].rolling(window=21).mean().shift(1)
df.dropna(inplace = True)
print(df)

#=== Part 3 ===
#HAR_RV regression
X = df[['RV_d', 'RV_w', 'RV_m']]
y = df['RV']
model = LinearRegression()
model.fit(X, y)
print(model.fit)
print("Coefficients(RV-d):", model.coef_[0])
print("Coefficients(RV-w):", model.coef_[1])
print("Coefficients(RV-m):", model.coef_[2])
print("Intercept:", model.intercept_)
print("R^2 Score:", model.score(X, y))
df['RV_pred'] = model.predict(X)

#=== Part 4 ===
#Arch default using percentage returns
returns = df['Return'] * 100
#test GARCH(1,1) model
garch_model = arch_model(returns, mean='Zero', vol='Garch', p=1, q=1)
garch_fit = garch_model.fit(disp='off')
print(garch_fit.summary())
df['GARCH_vol'] = garch_fit.conditional_volatility / 100 * np.sqrt(252)

#=== Part 5 ===
#In-sample comparison
mse_har = ((df['RV_pred'] - df['RV']) ** 2).mean()
mse_garch = ((df['GARCH_vol'] - df['RV']) ** 2).mean()
#print the MSE comparison
print("In-sample MSE comparison:")
print("HAR-RV MSE:", mse_har)
print("GARCH MSE:", mse_garch)

plt.figure(figsize=(16, 6))
plt.plot(df.index, df['RV'], label='Actual RV', alpha=0.4, color='gray')
plt.plot(df.index, df['RV_pred'], label='HAR-RV forecast', alpha=0.7, color='blue')
plt.plot(df.index, df['GARCH_vol'], label='GARCH forecast', alpha=0.7, color='red')
plt.title('In-sample Volatility Forecast Comparison')
plt.xlabel('Date')
plt.ylabel('Annualized Volatility')
plt.legend()
plt.grid(True, alpha=0.2)

#=== Part 6 ===
# Rolling out-of-sample comparison
# Strategy: sliding window of 1000 days, refit every 21 days

train_size = 1000
refit_every = 21

predictions_har = []
predictions_garch = []
actuals = []
prediction_dates = []

for t in range(train_size, len(df)):
    if (t - train_size) % 500 == 0:
        print(f"Processed day {t}/{len(df)}")

    df_t = df.iloc[t-1000:t]
    
    # HAR-RV: fit everyday
    X_t = df_t[['RV_d', 'RV_w', 'RV_m']]
    y_t = df_t['RV']
    model_t = LinearRegression()
    model_t.fit(X_t, y_t)
    features_t = df.iloc[t:t+1][['RV_d', 'RV_w', 'RV_m']] 
    prediction_t = model_t.predict(features_t)[0]
    
    # GARCH: refit everyday
    returns_t = df_t['Return'] * 100
    garch_model_t = arch_model(returns_t, mean='Zero', vol='Garch', p=1, q=1)
    garch_fit_t = garch_model_t.fit(disp='off')
    forecast = garch_fit_t.forecast(horizon=1)
    pred_garch = np.sqrt(forecast.variance.values[0, 0]) / 100 * np.sqrt(252)
    
    predictions_har.append(prediction_t)
    predictions_garch.append(pred_garch)
    actuals.append(df.iloc[t]['RV'])
    prediction_dates.append(df.index[t])

#Transfer to the form can be calculated
predictions_har = np.array(predictions_har)
predictions_garch = np.array(predictions_garch)
actuals = np.array(actuals)

mse_har_oos = np.mean((predictions_har - actuals) ** 2)
mse_garch_oos = np.mean((predictions_garch - actuals) ** 2)

print("\n==== Out-of-sample MSE comparison ====")
print("HAR-RV:mse_har_oos:", mse_har_oos)
print("GARCH:mse_garch_oos:", mse_garch_oos)

print("\n==== In-sample vs Out-of-sample ====")
print("HAR-RV  in-sample:",mse_har," out-of-sample:", mse_har_oos)
print("GARCH   in-sample:",mse_garch," out-of-sample:", mse_garch_oos)


#plot the out-of-sample forecasts vs actuals
plt.figure(figsize=(16, 6))
plt.plot(prediction_dates, actuals, label='Actual RV', alpha=0.4, color='gray')
plt.plot(prediction_dates, predictions_har, label='HAR-RV forecast', alpha=0.7, color='blue')
plt.plot(prediction_dates, predictions_garch, label='GARCH forecast', alpha=0.7, color='red')
plt.title('Out-of-sample Volatility Forecast Comparison')
plt.xlabel('Date')
plt.ylabel('Annualized Volatility')
plt.legend()
plt.grid(True, alpha=0.2)
plt.show()