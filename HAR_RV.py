#pip install yfinance pandas matplotlib numpy arch

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from arch import arch_model
from scipy import stats


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
k_hat = np.mean(np.abs(garch_fit.std_resid))
c_gauss = np.sqrt(2 / np.pi)          # 正态理论值,用来对照
print("k_hat (estimated E|z|):", k_hat, " | Gaussian theory:", c_gauss)

df['GARCH_vol'] = garch_fit.conditional_volatility / 100 * np.sqrt(252) * k_hat

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

predictions_har = []
predictions_garch = []
predictions_garch_g = []
predictions_rw = []
actuals = []
prediction_dates = []

for t in range(train_size, len(df)):
    if (t - train_size) % 500 == 0:
        print(f"Processed day {t}/{len(df)}")

    df_t = df.iloc[t-train_size:t]
    
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
    sigma_next = np.sqrt(forecast.variance.values[0, 0]) / 100

    k_hat_t = np.mean(np.abs(garch_fit_t.std_resid))
    pred_garch = sigma_next * k_hat_t * np.sqrt(252)
    pred_garch_g = sigma_next * np.sqrt(2 / np.pi) * np.sqrt(252)
    
    predictions_har.append(prediction_t)
    predictions_garch.append(pred_garch)
    predictions_garch_g.append(pred_garch_g)
    predictions_rw.append(df.iloc[t]['RV_d'])
    actuals.append(df.iloc[t]['RV'])
    prediction_dates.append(df.index[t])

#Transfer to the form can be calculated
predictions_har = np.array(predictions_har)
predictions_garch = np.array(predictions_garch)
predictions_garch_g = np.array(predictions_garch_g)
predictions_rw = np.array(predictions_rw)
actuals = np.array(actuals)

e_har   = predictions_har     - actuals
e_garch = predictions_garch   - actuals
e_garch_g = predictions_garch_g - actuals
e_rw    = predictions_rw      - actuals

print("\n==== Out-of-sample MSE ====")
print("HAR-RV               :", np.mean(e_har**2))
print("GARCH (k_hat scale)  :", np.mean(e_garch**2))
print("GARCH (Gaussian sqrt(2/pi)):", np.mean(e_garch_g**2))
print("Random walk baseline :", np.mean(e_rw**2))

def diebold_mariano(e1, e2, lag=5):
    """H0: Two model squared-error loss identical。DM<0 mean the first model has less error."""
    d = e1**2 - e2**2
    n = len(d)
    dbar = d.mean()
    dc = d - dbar
    var = np.mean(dc**2)
    for h in range(1, lag + 1):                      # Newey-West
        var += 2 * (1 - h / (lag + 1)) * np.mean(dc[h:] * dc[:-h])
    dm = dbar / np.sqrt(var / n)
    p = 2 * (1 - stats.norm.cdf(abs(dm)))
    return dm, p

print("\n==== Diebold-Mariano test ====")
for name, (a, b) in [
    ("HAR vs GARCH", (e_har, e_garch)),
    ("HAR vs RW",    (e_har, e_rw)),
    ("GARCH vs RW",  (e_garch, e_rw)),
]:
    dm, p = diebold_mariano(a, b)
    verdict = "significant" if p < 0.05 else "NOT significant"
    print(f"{name:14s}  DM = {dm:+.3f}  p = {p:.4f}  ({verdict})")
print("(DM < 0 means the first model has lower squared error.)")

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
