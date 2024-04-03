import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
from matplotlib import pyplot as plt

# Download Bitcoin price data
btc = yf.download('BTC-USD', start='2009-01-03',
                  end='2024-03-15', interval='1mo')
btc = btc.dropna()

# # Create time column
# btc['Time'] = (btc.index - pd.Timestamp("2009-01-03")) / np.timedelta64(1, 'M')
# Create time column
btc['Time'] = (btc.index - pd.Timestamp("2009-01-03")) / np.timedelta64(1, 'D')
btc['Time'] = btc['Time'] / 30.44

# Convert to log base 10
btc['Log Time'] = np.log10(btc['Time'])
btc['Log Close'] = np.log10(btc['Close'])

# Run regression
X = sm.add_constant(btc['Log Time'])
model = sm.OLS(btc['Log Close'], X)
results = model.fit()
btc['Model'] = results.predict(X)

# Print the regression formula
print(
    f"Log(Close Price) = {results.params[0]} + {results.params[1]} * Log(Days Since Earliest)")

# The rest of your code...
# Compute standard deviation
std_dev = np.std(btc['Log Close'] - btc['Model'])

# Create up and down columns
btc['Up'] = btc['Model'] + std_dev
btc['Down'] = btc['Model'] - std_dev

# Convert back to non-log data
btc['Model Price'] = 10 ** btc['Model']
btc['Up Price'] = 10 ** btc['Up']
btc['Down Price'] = 10 ** btc['Down']

# Compute correlation coefficient
corr_coef = np.corrcoef(btc['Log Close'], btc['Model'])[0, 1]

# Create a figure
plt.figure(figsize=(24, 6))

# Create first subplot for log data
plt.subplot(1, 2, 1)
plt.plot(btc['Log Time'].values, btc['Log Close'].values, label='Log Close')
plt.plot(btc['Log Time'].values, btc['Model'].values, label='Model')
plt.plot(btc['Log Time'].values, btc['Up'].values, label='Up')
plt.plot(btc['Log Time'].values, btc['Down'].values, label='Down')
plt.legend()
plt.title('Log Data')

# Create second subplot for non-log data
plt.subplot(1, 2, 2)
plt.plot(btc['Time'].values, btc['Close'].values, label='Close')
plt.plot(btc['Time'].values, btc['Model Price'].values, label='Model Price')
plt.plot(btc['Time'].values, btc['Up Price'].values, label='Up Price')
plt.plot(btc['Time'].values, btc['Down Price'].values, label='Down Price')
plt.legend()
plt.title('Non-Log Data')

# Show the figure with both subplots
plt.show()

print(f"Correlation coefficient: {corr_coef}")
