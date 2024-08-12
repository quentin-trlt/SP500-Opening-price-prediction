S&P 500 prediction - LinearRegression model :

- Packages needed : yfinance, numpy, pandas, sklearn, datetime
- Variables used : Opening price, closing price, daily variation, volume, VIX, RSI, Dollar Index, T-Bills.

Disclaimer  : since the model is only based on a linear regression, it's not the most accurate model that could exists, and a version based on more complex models is yet to come.
Still, the model is really accurate in predicting the variation of the price between the closing and the next opening.
A atotmatized trading bot based on this model is to come, taking overnights positions based on the predicted variation.
