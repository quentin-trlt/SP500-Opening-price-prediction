import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostRegressor
import requests
import time
import pytz
from datetime import datetime, timedelta

"""**Modèle de prédiction - Paramètres : ouverture, cloture, variation plus haut-plus bas, volume, RSI, VIX, Dollar Index, taux d'intérêt sur bon du Trésor**"""

def calculate_rsi(data, window):
    delta = data.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def train_model(x,y):
    global global_reg
    split_index = int(len(x) * 0.8)
    x_train = x[:split_index]
    y_train = y[:split_index]
    x_test = x[split_index:]
    y_test = y[split_index:]

    # Entraîner le modèle de régression
    global_reg = LinearRegression()
    global_reg.fit(x_train, y_train)
    y_pred = global_reg.predict(x_test)
    #return(x_test)

def predict(date_str, data):
    if global_reg is None:
        raise ValueError("Le modèle n'a pas été entraîné. Appelez train_model() d'abord.")

    date = pd.to_datetime(date_str).date()

    if data['Date'].dtype != 'datetime64[ns]':
        data['Date'] = pd.to_datetime(data['Date']).dt.date

    # Vérifier si la date est dans les données
    if date not in data['Date'].values:
        raise ValueError(f"La date {date_str} n'est pas présente dans les données.")

    index = data[data['Date'] == date].index[0]

    last_open = data['Open'].iloc[index]
    last_close = data['Close'].iloc[index]
    last_volume = data['Volume'].iloc[index]
    last_high_low_percent = data['High_Low_Percent'].iloc[index]
    last_rsi = data['RSI'].iloc[index]
    last_vix = data['VIX'].iloc[index]
    last_usd = data['Dollar Index'].iloc[index]
    last_tbill = data['T-Bill'].iloc[index]

    next_open_pred = global_reg.predict(np.array([[last_open, last_close, last_volume, last_high_low_percent, last_vix, last_usd, last_tbill, last_rsi]]))

    return next_open_pred[0]

def authenticate():
    response = requests.post(f'https://api.xtb.com/login', json={
        'user': 'quentintarlet525@gmail.com',
        'password': 'Qtarlet69*'
    })
    data = response.json()
    return data['streamSessionId']

def place_order(session_id, symbol, direction, price, volume):
    response = requests.post(f'https://api.xtb.com/trade', json={
        'sessionId': session_id,
        'symbol': symbol,
        'direction': direction,
        'price': price,
        'volume': volume,
        'type': 'MARKET',
        'stopLoss': None,
        'takeProfit': None
    })
    return response.json()

def get_live_price(session_id):
    response = requests.get(f'https://api.xtb.com/price', params={
        'sessionId': session_id,
        'symbol': 'S&P500'
    })
    return response.json()['price']

def close_position(session_id, position_id):
    response = requests.post(f'https://api.xtb.com/close', json={
        'sessionId': session_id,
        'positionId': position_id
    })
    return response.json()

def main():

    global_reg = None
    startdate = '2006-01-01'
    enddate = pd.Timestamp.now().strftime('%Y-%m-%d')

    sp500 = yf.Ticker("^GSPC")
    sp500_historical = sp500.history(start=startdate, end=enddate).reset_index()

    vix = yf.Ticker("^VIX")
    vix_historical = vix.history(start=startdate, end=enddate)['Close'].reset_index()
    vix_historical.drop(columns='Date', inplace=True)
    vix_historical.rename(columns={'Close': 'VIX'}, inplace=True)

    usd = yf.Ticker("DX-Y.NYB")
    usd_historical = usd.history(start=startdate, end=enddate)['Close'].reset_index()
    usd_historical.drop(columns='Date', inplace=True)
    usd_historical.rename(columns={'Close': 'Dollar Index'}, inplace=True)

    tbill = yf.Ticker("^IRX")
    tbill_historical = tbill.history(start=startdate, end=enddate)['Close'].reset_index()
    tbill_historical.rename(columns={'Close': 'T-Bill'}, inplace=True)

    sp500_historical = sp500_historical.join(vix_historical, how='left')
    sp500_historical = sp500_historical.join(usd_historical, how='left')
    sp500_historical['Date'] = pd.to_datetime(sp500_historical['Date']).dt.date
    tbill_historical['Date'] = pd.to_datetime(tbill_historical['Date']).dt.date

    sp500_historical = sp500_historical.merge(tbill_historical, on='Date', how='left')
    sp500_historical['T-Bill'] = sp500_historical['T-Bill'].interpolate(method='linear', limit_direction='both')

    sp500_historical['High_Low_Percent'] = (sp500_historical['High'] - sp500_historical['Low']) / sp500_historical['Low'] * 100
    sp500_historical.drop(columns=['High', 'Low'], inplace=True)

    sp500_historical['RSI'] = calculate_rsi(sp500_historical['Close'], 14)
    sp500_historical['RSI'].fillna(sp500_historical['RSI'].mean(), inplace=True)
    x = sp500_historical[['Open', 'Close', 'Volume', 'High_Low_Percent', 'VIX', 'Dollar Index','T-Bill', 'RSI']].iloc[:-1].values
    y = sp500_historical['Open'].iloc[1:].values
    print(sp500_historical.tail())
    train_model(x,y)
    
    now = datetime.now(pytz.timezone('Europe/Paris'))

    if now.time() < datetime.strptime("15:30", "%H:%M").time():
        target_date = now - timedelta(days=1)
    else:
        target_date = now

    # Si la date est un samedi ou un dimanche, ajuster à la date du vendredi précédent
    if target_date.weekday() in [5, 6]:  
        days_to_last_friday = target_date.weekday() - 4  
        target_date = target_date - timedelta(days=days_to_last_friday)


    date = target_date.strftime("%Y-%m-%d")

    next_open_pred = predict(date, sp500_historical)
    print("Prédiction d'ouverture :", next_open_pred)
    print("Dernière cloture :", sp500_historical.loc[sp500_historical['Date'] == pd.to_datetime(date).date(), 'Close'].values[-1])

if __name__ == "__main__":
    main()