import ccxt 
import pandas as pd 


binance = ccxt.binance()
btc_ohlcv = binance.fetch_ohlcv("BTC/USDT")

df = pd.DataFrame(btc_ohlcv, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
df.set_index('datetime', inplace=True)
print(df)