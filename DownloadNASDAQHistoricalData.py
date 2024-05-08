import yfinance as yf
data = yf.download("SPY AAPL", period="1mo")

from pandas_datareader import data as pdr

yf.pdr_override() # <== that's all it takes :-)

# download dataframe
data = pdr.get_data_yahoo("SPY", start="2019-01-01", end="2024-05-4")
data.to_csv('test_stock_data_AAPL.csv', index=False)

print(data.head())