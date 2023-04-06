import pandas as pd
import yfinance as yf

symbol = "RELIANCE.NS"
start = "2022-04-06"
end = "2023-04-06"

data = yf.download(symbol, start=start, end=end)

data.to_csv("Reliance.csv")
