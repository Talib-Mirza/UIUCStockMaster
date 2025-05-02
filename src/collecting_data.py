
#JPM JNJ PG XOM
import yfinance as yf
import pandas as pd
import ta  # Technical Analysis library
from fredapi import Fred  # Federal Reserve Economic Data
import numpy as np
import requests

def fetch_OHLCV(ticker, start_date=None, end_date=None, interval='1d'):
    """
    Fetches OHCLV stock data and drops missing values
    Returns a DataFrame with datetime index and OHLCV columns
    """
    # Fetch the data
    ticker = yf.Ticker(ticker)
    data = ticker.history(start=start_date, end=end_date, interval=interval)
    ticker.financials.T.iloc[::-1].to_csv('data/raw/KO/financial.csv')
    print("Financials columns:", len(ticker.quarterly_financials.T.columns))
    ticker.balance_sheet.T.iloc[::-1].to_csv('data/raw/KO/balance_sheet.csv')
    print("Balance Sheet columns:", len(ticker.quarterly_balance_sheet.T.columns))
    ticker.cashflow.T.iloc[::-1].to_csv('data/raw/KO/cashflow.csv')
    print("Cashflow columns:", len(ticker.quarterly_cashflow.T.columns))
    ticker.income_stmt.T.iloc[::-1].to_csv('data/raw/KO/income_stmt.csv')
    print("Income Statement columns:", len(ticker.income_stmt.T.columns))
    
    return data.drop(['Dividends', 'Stock Splits'], axis=1)

#fetch_data(ticker, start_time, end_time, interval='1d')


# The code below requires an API Key which can be generated from the url below. I believe the api allows
# like 1000 requests per month
def fetch_news_articles(ticker,from_date=None, to_date=None, language='en', page_size=100):

    url = 'https://newsapi.org/v2/everything'
    params = {
        'q': ticker,
        'apiKey': "INSERT YOUR API KEY HERE!!!",
        'language': language,
        'pageSize': page_size,
        'from': from_date,
        'to': to_date,
        'sortBy': 'relevancy'
    }

    response = requests.get(url, params=params)

    data = response.json()

    if data.get('status') == 'ok':
        articles = data.get('articles', [])
        news_df = pd.DataFrame(articles)[['publishedAt', 'title', 'description', 'url', 'content']]
        news_df['publishedAt'] = pd.to_datetime(news_df['publishedAt'])
        return news_df
    else:
        raise Exception(f"Error fetching news: {data.get('message', 'Unknown error')}")

# Only works up to a month back, currently only returns the title and description of articles in a pd
# dataframe. Will probably be updated soon

#news_df = fetch_news_articles("GOOG", "2025-01-06", "2025-02-06")
#print(news_df)

def calculate_technical_indicators(df):
    """
    Calculates various technical indicators using the TA library
    Input: DataFrame with OHLCV data
    Returns: DataFrame with technical indicators added
    """
    # Trend Indicators
    df['sma_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['ema_20'] = ta.trend.ema_indicator(df['Close'], window=20)
    df['macd'] = ta.trend.macd_diff(df['Close'])
    
    # Momentum Indicators
    df['rsi'] = ta.momentum.rsi(df['Close'])
    df['stoch'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'])
    
    # Volatility Indicators
    df['bollinger_high'] = ta.volatility.bollinger_hband(df['Close'])
    df['bollinger_low'] = ta.volatility.bollinger_lband(df['Close'])
    df['atr'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
    
    # Volume Indicators
    df['obv'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
    
    return df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1)  # Remove any rows with NaN values

def fetch_economic_data(start_date=None, end_date=None):
    """
    Fetches relevant economic indicators from FRED
    Requires FRED API key set as environment variable FRED_API_KEY
    """
    fred = Fred(api_key='67002050e0d50c806abd1618856e16a3')  # Replace with your FRED API key
    
    indicators = {
        'M1': 'M1SL',
        'M2': 'M2SL',
        'Interest_Rate': 'DGS10',
        'Inflation': 'CPIAUCNS',
        'Unemployment_Rate': 'UNRATE', # Unemployment Rate
        'Manufacturing_Production': 'INDPRO',
        'Mortgage_Rate': 'MORTGAGE30US',
        'GDP': 'GDP',              # Gross Domestic Product
        'Consumer Price Index': 'CPIAUCSL',    # Consumer Price Index
        'DFF': 'DFF',              # Federal Funds Rate      
    }
    
    economic_data = pd.DataFrame()
    for name, series_id in indicators.items():
        series = fred.get_series(series_id, start_date, end_date)
        economic_data[name] = series
    
    return economic_data  # Forward fill missing values


# Example usage:
if __name__ == "__main__":

    ticker = "KO"
    # Fetch base OHLCV data
    ohlcv_data = fetch_OHLCV(ticker, start_date="2018-11-01", end_date="2025-03-30")
    
    # Add technical indicators
    technical_data = calculate_technical_indicators(ohlcv_data.copy())
    
    # Fetch economic data
    economic_data = fetch_economic_data("2019-01-01", "2025-03-30")
    print(technical_data.columns)

    # Combine all data
    #combined_data = pd.concat([technical_data, economic_data, sentiment_data], axis=1)
    
    # Save to CSV
    ohlcv_data.to_csv('data/raw/KO/ohlcv_data.csv', index=True)
    technical_data.to_csv('data/raw/KO/technical_data.csv', index=True)
    economic_data.to_csv('data/raw/economic_data.csv', index=True)

    
