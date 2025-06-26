import yfinance as yf

def get_yfinance_ohlc(ticker, start_date, interval, end_date=None):
    company=yf.Ticker(ticker)
    ohlc_data=company.history(interval=interval,start=start_date)
    return ohlc_data