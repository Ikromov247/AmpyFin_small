import pandas as pd
import datetime

def get_data(ticker:str, start_date:datetime.date, end_date:datetime.date)->pd.DataFrame:
    """
    Get historical stock price data. 
    Adjust for your own data source.
    Implement error handling here
    Returns:
    pd.DataFrame with timestamp as index and prices as values. 
    expected column names: open, high, low, close, volume, ticker. Case sensitive. Validate before returning
    """
    pass