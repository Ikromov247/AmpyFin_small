import pandas as pd
import numpy as np
from typing import List, Dict, Any

# Import the indicator functions
from talib_rl import *

def extract_features(ticker: str, data: pd.DataFrame) -> Dict[str, Any]:
    """
    Extract features from historical data for RL agent consumption.
    
    This function calculates various technical indicators and returns them
    in a format suitable for an RL agent to make trading decisions.
    
    Parameters:
    -----------
    ticker : str
        The stock ticker symbol
    data : pd.DataFrame
        Historical price data with OHLCV columns
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing all calculated features
    """
    features = {}
    
    # Basic price information
    features['ticker'] = ticker
    features['current_price'] = data['close'].iloc[-1]
    features['daily_return'] = data['close'].iloc[-1] / data['close'].iloc[-2] - 1 if len(data) > 1 else 0
    
    # Trend indicators
    features['sma_20'] = SMA_indicator(ticker, data.copy())
    features['ema_20'] = EMA_indicator(ticker, data.copy())
    
    # Price relative to moving averages
    features['price_to_sma'] = features['current_price'] / features['sma_20'] if features['sma_20'] != 0 else 1
    features['price_to_ema'] = features['current_price'] / features['ema_20'] if features['ema_20'] != 0 else 1
    
    # Momentum indicators
    features['rsi'] = RSI_indicator(ticker, data.copy())
    
    macd_values = MACD_indicator(ticker, data.copy())
    features['macd'] = macd_values['macd']
    features['macd_signal'] = macd_values['macdsignal']
    features['macd_hist'] = macd_values['macdhist']
    
    # Volatility indicators
    features['atr'] = ATR_indicator(ticker, data.copy())
    
    bbands = BBANDS_indicator(ticker, data.copy())
    features['bbands_upper'] = bbands['upper']
    features['bbands_middle'] = bbands['middle']
    features['bbands_lower'] = bbands['lower']
    features['percent_b'] = bbands['percent_b']
    
    # Volume indicators
    features['obv'] = OBV_indicator(ticker, data.copy())
    
    # Pattern recognition (select a few important ones)
    features['cdl_engulfing'] = CDLENGULFING_indicator(ticker, data.copy())
    features['cdl_hammer'] = CDLHAMMER_indicator(ticker, data.copy())
    features['cdl_doji'] = CDLDOJI_indicator(ticker, data.copy())
    
    # Statistical indicators
    features['linearreg_slope'] = LINEARREG_SLOPE_indicator(ticker, data.copy())
    
    return features

def batch_extract_features(tickers: List[str], data_dict: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
    """
    Extract features for multiple tickers.
    
    Parameters:
    -----------
    tickers : List[str]
        List of stock ticker symbols
    data_dict : Dict[str, pd.DataFrame]
        Dictionary mapping tickers to their historical data
        
    Returns:
    --------
    List[Dict[str, Any]]
        List of feature dictionaries for each ticker
    """
    features_list = []
    
    for ticker in tickers:
        if ticker in data_dict:
            features = extract_features(ticker, data_dict[ticker])
            features_list.append(features)
    
    return features_list

def create_rl_state(features: Dict[str, Any]) -> np.ndarray:
    """
    Convert features dictionary to a state vector for RL agent.
    
    Parameters:
    -----------
    features : Dict[str, Any]
        Dictionary of extracted features
        
    Returns:
    --------
    np.ndarray
        State vector for RL agent
    """
    # Select and order the features for the state vector
    # You may want to normalize these values for your RL agent
    state = np.array([
        features['price_to_sma'],
        features['price_to_ema'],
        features['rsi'] / 100.0,  # Normalize RSI to [0,1]
        features['macd'],
        features['macd_hist'],
        features['percent_b'],
        features['linearreg_slope'],
        features['daily_return'],
        features['cdl_engulfing'] / 100.0 if abs(features['cdl_engulfing']) == 100 else 0,  # Normalize pattern signals
        features['cdl_hammer'] / 100.0 if abs(features['cdl_hammer']) == 100 else 0,
        features['cdl_doji'] / 100.0 if abs(features['cdl_doji']) == 100 else 0
    ])
    
    return state

def get_rl_states_for_portfolio(tickers: List[str], data_dict: Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
    """
    Get RL state vectors for all tickers in a portfolio.
    
    Parameters:
    -----------
    tickers : List[str]
        List of stock ticker symbols
    data_dict : Dict[str, pd.DataFrame]
        Dictionary mapping tickers to their historical data
    
    Returns:
    --------
    Dict[str, np.ndarray]
        Dictionary mapping tickers to their state vectors
    """
    states = {}
    features_list = batch_extract_features(tickers, data_dict)
    
    for i, ticker in enumerate(tickers):
        if i < len(features_list):
            states[ticker] = create_rl_state(features_list[i])
    
    return states

# Example usage
if __name__ == "__main__":
    # Example data for demonstration
    sample_data = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=100),
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 105,
        'low': np.random.randn(100).cumsum() + 95,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100)
    })
    
    # Extract features for a single ticker
    features = extract_features('AAPL', sample_data)
    print(f"Extracted {len(features)} features for AAPL")
    
    # Create RL state
    state = create_rl_state(features)
    print(f"RL state shape: {state.shape}")
    
    # Example with multiple tickers
    tickers = ['AAPL', 'MSFT', 'GOOG']
    data_dict = {ticker: sample_data.copy() for ticker in tickers}
    
    states = get_rl_states_for_portfolio(tickers, data_dict)
    print(f"Created states for {len(states)} tickers")