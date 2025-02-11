import talib as ta 
import numpy as np
import pandas as pd
import sys

sys.path.append('..')

from control import trade_asset_limit


def simulate_strategy(strategy, ticker, current_price, historical_data, account_cash, portfolio_qty, total_portfolio_value):
   max_investment = total_portfolio_value * trade_asset_limit
   action = strategy(ticker, historical_data)
   
   if action == 'Buy':
      return 'buy', min(int(max_investment // current_price), int(account_cash // current_price))
   elif action == 'Sell' and portfolio_qty > 0:
      return 'sell', min(portfolio_qty, max(1, int(portfolio_qty * 0.5))) # sell half of the portfolio
   else:
      return 'hold', 0

"""
Interfaces for using TA-Lib indicators. 

The functions are named as the indicator name + _indicator.
The functions take a ticker and a pandas dataframe as input.
The dataframe must have a 'Date' column and a 'Close' column.
The functions return a string 'Buy', 'Sell', or 'Hold'.
"""

# Overlap Studies
def BBANDS_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Bollinger Bands (BBANDS) indicator."""  
      
   upper, middle, lower = ta.BBANDS(data['Close'], timeperiod=20)  
   # sells when the current price is above the upper band
   # and buys when the current price is below the lower band
   # otherwise, hold
   if data['Close'].iloc[-1] > upper.iloc[-1]:  
      return 'Sell'  
   elif data['Close'].iloc[-1] < lower.iloc[-1]:  
      return 'Buy'  
   else:  
      return 'Hold'  

def DEMA_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Double Exponential Moving Average (DEMA) indicator."""  
      
   dema = ta.DEMA(data['Close'], timeperiod=30)  
   if data['Close'].iloc[-1] > dema.iloc[-1]:  
      return 'Buy'  
   elif data['Close'].iloc[-1] < dema.iloc[-1]:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def EMA_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Exponential Moving Average (EMA) indicator."""  
      
   ema = ta.EMA(data['Close'], timeperiod=30)  
   if data['Close'].iloc[-1] > ema.iloc[-1]:  
      return 'Buy'  
   elif data['Close'].iloc[-1] < ema.iloc[-1]:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def HT_TRENDLINE_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Hilbert Transform - Instantaneous Trendline (HT_TRENDLINE) indicator."""  
      
   ht_trendline = ta.HT_TRENDLINE(data['Close'])  
   if data['Close'].iloc[-1] > ht_trendline.iloc[-1]:  
      return 'Buy'  
   elif data['Close'].iloc[-1] < ht_trendline.iloc[-1]:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def KAMA_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Kaufman Adaptive Moving Average (KAMA) indicator."""  
      
   kama = ta.KAMA(data['Close'], timeperiod=30)  
   if data['Close'].iloc[-1] > kama.iloc[-1]:  
      return 'Buy'  
   elif data['Close'].iloc[-1] < kama.iloc[-1]:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def MA_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Moving average (MA) indicator."""  
      
   ma = ta.MA(data['Close'], timeperiod=30, matype=0)  
   if data['Close'].iloc[-1] > ma.iloc[-1]:  
      return 'Buy'  
   elif data['Close'].iloc[-1] < ma.iloc[-1]:  
      return 'Sell'  
   else:  
      return 'Hold'  

def MAMA_indicator(ticker:str, data:pd.DataFrame)->str:
    """
    MESA Adaptive Moving Average (MAMA) indicator.
    
    Parameters:
    - ticker (str): Stock ticker symbol.

    Returns:
    - str: 'Buy', 'Sell', or 'Hold'.
    """
   
    close_prices = data['Close'].values

    # Validate enough data
    """
    if len(close_prices) < 32:  # Minimum length required by MAMA
        raise ValueError("Not enough data to compute MAMA.")
    """
    # Calculate MAMA and FAMA
    try:
        mama, fama = ta.MAMA(close_prices, fastlimit=0.5, slowlimit=0.05)
    except Exception as e:
        raise RuntimeError(f"Error computing MAMA for {ticker}: {e}")

    # Current price and last computed MAMA
    current_price = close_prices[-1]
    current_mama = mama[-1]

    # Generate signal
    if current_price > current_mama:
        return "Buy"
    elif current_price < current_mama:
        return "Sell"
    else:
        return "Hold"
  
def MAVP_indicator(ticker:str, data:pd.DataFrame)->str:
    """
    Moving Average with Variable Period (MAVP) indicator.
    
    Parameters:
    - ticker (str): Stock ticker symbol.

    Returns:
    - str: 'Buy', 'Sell', or 'Hold'.
    """
     
    close_prices = data['Close'].values
    """
    # Validate enough data
    if len(close_prices) < 30:  # Ensure enough data for MAVP calculation
        raise ValueError("Not enough data to compute MAVP.")
    """
    # Define variable periods as a NumPy array
    variable_periods = np.full(len(close_prices), 30, dtype=np.float64)
    # Calculate MAVP
    try:
        mavp = ta.MAVP(close_prices, periods=variable_periods)
    except Exception as e:
        raise RuntimeError(f"Error computing MAVP for {ticker}: {e}")

    # Current price and last computed MAVP
    current_price = close_prices[-1]
    current_mavp = mavp[-1]

    # Generate signal
    if current_price > current_mavp:
        return "Buy"
    elif current_price < current_mavp:
        return "Sell"
    else:
        return "Hold"

  
def MIDPOINT_indicator(ticker:str, data:pd.DataFrame)->str:  
   """MidPoint over period (MIDPOINT) indicator."""  
      
   midpoint = ta.MIDPOINT(data['Close'], timeperiod=14)  
   if data['Close'].iloc[-1] > midpoint.iloc[-1]:  
      return 'Buy'  
   elif data['Close'].iloc[-1] < midpoint.iloc[-1]:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def MIDPRICE_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Midpoint Price over period (MIDPRICE) indicator."""  
      
   midprice = ta.MIDPRICE(data['High'], data['Low'], timeperiod=14)  
   if data['Close'].iloc[-1] > midprice.iloc[-1]:  
      return 'Buy'  
   elif data['Close'].iloc[-1] < midprice.iloc[-1]:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def SAR_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Parabolic SAR (SAR) indicator."""  
      
   sar = ta.SAR(data['High'], data['Low'], acceleration=0, maximum=0)  
   if data['Close'].iloc[-1] > sar.iloc[-1]:  
      return 'Buy'  
   elif data['Close'].iloc[-1] < sar.iloc[-1]:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def SAREXT_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Parabolic SAR - Extended (SAREXT) indicator."""  
      
   sarext = ta.SAREXT(data['High'], data['Low'], startvalue=0, offsetonreverse=0, accelerationinitlong=0, accelerationlong=0, accelerationmaxlong=0, accelerationinitshort=0, accelerationshort=0, accelerationmaxshort=0)  
   if data['Close'].iloc[-1] > sarext.iloc[-1]:  
      return 'Buy'  
   elif data['Close'].iloc[-1] < sarext.iloc[-1]:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def SMA_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Simple Moving Average (SMA) indicator."""  
      
   sma = ta.SMA(data['Close'], timeperiod=30)  
   if data['Close'].iloc[-1] > sma.iloc[-1]:  
      return 'Buy'  
   elif data['Close'].iloc[-1] < sma.iloc[-1]:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def T3_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Triple Exponential Moving Average (T3) indicator."""  
      
   t3 = ta.T3(data['Close'], timeperiod=30, vfactor=0)  
   if data['Close'].iloc[-1] > t3.iloc[-1]:  
      return 'Buy'  
   elif data['Close'].iloc[-1] < t3.iloc[-1]:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def TEMA_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Triple Exponential Moving Average (TEMA) indicator."""  
      
   tema = ta.TEMA(data['Close'], timeperiod=30)  
   if data['Close'].iloc[-1] > tema.iloc[-1]:  
      return 'Buy'  
   elif data['Close'].iloc[-1] < tema.iloc[-1]:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def TRIMA_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Triangular Moving Average (TRIMA) indicator."""  
      
   trima = ta.TRIMA(data['Close'], timeperiod=30)  
   if data['Close'].iloc[-1] > trima.iloc[-1]:  
      return 'Buy'  
   elif data['Close'].iloc[-1] < trima.iloc[-1]:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def WMA_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Weighted Moving Average (WMA) indicator."""  
      
   wma = ta.WMA(data['Close'], timeperiod=30)  
   if data['Close'].iloc[-1] > wma.iloc[-1]:  
      return 'Buy'  
   elif data['Close'].iloc[-1] < wma.iloc[-1]:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
# Momentum Indicators  
  
def ADX_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Average Directional Movement Index (ADX) indicator."""  
      
   adx = ta.ADX(data['High'], data['Low'], data['Close'], timeperiod=14)  
   if adx.iloc[-1] > 25:  
      return 'Buy'  
   elif adx.iloc[-1] < 20:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def ADXR_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Average Directional Movement Index Rating (ADXR) indicator."""  
      
   adxr = ta.ADXR(data['High'], data['Low'], data['Close'], timeperiod=14)  
   if adxr.iloc[-1] > 25:  
      return 'Buy'  
   elif adxr.iloc[-1] < 20:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def APO_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Absolute Price Oscillator (APO) indicator."""  
      
   apo = ta.APO(data['Close'], fastperiod=12, slowperiod=26, matype=0)  
   if apo.iloc[-1] > 0:  
      return 'Buy'  
   elif apo.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def AROON_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Aroon (AROON) indicator."""  
      
   aroon_down, aroon_up = ta.AROON(data['High'], data['Low'], timeperiod=14)  
   if aroon_up.iloc[-1] > 70:  
      return 'Buy'  
   elif aroon_down.iloc[-1] > 70:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def AROONOSC_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Aroon Oscillator (AROONOSC) indicator."""  
      
   aroonosc = ta.AROONOSC(data['High'], data['Low'], timeperiod=14)  
   if aroonosc.iloc[-1] > 0:  
      return 'Buy'  
   elif aroonosc.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def BOP_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Balance Of Power (BOP) indicator."""  
      
   bop = ta.BOP(data['Open'], data['High'], data['Low'], data['Close'])  
   if bop.iloc[-1] > 0:  
      return 'Buy'  
   elif bop.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def CCI_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Commodity Channel Index (CCI) indicator."""  
      
   cci = ta.CCI(data['High'], data['Low'], data['Close'], timeperiod=14)  
   if cci.iloc[-1] > 100:  
      return 'Buy'  
   elif cci.iloc[-1] < -100:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def CMO_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Chande Momentum Oscillator (CMO) indicator."""  
      
   cmo = ta.CMO(data['Close'], timeperiod=14)  
   if cmo.iloc[-1] > 50:  
      return 'Buy'  
   elif cmo.iloc[-1] < -50:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def DX_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Directional Movement Index (DX) indicator."""  
      
   dx = ta.DX(data['High'], data['Low'], data['Close'], timeperiod=14)  
   if dx.iloc[-1] > 25:  
      return 'Buy'  
   elif dx.iloc[-1] < 20:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def MACD_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Moving Average Convergence/Divergence (MACD) indicator."""  
      
   macd, macdsignal, macdhist = ta.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)  
   if macdhist.iloc[-1] > 0:  
      return 'Buy'  
   elif macdhist.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def MACDEXT_indicator(ticker:str, data:pd.DataFrame)->str:  
   """MACD with controllable MA type (MACDEXT) indicator."""  
      
   macd, macdsignal, macdhist = ta.MACDEXT(data['Close'], fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0)  
   if macdhist.iloc[-1] > 0:  
      return 'Buy'  
   elif macdhist.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def MACDFIX_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Moving Average Convergence/Divergence Fix 12/26 (MACDFIX) indicator."""  
      
   macd, macdsignal, macdhist = ta.MACDFIX(data['Close'], signalperiod=9)  
   if macdhist.iloc[-1] > 0:  
      return 'Buy'  
   elif macdhist.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def MFI_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Money Flow Index (MFI) indicator."""  
      
   mfi = ta.MFI(data['High'], data['Low'], data['Close'], data['Volume'], timeperiod=14)  
   if mfi.iloc[-1] > 80:  
      return 'Sell'  
   elif mfi.iloc[-1] < 20:  
      return 'Buy'  
   else:  
      return 'Hold'  
  
def MINUS_DI_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Minus Directional Indicator (MINUS_DI) indicator."""  
      
   minus_di = ta.MINUS_DI(data['High'], data['Low'], data['Close'], timeperiod=14)  
   if minus_di.iloc[-1] > 25:  
      return 'Sell'  
   elif minus_di.iloc[-1] < 20:  
      return 'Buy'  
   else:  
      return 'Hold'  
  
def MINUS_DM_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Minus Directional Movement (MINUS_DM) indicator."""  
      
   minus_dm = ta.MINUS_DM(data['High'], data['Low'], timeperiod=14)  
   if minus_dm.iloc[-1] > 0:  
      return 'Sell'  
   elif minus_dm.iloc[-1] < 0:  
      return 'Buy'  
   else:  
      return 'Hold'  
  
def MOM_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Momentum (MOM) indicator."""  
      
   mom = ta.MOM(data['Close'], timeperiod=10)  
   if mom.iloc[-1] > 0:  
      return 'Buy'  
   elif mom.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def PLUS_DI_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Plus Directional Indicator (PLUS_DI) indicator."""  
      
   plus_di = ta.PLUS_DI(data['High'], data['Low'], data['Close'], timeperiod=14)  
   if plus_di.iloc[-1] > 25:  
      return 'Buy'  
   elif plus_di.iloc[-1] < 20:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def PLUS_DM_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Plus Directional Movement (PLUS_DM) indicator."""  
      
   plus_dm = ta.PLUS_DM(data['High'], data['Low'], timeperiod=14)  
   if plus_dm.iloc[-1] > 0:  
      return 'Buy'  
   elif plus_dm.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def PPO_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Percentage Price Oscillator (PPO) indicator."""  
      
   ppo = ta.PPO(data['Close'], fastperiod=12, slowperiod=26, matype=0)  
   if ppo.iloc[-1] > 0:  
      return 'Buy'  
   elif ppo.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def ROC_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Rate of change : ((price/prevPrice)-1)*100 (ROC) indicator."""  
      
   roc = ta.ROC(data['Close'], timeperiod=10)  
   if roc.iloc[-1] > 0:  
      return 'Buy'  
   elif roc.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def ROCP_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Rate of change Percentage: (price-prevPrice)/prevPrice (ROCP) indicator."""  
      
   rocp = ta.ROCP(data['Close'], timeperiod=10)  
   if rocp.iloc[-1] > 0:  
      return 'Buy'  
   elif rocp.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def ROCR_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Rate of change ratio: (price/prevPrice) (ROCR) indicator."""  
      
   rocr = ta.ROCR(data['Close'], timeperiod=10)  
   if rocr.iloc[-1] > 1:  
      return 'Buy'  
   elif rocr.iloc[-1] < 1:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def ROCR100_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Rate of change ratio 100 scale: (price/prevPrice)*100 (ROCR100) indicator."""  
      
   rocr100 = ta.ROCR100(data['Close'], timeperiod=10)  
   if rocr100.iloc[-1] > 100:  
      return 'Buy'  
   elif rocr100.iloc[-1] < 100:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def RSI_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Relative Strength Index (RSI) indicator."""  
      
   rsi = ta.RSI(data['Close'], timeperiod=14)  
   if rsi.iloc[-1] > 70:  
      return 'Sell'  
   elif rsi.iloc[-1] < 30:  
      return 'Buy'  
   else:  
      return 'Hold'  
  
def STOCH_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Stochastic (STOCH) indicator."""  
      
   slowk, slowd = ta.STOCH(data['High'], data['Low'], data['Close'], fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)  
   if slowk.iloc[-1] > 80:  
      return 'Sell'  
   elif slowk.iloc[-1] < 20:  
      return 'Buy'  
   else:  
      return 'Hold'  
  
def STOCHF_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Stochastic Fast (STOCHF) indicator."""  
      
   fastk, fastd = ta.STOCHF(data['High'], data['Low'], data['Close'], fastk_period=5, fastd_period=3, fastd_matype=0)  
   if fastk.iloc[-1] > 80:  
      return 'Sell'  
   elif fastk.iloc[-1] < 20:  
      return 'Buy'  
   else:  
      return 'Hold'  
  
def STOCHRSI_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Stochastic Relative Strength Index (STOCHRSI) indicator."""  
      
   fastk, fastd = ta.STOCHRSI(data['Close'], timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)  
   if fastk.iloc[-1] > 80:  
      return 'Sell'  
   elif fastk.iloc[-1] < 20:  
      return 'Buy'  
   else:  
      return 'Hold'  
  
def TRIX_indicator(ticker:str, data:pd.DataFrame)->str:  
   """1-day Rate-Of-Change (ROC) of a Triple Smooth EMA (TRIX) indicator."""  
      
   trix = ta.TRIX(data['Close'], timeperiod=30)  
   if trix.iloc[-1] > 0:  
      return 'Buy'  
   elif trix.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def ULTOSC_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Ultimate Oscillator (ULTOSC) indicator."""  
      
   ultosc = ta.ULTOSC(data['High'], data['Low'], data['Close'], timeperiod1=7, timeperiod2=14, timeperiod3=28)  
   if ultosc.iloc[-1] > 70:  
      return 'Sell'  
   elif ultosc.iloc[-1] < 30:  
      return 'Buy'  
   else:  
      return 'Hold'  
  
def WILLR_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Williams' %R (WILLR) indicator."""  
      
   willr = ta.WILLR(data['High'], data['Low'], data['Close'], timeperiod=14)  
   if willr.iloc[-1] > -20:  
      return 'Sell'  
   elif willr.iloc[-1] < -80:  
      return 'Buy'  
   else:  
      return 'Hold'  
  
# Volume Indicators  
  
def AD_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Chaikin A/D Line (AD) indicator."""  
      
   ad = ta.AD(data['High'], data['Low'], data['Close'], data['Volume'])  
   if ad.iloc[-1] > 0:  
      return 'Buy'  
   elif ad.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def ADOSC_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Chaikin A/D Oscillator (ADOSC) indicator."""  
      
   adosc = ta.ADOSC(data['High'], data['Low'], data['Close'], data['Volume'], fastperiod=3, slowperiod=10)  
   if adosc.iloc[-1] > 0:  
      return 'Buy'  
   elif adosc.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def OBV_indicator(ticker:str, data:pd.DataFrame)->str:  
   """On Balance Volume (OBV) indicator."""  
      
   obv = ta.OBV(data['Close'], data['Volume'])  
   if obv.iloc[-1] > 0:  
      return 'Buy'  
   elif obv.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
# Cycle Indicators  
  
def HT_DCPERIOD_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Hilbert Transform - Dominant Cycle Period (HT_DCPERIOD) indicator."""  
      
   ht_dcperiod = ta.HT_DCPERIOD(data['Close'])  
   if ht_dcperiod.iloc[-1] > 20:  
      return 'Buy'  
   elif ht_dcperiod.iloc[-1] < 10:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def HT_DCPHASE_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Hilbert Transform - Dominant Cycle Phase (HT_DCPHASE) indicator."""  
      
   ht_dcphase = ta.HT_DCPHASE(data['Close'])  
   if ht_dcphase.iloc[-1] > 0:  
      return 'Buy'  
   elif ht_dcphase.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def HT_PHASOR_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Hilbert Transform - Phasor Components (HT_PHASOR) indicator."""  
      
   inphase, quadrature = ta.HT_PHASOR(data['Close'])  
   if inphase.iloc[-1] > 0:  
      return 'Buy'  
   elif inphase.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def HT_SINE_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Hilbert Transform - SineWave (HT_SINE) indicator."""  
      
   sine, leadsine = ta.HT_SINE(data['Close'])  
   if sine.iloc[-1] > 0:  
      return 'Buy'  
   elif sine.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def HT_TRENDMODE_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Hilbert Transform - Trend vs Cycle Mode (HT_TRENDMODE) indicator."""  
      
   ht_trendmode = ta.HT_TRENDMODE(data['Close'])  
   if ht_trendmode.iloc[-1] > 0:  
      return 'Buy'  
   elif ht_trendmode.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
# Price Transform  
  
def AVGPRICE_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Average Price (AVGPRICE) indicator."""  
      
   avgprice = ta.AVGPRICE(data['Open'], data['High'], data['Low'], data['Close'])  
   if data['Close'].iloc[-1] > avgprice.iloc[-1]:  
      return 'Buy'  
   elif data['Close'].iloc[-1] < avgprice.iloc[-1]:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def MEDPRICE_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Median Price (MEDPRICE) indicator."""  
      
   medprice = ta.MEDPRICE(data['High'], data['Low'])  
   if data['Close'].iloc[-1] > medprice.iloc[-1]:  
      return 'Buy'  
   elif data['Close'].iloc[-1] < medprice.iloc[-1]:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def TYPPRICE_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Typical Price (TYPPRICE) indicator."""  
      
   typprice = ta.TYPPRICE(data['High'], data['Low'], data['Close'])  
   if data['Close'].iloc[-1] > typprice.iloc[-1]:  
      return 'Buy'  
   elif data['Close'].iloc[-1] < typprice.iloc[-1]:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def WCLPRICE_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Weighted Close Price (WCLPRICE) indicator."""  
      
   wclprice = ta.WCLPRICE(data['High'], data['Low'], data['Close'])  
   if data['Close'].iloc[-1] > wclprice.iloc[-1]:  
      return 'Buy'  
   elif data['Close'].iloc[-1] < wclprice.iloc[-1]:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
# Volatility Indicators  
  
def ATR_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Average True Range (ATR) indicator."""  
      
   atr = ta.ATR(data['High'], data['Low'], data['Close'], timeperiod=14)  
   if atr.iloc[-1] > 20:  
      return 'Buy'  
   elif atr.iloc[-1] < 10:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def NATR_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Normalized Average True Range (NATR) indicator."""  
      
   natr = ta.NATR(data['High'], data['Low'], data['Close'], timeperiod=14)  
   if natr.iloc[-1] > 20:  
      return 'Buy'  
   elif natr.iloc[-1] < 10:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def TRANGE_indicator(ticker:str, data:pd.DataFrame)->str:  
   """True Range (TRANGE) indicator."""  
      
   trange = ta.TRANGE(data['High'], data['Low'], data['Close'])  
   if trange.iloc[-1] > 20:  
      return 'Buy'  
   elif trange.iloc[-1] < 10:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
# Pattern Recognition  
  
def CDL2CROWS_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Two Crows (CDL2CROWS) indicator."""  
      
   cdl2crows = ta.CDL2CROWS(data['Open'], data['High'], data['Low'], data['Close'])  
   if cdl2crows.iloc[-1] > 0:  
      return 'Buy'  
   elif cdl2crows.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def CDL3BLACKCROWS_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Three Black Crows (CDL3BLACKCROWS) indicator."""  
      
   cdl3blackcrows = ta.CDL3BLACKCROWS(data['Open'], data['High'], data['Low'], data['Close'])  
   if cdl3blackcrows.iloc[-1] > 0:  
      return 'Buy'  
   elif cdl3blackcrows.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def CDL3INSIDE_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Three Inside Up/Down (CDL3INSIDE) indicator."""  
      
   cdl3inside = ta.CDL3INSIDE(data['Open'], data['High'], data['Low'], data['Close'])  
   if cdl3inside.iloc[-1] > 0:  
      return 'Buy'  
   elif cdl3inside.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def CDL3LINESTRIKE_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Three-Line Strike (CDL3LINESTRIKE) indicator."""  
      
   cdl3linestrike = ta.CDL3LINESTRIKE(data['Open'], data['High'], data['Low'], data['Close'])  
   if cdl3linestrike.iloc[-1] > 0:  
      return 'Buy'  
   elif cdl3linestrike.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def CDL3OUTSIDE_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Three Outside Up/Down (CDL3OUTSIDE) indicator."""  
      
   cdl3outside = ta.CDL3OUTSIDE(data['Open'], data['High'], data['Low'], data['Close'])  
   if cdl3outside.iloc[-1] > 0:  
      return 'Buy'  
   elif cdl3outside.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def CDL3STARSINSOUTH_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Three Stars In The South (CDL3STARSINSOUTH) indicator."""  
      
   cdl3starsinsouth = ta.CDL3STARSINSOUTH(data['Open'], data['High'], data['Low'], data['Close'])  
   if cdl3starsinsouth.iloc[-1] > 0:  
      return 'Buy'  
   elif cdl3starsinsouth.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def CDL3WHITESOLDIERS_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Three Advancing White Soldiers (CDL3WHITESOLDIERS) indicator."""  
      
   cdl3whitesoldiers = ta.CDL3WHITESOLDIERS(data['Open'], data['High'], data['Low'], data['Close'])  
   if cdl3whitesoldiers.iloc[-1] > 0:  
      return 'Buy'  
   elif cdl3whitesoldiers.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def CDLABANDONEDBABY_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Abandoned Baby (CDLABANDONEDBABY) indicator."""  
      
   cdlabandonedbaby = ta.CDLABANDONEDBABY(data['Open'], data['High'], data['Low'], data['Close'], penetration=0)  
   if cdlabandonedbaby.iloc[-1] > 0:  
      return 'Buy'  
   elif cdlabandonedbaby.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def CDLADVANCEBLOCK_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Advance Block (CDLADVANCEBLOCK) indicator."""  
      
   cdladvanceblock = ta.CDLADVANCEBLOCK(data['Open'], data['High'], data['Low'], data['Close'])  
   if cdladvanceblock.iloc[-1] > 0:  
      return 'Buy'  
   elif cdladvanceblock.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def CDLBELTHOLD_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Belt-hold (CDLBELTHOLD) indicator."""  
      
   cdlbelthold = ta.CDLBELTHOLD(data['Open'], data['High'], data['Low'], data['Close'])  
   if cdlbelthold.iloc[-1] > 0:  
      return 'Buy'  
   elif cdlbelthold.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def CDLBREAKAWAY_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Breakaway (CDLBREAKAWAY) indicator."""  
      
   cdlbreakaway = ta.CDLBREAKAWAY(data['Open'], data['High'], data['Low'], data['Close'])  
   if cdlbreakaway.iloc[-1] > 0:  
      return 'Buy'  
   elif cdlbreakaway.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def CDLCLOSINGMARUBOZU_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Closing Marubozu (CDLCLOSINGMARUBOZU) indicator."""  
      
   cdlclosingmarubozu = ta.CDLCLOSINGMARUBOZU(data['Open'], data['High'], data['Low'], data['Close'])  
   if cdlclosingmarubozu.iloc[-1] > 0:  
      return 'Buy'  
   elif cdlclosingmarubozu.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def CDLCONCEALBABYSWALL_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Concealing Baby Swallow (CDLCONCEALBABYSWALL) indicator."""  
      
   cdlconcealbabyswall = ta.CDLCONCEALBABYSWALL(data['Open'], data['High'], data['Low'], data['Close'])  
   if cdlconcealbabyswall.iloc[-1] > 0:  
      return 'Buy'  
   elif cdlconcealbabyswall.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def CDLCOUNTERATTACK_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Counterattack (CDLCOUNTERATTACK) indicator."""  
      
   cdlcounterattack = ta.CDLCOUNTERATTACK(data['Open'], data['High'], data['Low'], data['Close'])  
   if cdlcounterattack.iloc[-1] > 0:  
      return 'Buy'  
   elif cdlcounterattack.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def CDLDARKCLOUDCOVER_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Dark Cloud Cover (CDLDARKCLOUDCOVER) indicator."""  
      
   cdldarkcloudcover = ta.CDLDARKCLOUDCOVER(data['Open'], data['High'], data['Low'], data['Close'], penetration=0)  
   if cdldarkcloudcover.iloc[-1] > 0:  
      return 'Buy'  
   elif cdldarkcloudcover.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def CDLDOJI_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Doji (CDLDOJI) indicator."""  
      
   cdldoji = ta.CDLDOJI(data['Open'], data['High'], data['Low'], data['Close'])  
   if cdldoji.iloc[-1] > 0:  
      return 'Buy'  
   elif cdldoji.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def CDLDOJISTAR_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Doji Star (CDLDOJISTAR) indicator."""  
      
   cdldojistar = ta.CDLDOJISTAR(data['Open'], data['High'], data['Low'], data['Close'])  
   if cdldojistar.iloc[-1] > 0:  
      return 'Buy'  
   elif cdldojistar.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def CDLDRAGONFLYDOJI_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Dragonfly Doji (CDLDRAGONFLYDOJI) indicator."""  
      
   cdldragonflydoji = ta.CDLDRAGONFLYDOJI(data['Open'], data['High'], data['Low'], data['Close'])  
   if cdldragonflydoji.iloc[-1] > 0:  
      return 'Buy'  
   elif cdldragonflydoji.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def CDLENGULFING_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Engulfing Pattern (CDLENGULFING) indicator."""  
      
   cdlengulfing = ta.CDLENGULFING(data['Open'], data['High'], data['Low'], data['Close'])  
   if cdlengulfing.iloc[-1] > 0:  
      return 'Buy'  
   elif cdlengulfing.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def CDLEVENINGDOJISTAR_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Evening Doji Star (CDLEVENINGDOJISTAR) indicator."""  
      
   cdlEveningDojiStar = ta.CDLEVENINGDOJISTAR(data['Open'], data['High'], data['Low'], data['Close'], penetration=0)  
   if cdlEveningDojiStar.iloc[-1] > 0:  
      return 'Buy'  
   elif cdlEveningDojiStar.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def CDLEVENINGSTAR_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Evening Star (CDLEVENINGSTAR) indicator."""  
      
   cdlEveningStar = ta.CDLEVENINGSTAR(data['Open'], data['High'], data['Low'], data['Close'], penetration=0)  
   if cdlEveningStar.iloc[-1] > 0:  
      return 'Buy'  
   elif cdlEveningStar.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def CDLGAPSIDESIDEWHITE_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Up/Down-gap side-by-side white lines (CDLGAPSIDESIDEWHITE) indicator."""  
      
   cdlgapsidesidewhite = ta.CDLGAPSIDESIDEWHITE(data['Open'], data['High'], data['Low'], data['Close'])  
   if cdlgapsidesidewhite.iloc[-1] > 0:  
      return 'Buy'  
   elif cdlgapsidesidewhite.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def CDLGRAVESTONEDOJI_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Gravestone Doji (CDLGRAVESTONEDOJI) indicator."""  
      
   cdlgravestonedoji = ta.CDLGRAVESTONEDOJI(data['Open'], data['High'], data['Low'], data['Close'])  
   if cdlgravestonedoji.iloc[-1] > 0:  
      return 'Buy'  
   elif cdlgravestonedoji.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def CDLHAMMER_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Hammer (CDLHAMMER) indicator."""  
      
   cdlhammer = ta.CDLHAMMER(data['Open'], data['High'], data['Low'], data['Close'])  
   if cdlhammer.iloc[-1] > 0:  
      return 'Buy'  
   elif cdlhammer.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def CDLHANGINGMAN_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Hanging Man (CDLHANGINGMAN) indicator."""  
      
   cdlhangingman = ta.CDLHANGINGMAN(data['Open'], data['High'], data['Low'], data['Close'])  
   if cdlhangingman.iloc[-1] > 0:  
      return 'Buy'  
   elif cdlhangingman.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def CDLHARAMI_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Harami Pattern (CDLHARAMI) indicator."""  
      
   cdlharami = ta.CDLHARAMI(data['Open'], data['High'], data['Low'], data['Close'])  
   if cdlharami.iloc[-1] > 0:  
      return 'Buy'  
   elif cdlharami.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def CDLHARAMICROSS_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Harami Cross Pattern (CDLHARAMICROSS) indicator."""  
      
   cdlharamicross = ta.CDLHARAMICROSS(data['Open'], data['High'], data['Low'], data['Close'])  
   if cdlharamicross.iloc[-1] > 0:  
      return 'Buy'  
   elif cdlharamicross.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def CDLHIGHWAVE_indicator(ticker:str, data:pd.DataFrame)->str:  
   """High-Wave Candle (CDLHIGHWAVE) indicator."""  
      
   cdlhighwave = ta.CDLHIGHWAVE(data['Open'], data['High'], data['Low'], data['Close'])  
   if cdlhighwave.iloc[-1] > 0:  
      return 'Buy'  
   elif cdlhighwave.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def CDLHIKKAKE_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Hikkake Pattern (CDLHIKKAKE) indicator."""  
      
   cdlhikkake = ta.CDLHIKKAKE(data['Open'], data['High'], data['Low'], data['Close'])  
   if cdlhikkake.iloc[-1] > 0:  
      return 'Buy'  
   elif cdlhikkake.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def CDLHIKKAKEMOD_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Modified Hikkake Pattern (CDLHIKKAKEMOD) indicator."""  
      
   cdlhikkakemod = ta.CDLHIKKAKEMOD(data['Open'], data['High'], data['Low'], data['Close'])  
   if cdlhikkakemod.iloc[-1] > 0:  
      return 'Buy'  
   elif cdlhikkakemod.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def CDLHOMINGPIGEON_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Homing Pigeon (CDLHOMINGPIGEON) indicator."""  
      
   cdlhomingpigeon = ta.CDLHOMINGPIGEON(data['Open'], data['High'], data['Low'], data['Close'])  
   if cdlhomingpigeon.iloc[-1] > 0:  
      return 'Buy'  
   elif cdlhomingpigeon.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def CDLIDENTICAL3CROWS_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Identical Three Crows (CDLIDENTICAL3CROWS) indicator."""  
      
   cdlidentical3crows = ta.CDLIDENTICAL3CROWS(data['Open'], data['High'], data['Low'], data['Close'])  
   if cdlidentical3crows.iloc[-1] > 0:  
      return 'Buy'  
   elif cdlidentical3crows.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def CDLINNECK_indicator(ticker:str, data:pd.DataFrame)->str:  
   """In-Neck Pattern (CDLINNECK) indicator."""  
      
   cdlInNeck = ta.CDLINNECK(data['Open'], data['High'], data['Low'], data['Close'])  
   if cdlInNeck.iloc[-1] > 0:  
      return 'Buy'  
   elif cdlInNeck.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def CDLINVERTEDHAMMER_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Inverted Hammer (CDLINVERTEDHAMMER) indicator."""  
      
   cdlInvertedHammer = ta.CDLINVERTEDHAMMER(data['Open'], data['High'], data['Low'], data['Close'])  
   if cdlInvertedHammer.iloc[-1] > 0:  
      return 'Buy'  
   elif cdlInvertedHammer.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def CDLKICKING_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Kicking (CDLKICKING) indicator."""  
      
   cdlkicking = ta.CDLKICKING(data['Open'], data['High'], data['Low'], data['Close'])  
   if cdlkicking.iloc[-1] > 0:  
      return 'Buy'  
   elif cdlkicking.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'

   
def CDLKICKINGBYLENGTH_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Kicking - bull/bear determined by the longer marubozu (CDLKICKINGBYLENGTH) indicator."""  
      
   cdlkickingbylength = ta.CDLKICKINGBYLENGTH(data['Open'], data['High'], data['Low'], data['Close'])  
   if cdlkickingbylength.iloc[-1] > 0:  
      return 'Buy'  
   elif cdlkickingbylength.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def CDLLADDERBOTTOM_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Ladder Bottom (CDLLADDERBOTTOM) indicator."""  
      
   cdlladderbottom = ta.CDLLADDERBOTTOM(data['Open'], data['High'], data['Low'], data['Close'])  
   if cdlladderbottom.iloc[-1] > 0:  
      return 'Buy'  
   elif cdlladderbottom.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def CDLLONGLEGGEDDOJI_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Long Legged Doji (CDLLONGLEGGEDDOJI) indicator."""  
      
   cdllongleggeddoji = ta.CDLLONGLEGGEDDOJI(data['Open'], data['High'], data['Low'], data['Close'])  
   if cdllongleggeddoji.iloc[-1] > 0:  
      return 'Buy'  
   elif cdllongleggeddoji.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def CDLLONGLINE_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Long Line Candle (CDLLONGLINE) indicator."""  
      
   cdllongline = ta.CDLLONGLINE(data['Open'], data['High'], data['Low'], data['Close'])  
   if cdllongline.iloc[-1] > 0:  
      return 'Buy'  
   elif cdllongline.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def CDLMARUBOZU_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Marubozu (CDLMARUBOZU) indicator."""  
      
   cdlmarubozu = ta.CDLMARUBOZU(data['Open'], data['High'], data['Low'], data['Close'])  
   if cdlmarubozu.iloc[-1] > 0:  
      return 'Buy'  
   elif cdlmarubozu.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def CDLMATCHINGLOW_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Matching Low (CDLMATCHINGLOW) indicator."""  
      
   cdlmatchinglow = ta.CDLMATCHINGLOW(data['Open'], data['High'], data['Low'], data['Close'])  
   if cdlmatchinglow.iloc[-1] > 0:  
      return 'Buy'  
   elif cdlmatchinglow.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def CDLMATHOLD_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Mat Hold (CDLMATHOLD) indicator."""  
      
   cdlmathold = ta.CDLMATHOLD(data['Open'], data['High'], data['Low'], data['Close'], penetration=0)  
   if cdlmathold.iloc[-1] > 0:  
      return 'Buy'  
   elif cdlmathold.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def CDLMORNINGDOJISTAR_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Morning Doji Star (CDLMORNINGDOJISTAR) indicator."""  
      
   cdlmorningdojistar = ta.CDLMORNINGDOJISTAR(data['Open'], data['High'], data['Low'], data['Close'], penetration=0)  
   if cdlmorningdojistar.iloc[-1] > 0:  
      return 'Buy'  
   elif cdlmorningdojistar.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def CDLMORNINGSTAR_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Morning Star (CDLMORNINGSTAR) indicator."""  
      
   cdlmorningstar = ta.CDLMORNINGSTAR(data['Open'], data['High'], data['Low'], data['Close'], penetration=0)  
   if cdlmorningstar.iloc[-1] > 0:  
      return 'Buy'  
   elif cdlmorningstar.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def CDLONNECK_indicator(ticker:str, data:pd.DataFrame)->str:  
   """On-Neck Pattern (CDLONNECK) indicator."""  
      
   cdlonneck = ta.CDLONNECK(data['Open'], data['High'], data['Low'], data['Close'])  
   if cdlonneck.iloc[-1] > 0:  
      return 'Buy'  
   elif cdlonneck.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def CDLPIERCING_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Piercing Pattern (CDLPIERCING) indicator."""  
      
   cdlpiercing = ta.CDLPIERCING(data['Open'], data['High'], data['Low'], data['Close'])  
   if cdlpiercing.iloc[-1] > 0:  
      return 'Buy'  
   elif cdlpiercing.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def CDLRICKSHAWMAN_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Rickshaw Man (CDLRICKSHAWMAN) indicator."""  
      
   cdlrickshawman = ta.CDLRICKSHAWMAN(data['Open'], data['High'], data['Low'], data['Close'])  
   if cdlrickshawman.iloc[-1] > 0:  
      return 'Buy'  
   elif cdlrickshawman.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def CDLRISEFALL3METHODS_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Rising/Falling Three Methods (CDLRISEFALL3METHODS) indicator."""  
      
   cdlrisefall3methods = ta.CDLRISEFALL3METHODS(data['Open'], data['High'], data['Low'], data['Close'])  
   if cdlrisefall3methods.iloc[-1] > 0:  
      return 'Buy'  
   elif cdlrisefall3methods.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def CDLSEPARATINGLINES_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Separating Lines (CDLSEPARATINGLINES) indicator."""  
      
   cdlseparatinglines = ta.CDLSEPARATINGLINES(data['Open'], data['High'], data['Low'], data['Close'])  
   if cdlseparatinglines.iloc[-1] > 0:  
      return 'Buy'  
   elif cdlseparatinglines.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def CDLSHOOTINGSTAR_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Shooting Star (CDLSHOOTINGSTAR) indicator."""  
      
   cdlshootingstar = ta.CDLSHOOTINGSTAR(data['Open'], data['High'], data['Low'], data['Close'])  
   if cdlshootingstar.iloc[-1] > 0:  
      return 'Buy'  
   elif cdlshootingstar.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def CDLSHORTLINE_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Short Line Candle (CDLSHORTLINE) indicator."""  
      
   cdlshortline = ta.CDLSHORTLINE(data['Open'], data['High'], data['Low'], data['Close'])  
   if cdlshortline.iloc[-1] > 0:  
      return 'Buy'  
   elif cdlshortline.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def CDLSPINNINGTOP_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Spinning Top (CDLSPINNINGTOP) indicator."""  
      
   cdlspinningtop = ta.CDLSPINNINGTOP(data['Open'], data['High'], data['Low'], data['Close'])  
   if cdlspinningtop.iloc[-1] > 0:  
      return 'Buy'  
   elif cdlspinningtop.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def CDLSTALLEDPATTERN_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Stalled Pattern (CDLSTALLEDPATTERN) indicator."""  
      
   cdlstalledpattern = ta.CDLSTALLEDPATTERN(data['Open'], data['High'], data['Low'], data['Close'])  
   if cdlstalledpattern.iloc[-1] > 0:  
      return 'Buy'  
   elif cdlstalledpattern.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def CDLSTICKSANDWICH_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Stick Sandwich (CDLSTICKSANDWICH) indicator."""  
      
   cdlsticksandwich = ta.CDLSTICKSANDWICH(data['Open'], data['High'], data['Low'], data['Close'])  
   if cdlsticksandwich.iloc[-1] > 0:  
      return 'Buy'  
   elif cdlsticksandwich.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def CDLTAKURI_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Takuri (Dragonfly Doji with very long lower shadow) (CDLTAKURI) indicator."""  
      
   cdltakuri = ta.CDLTAKURI(data['Open'], data['High'], data['Low'], data['Close'])  
   if cdltakuri.iloc[-1] > 0:  
      return 'Buy'  
   elif cdltakuri.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def CDLTASUKIGAP_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Tasuki Gap (CDLTASUKIGAP) indicator."""  
      
   cdltasukigap = ta.CDLTASUKIGAP(data['Open'], data['High'], data['Low'], data['Close'])  
   if cdltasukigap.iloc[-1] > 0:  
      return 'Buy'  
   elif cdltasukigap.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def CDLTHRUSTING_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Thrusting Pattern (CDLTHRUSTING) indicator."""  
      
   cdlthrusting = ta.CDLTHRUSTING(data['Open'], data['High'], data['Low'], data['Close'])  
   if cdlthrusting.iloc[-1] > 0:  
      return 'Buy'  
   elif cdlthrusting.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def CDLTRISTAR_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Tristar Pattern (CDLTRISTAR) indicator."""  
      
   cdltristar = ta.CDLTRISTAR(data['Open'], data['High'], data['Low'], data['Close'])  
   if cdltristar.iloc[-1] > 0:  
      return 'Buy'  
   elif cdltristar.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def CDLUNIQUE3RIVER_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Unique 3 River (CDLUNIQUE3RIVER) indicator."""  
      
   cdlunique3river = ta.CDLUNIQUE3RIVER(data['Open'], data['High'], data['Low'], data['Close'])  
   if cdlunique3river.iloc[-1] > 0:  
      return 'Buy'  
   elif cdlunique3river.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def CDLUPSIDEGAP2CROWS_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Upside Gap Two Crows (CDLUPSIDEGAP2CROWS) indicator."""  
      
   cdlupsidegap2crows = ta.CDLUPSIDEGAP2CROWS(data['Open'], data['High'], data['Low'], data['Close'])  
   if cdlupsidegap2crows.iloc[-1] > 0:  
      return 'Buy'  
   elif cdlupsidegap2crows.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def CDLXSIDEGAP3METHODS_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Upside/Downside Gap Three Methods (CDLXSIDEGAP3METHODS) indicator."""  
      
   cdlxsidegap3methods = ta.CDLXSIDEGAP3METHODS(data['Open'], data['High'], data['Low'], data['Close'])  
   if cdlxsidegap3methods.iloc[-1] > 0:  
      return 'Buy'  
   elif cdlxsidegap3methods.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
# Statistic Functions  
  
def BETA_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Beta (BETA) indicator."""  
      
   beta = ta.BETA(data['High'], data['Low'], timeperiod=5)  
   if beta.iloc[-1] > 1:  
      return 'Buy'  
   elif beta.iloc[-1] < 1:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def CORREL_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Pearson's Correlation Coefficient (r) (CORREL) indicator."""  
      
   correl = ta.CORREL(data['High'], data['Low'], timeperiod=30)  
   if correl.iloc[-1] > 0.5:  
      return 'Buy'  
   elif correl.iloc[-1] < -0.5:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def LINEARREG_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Linear Regression (LINEARREG) indicator."""  
      
   linearreg = ta.LINEARREG(data['Close'], timeperiod=14)  
   if data['Close'].iloc[-1] > linearreg.iloc[-1]:  
      return 'Buy'  
   elif data['Close'].iloc[-1] < linearreg.iloc[-1]:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def LINEARREG_ANGLE_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Linear Regression Angle (LINEARREG_ANGLE) indicator."""  
      
   linearreg_angle = ta.LINEARREG_ANGLE(data['Close'], timeperiod=14)  
   if linearreg_angle.iloc[-1] > 0:  
      return 'Buy'  
   elif linearreg_angle.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def LINEARREG_INTERCEPT_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Linear Regression Intercept (LINEARREG_INTERCEPT) indicator."""  
      
   linearreg_intercept = ta.LINEARREG_INTERCEPT(data['Close'], timeperiod=14)  
   if data['Close'].iloc[-1] > linearreg_intercept.iloc[-1]:  
      return 'Buy'  
   elif data['Close'].iloc[-1] < linearreg_intercept.iloc[-1]:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def LINEARREG_SLOPE_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Linear Regression Slope (LINEARREG_SLOPE) indicator."""  
      
   linearreg_slope = ta.LINEARREG_SLOPE(data['Close'], timeperiod=14)  
   if linearreg_slope.iloc[-1] > 0:  
      return 'Buy'  
   elif linearreg_slope.iloc[-1] < 0:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def STDDEV_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Standard Deviation (STDDEV) indicator."""  
      
   stddev = ta.STDDEV(data['Close'], timeperiod=20, nbdev=1)  
   if stddev.iloc[-1] > 20:  
      return 'Buy'  
   elif stddev.iloc[-1] < 10:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def TSF_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Time Series Forecast (TSF) indicator."""  
      
   tsf = ta.TSF(data['Close'], timeperiod=14)  
   if data['Close'].iloc[-1] > tsf.iloc[-1]:  
      return 'Buy'  
   elif data['Close'].iloc[-1] < tsf.iloc[-1]:  
      return 'Sell'  
   else:  
      return 'Hold'  
  
def VAR_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Variance (VAR) indicator."""  
      
   var = ta.VAR(data['Close'], timeperiod=5, nbdev=1)  
   if var.iloc[-1] > 20:  
      return 'Buy'  
   elif var.iloc[-1] < 10:  
      return 'Sell'  
   else:  
      return 'Hold'

