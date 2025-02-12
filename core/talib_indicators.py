import talib as ta 
import numpy as np
import pandas as pd
import sys

sys.path.append('..')

from core.control import trade_asset_limit


def simulate_strategy(
      strategy:callable, 
      ticker:str, 
      current_price:float, 
      historical_data:pd.DataFrame, 
      account_cash:float, 
      portfolio_qty:int, 
      total_portfolio_value:float
   )->tuple[str, int]:
   """
   Simulate a trade for a ticker on a strategy.
   """
   max_investment = total_portfolio_value * trade_asset_limit
   action = strategy(ticker, historical_data)
   
   if action == 'buy':
      return 'buy', min(
                        int(max_investment // current_price - portfolio_qty), # to not exceed max investment in one stock
                        int(account_cash // current_price)) # to not exceed amount of cash in the account
   elif action == 'sell' and portfolio_qty >= 1:
      return 'sell', min(portfolio_qty, max(1, int(portfolio_qty * 0.5))) # sell half of the portfolio
   else:
      return 'hold', 0

"""
Interfaces for using TA-Lib indicators. 

The functions are named as the indicator name + _indicator.
The functions take a ticker and a pandas dataframe as input.
The dataframe must have a 'date' column and a 'close' column.
The functions return a string 'buy', 'sell', or 'hold'.
"""

# Overlap Studies
def BBANDS_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Bollinger Bands (BBANDS) indicator."""  
      
   upper, middle, lower = ta.BBANDS(data['close'], timeperiod=20)  
   # sells when the current price is above the upper band
   # and buys when the current price is below the lower band
   # otherwise, hold
   if data['close'].iloc[-1] > upper.iloc[-1]:  
      return 'sell'
   elif data['close'].iloc[-1] < lower.iloc[-1]:  
      return 'buy'  
   else:  
      return 'hold' 

def DEMA_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Double Exponential Moving Average (DEMA) indicator."""  
      
   dema = ta.DEMA(data['close'], timeperiod=30)  
   if data['close'].iloc[-1] > dema.iloc[-1]:  
      return 'buy'  
   elif data['close'].iloc[-1] < dema.iloc[-1]:  
      return 'sell'  
   else:  
      return 'hold'  
  
def EMA_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Exponential Moving Average (EMA) indicator."""  
      
   ema = ta.EMA(data['close'], timeperiod=30)  
   if data['close'].iloc[-1] > ema.iloc[-1]:  
      return 'buy'  
   elif data['close'].iloc[-1] < ema.iloc[-1]:  
      return 'sell'  
   else:  
      return 'hold'  
  
def HT_TRENDLINE_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Hilbert Transform - Instantaneous Trendline (HT_TRENDLINE) indicator."""  
      
   ht_trendline = ta.HT_TRENDLINE(data['close'])  
   if data['close'].iloc[-1] > ht_trendline.iloc[-1]:  
      return 'buy'  
   elif data['close'].iloc[-1] < ht_trendline.iloc[-1]:  
      return 'sell'  
   else:  
      return 'hold'  
  
def KAMA_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Kaufman Adaptive Moving Average (KAMA) indicator."""  
      
   kama = ta.KAMA(data['close'], timeperiod=30)  
   if data['close'].iloc[-1] > kama.iloc[-1]:  
      return 'buy'  
   elif data['close'].iloc[-1] < kama.iloc[-1]:  
      return 'sell'  
   else:  
      return 'hold'  
  
def MA_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Moving average (MA) indicator."""  
      
   ma = ta.MA(data['close'], timeperiod=30, matype=0)  
   if data['close'].iloc[-1] > ma.iloc[-1]:  
      return 'buy'  
   elif data['close'].iloc[-1] < ma.iloc[-1]:  
      return 'sell'  
   else:  
      return 'hold'  

def MAMA_indicator(ticker:str, data:pd.DataFrame)->str:
    """
    MESA Adaptive Moving Average (MAMA) indicator.
    
    Parameters:
    - ticker (str): Stock ticker symbol.

    Returns:
    - str: 'buy', 'sell', or 'hold'.
    """
   
    close_prices = data['close'].values

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
        return "sell"
    else:
        return "Hold"
  
def MAVP_indicator(ticker:str, data:pd.DataFrame)->str:
    """
    Moving Average with Variable Period (MAVP) indicator.
    
    Parameters:
    - ticker (str): Stock ticker symbol.

    Returns:
    - str: 'buy', 'sell', or 'hold'.
    """
     
    close_prices = data['close'].values
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
        return "sell"
    else:
        return "Hold"

  
def MIDPOINT_indicator(ticker:str, data:pd.DataFrame)->str:  
   """MidPoint over period (MIDPOINT) indicator."""  
      
   midpoint = ta.MIDPOINT(data['close'], timeperiod=14)  
   if data['close'].iloc[-1] > midpoint.iloc[-1]:  
      return 'buy'  
   elif data['close'].iloc[-1] < midpoint.iloc[-1]:  
      return 'sell'  
   else:  
      return 'hold'  
  
def MIDPRICE_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Midpoint Price over period (MIDPRICE) indicator."""  
      
   midprice = ta.MIDPRICE(data['high'], data['low'], timeperiod=14)  
   if data['close'].iloc[-1] > midprice.iloc[-1]:  
      return 'buy'  
   elif data['close'].iloc[-1] < midprice.iloc[-1]:  
      return 'sell'  
   else:  
      return 'hold'  
  
def SAR_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Parabolic SAR (SAR) indicator."""  
      
   sar = ta.SAR(data['high'], data['low'], acceleration=0, maximum=0)  
   if data['close'].iloc[-1] > sar.iloc[-1]:  
      return 'buy'  
   elif data['close'].iloc[-1] < sar.iloc[-1]:  
      return 'sell'  
   else:  
      return 'hold'  
  
def SAREXT_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Parabolic SAR - Extended (SAREXT) indicator."""  
      
   sarext = ta.SAREXT(data['high'], data['low'], 
                      startvalue=0, offsetonreverse=0, 
                      accelerationinitlong=0, accelerationlong=0, 
                      accelerationmaxlong=0, accelerationinitshort=0, 
                      accelerationshort=0, accelerationmaxshort=0)  
   if data['close'].iloc[-1] > sarext.iloc[-1]:  
      return 'buy'  
   elif data['close'].iloc[-1] < sarext.iloc[-1]:  
      return 'sell'  
   else:  
      return 'hold'  
  
def SMA_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Simple Moving Average (SMA) indicator."""  
      
   sma = ta.SMA(data['close'], timeperiod=30)  
   if data['close'].iloc[-1] > sma.iloc[-1]:  
      return 'buy'  
   elif data['close'].iloc[-1] < sma.iloc[-1]:  
      return 'sell'  
   else:  
      return 'hold'  
  
def T3_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Triple Exponential Moving Average (T3) indicator."""  
      
   t3 = ta.T3(data['close'], timeperiod=30, vfactor=0)  
   if data['close'].iloc[-1] > t3.iloc[-1]:  
      return 'buy'  
   elif data['close'].iloc[-1] < t3.iloc[-1]:  
      return 'sell'  
   else:  
      return 'hold'  
  
def TEMA_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Triple Exponential Moving Average (TEMA) indicator."""  
      
   tema = ta.TEMA(data['close'], timeperiod=30)  
   if data['close'].iloc[-1] > tema.iloc[-1]:  
      return 'buy'  
   elif data['close'].iloc[-1] < tema.iloc[-1]:  
      return 'sell'  
   else:  
      return 'hold'  
  
def TRIMA_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Triangular Moving Average (TRIMA) indicator."""  
      
   trima = ta.TRIMA(data['close'], timeperiod=30)  
   if data['close'].iloc[-1] > trima.iloc[-1]:  
      return 'buy'  
   elif data['close'].iloc[-1] < trima.iloc[-1]:  
      return 'sell'  
   else:  
      return 'hold'  
  
def WMA_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Weighted Moving Average (WMA) indicator."""  
      
   wma = ta.WMA(data['close'], timeperiod=30)  
   if data['close'].iloc[-1] > wma.iloc[-1]:  
      return 'buy'  
   elif data['close'].iloc[-1] < wma.iloc[-1]:  
      return 'sell'  
   else:  
      return 'hold'  
  
# Momentum Indicators  
  
def ADX_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Average Directional Movement Index (ADX) indicator."""  
      
   adx = ta.ADX(data['high'], data['low'], data['close'], timeperiod=14)  
   if adx.iloc[-1] > 25:  
      return 'buy'  
   elif adx.iloc[-1] < 20:  
      return 'sell'  
   else:  
      return 'hold'  
  
def ADXR_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Average Directional Movement Index Rating (ADXR) indicator."""  
      
   adxr = ta.ADXR(data['high'], data['low'], data['close'], timeperiod=14)  
   if adxr.iloc[-1] > 25:  
      return 'buy'  
   elif adxr.iloc[-1] < 20:  
      return 'sell'  
   else:  
      return 'hold'  
  
def APO_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Absolute Price Oscillator (APO) indicator."""  
      
   apo = ta.APO(data['close'], fastperiod=12, slowperiod=26, matype=0)  
   if apo.iloc[-1] > 0:  
      return 'buy'  
   elif apo.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def AROON_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Aroon (AROON) indicator."""  
      
   aroon_down, aroon_up = ta.AROON(data['high'], data['low'], timeperiod=14)  
   if aroon_up.iloc[-1] > 70:  
      return 'buy'  
   elif aroon_down.iloc[-1] > 70:  
      return 'sell'  
   else:  
      return 'hold'  
  
def AROONOSC_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Aroon Oscillator (AROONOSC) indicator."""  
      
   aroonosc = ta.AROONOSC(data['high'], data['low'], timeperiod=14)  
   if aroonosc.iloc[-1] > 0:  
      return 'buy'  
   elif aroonosc.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def BOP_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Balance Of Power (BOP) indicator."""  
      
   bop = ta.BOP(data['open'], data['high'], data['low'], data['close'])  
   if bop.iloc[-1] > 0:  
      return 'buy'  
   elif bop.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def CCI_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Commodity Channel Index (CCI) indicator."""  
      
   cci = ta.CCI(data['high'], data['low'], data['close'], timeperiod=14)  
   if cci.iloc[-1] > 100:  
      return 'buy'  
   elif cci.iloc[-1] < -100:  
      return 'sell'  
   else:  
      return 'hold'  
  
def CMO_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Chande Momentum Oscillator (CMO) indicator."""  
      
   cmo = ta.CMO(data['close'], timeperiod=14)  
   if cmo.iloc[-1] > 50:  
      return 'buy'  
   elif cmo.iloc[-1] < -50:  
      return 'sell'  
   else:  
      return 'hold'  
  
def DX_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Directional Movement Index (DX) indicator."""  
      
   dx = ta.DX(data['high'], data['low'], data['close'], timeperiod=14)  
   if dx.iloc[-1] > 25:  
      return 'buy'  
   elif dx.iloc[-1] < 20:  
      return 'sell'  
   else:  
      return 'hold'  
  
def MACD_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Moving Average Convergence/Divergence (MACD) indicator."""  
      
   macd, macdsignal, macdhist = ta.MACD(data['close'], fastperiod=12, slowperiod=26, signalperiod=9)  
   if macdhist.iloc[-1] > 0:  
      return 'buy'  
   elif macdhist.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def MACDEXT_indicator(ticker:str, data:pd.DataFrame)->str:  
   """MACD with controllable MA type (MACDEXT) indicator."""  
      
   macd, macdsignal, macdhist = ta.MACDEXT(data['close'], fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0)  
   if macdhist.iloc[-1] > 0:  
      return 'buy'  
   elif macdhist.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def MACDFIX_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Moving Average Convergence/Divergence Fix 12/26 (MACDFIX) indicator."""  
      
   macd, macdsignal, macdhist = ta.MACDFIX(data['close'], signalperiod=9)  
   if macdhist.iloc[-1] > 0:  
      return 'buy'  
   elif macdhist.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def MFI_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Money Flow Index (MFI) indicator."""  
      
   mfi = ta.MFI(data['high'], data['low'], data['close'], data['volume'], timeperiod=14)  
   if mfi.iloc[-1] > 80:  
      return 'sell'  
   elif mfi.iloc[-1] < 20:  
      return 'buy'  
   else:  
      return 'hold'  
  
def MINUS_DI_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Minus Directional Indicator (MINUS_DI) indicator."""  
      
   minus_di = ta.MINUS_DI(data['high'], data['low'], data['close'], timeperiod=14)  
   if minus_di.iloc[-1] > 25:  
      return 'sell'  
   elif minus_di.iloc[-1] < 20:  
      return 'buy'  
   else:  
      return 'hold'  
  
def MINUS_DM_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Minus Directional Movement (MINUS_DM) indicator."""  
      
   minus_dm = ta.MINUS_DM(data['high'], data['low'], timeperiod=14)  
   if minus_dm.iloc[-1] > 0:  
      return 'sell'  
   elif minus_dm.iloc[-1] < 0:  
      return 'buy'  
   else:  
      return 'hold'  
  
def MOM_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Momentum (MOM) indicator."""  
      
   mom = ta.MOM(data['close'], timeperiod=10)  
   if mom.iloc[-1] > 0:  
      return 'buy'  
   elif mom.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def PLUS_DI_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Plus Directional Indicator (PLUS_DI) indicator."""  
      
   plus_di = ta.PLUS_DI(data['high'], data['low'], data['close'], timeperiod=14)  
   if plus_di.iloc[-1] > 25:  
      return 'buy'  
   elif plus_di.iloc[-1] < 20:  
      return 'sell'  
   else:  
      return 'hold'  
  
def PLUS_DM_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Plus Directional Movement (PLUS_DM) indicator."""  
      
   plus_dm = ta.PLUS_DM(data['high'], data['low'], timeperiod=14)  
   if plus_dm.iloc[-1] > 0:  
      return 'buy'  
   elif plus_dm.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def PPO_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Percentage Price Oscillator (PPO) indicator."""  
      
   ppo = ta.PPO(data['close'], fastperiod=12, slowperiod=26, matype=0)  
   if ppo.iloc[-1] > 0:  
      return 'buy'  
   elif ppo.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def ROC_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Rate of change : ((price/prevPrice)-1)*100 (ROC) indicator."""  
      
   roc = ta.ROC(data['close'], timeperiod=10)  
   if roc.iloc[-1] > 0:  
      return 'buy'  
   elif roc.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def ROCP_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Rate of change Percentage: (price-prevPrice)/prevPrice (ROCP) indicator."""  
      
   rocp = ta.ROCP(data['close'], timeperiod=10)  
   if rocp.iloc[-1] > 0:  
      return 'buy'  
   elif rocp.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def ROCR_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Rate of change ratio: (price/prevPrice) (ROCR) indicator."""  
      
   rocr = ta.ROCR(data['close'], timeperiod=10)  
   if rocr.iloc[-1] > 1:  
      return 'buy'  
   elif rocr.iloc[-1] < 1:  
      return 'sell'  
   else:  
      return 'hold'  
  
def ROCR100_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Rate of change ratio 100 scale: (price/prevPrice)*100 (ROCR100) indicator."""  
      
   rocr100 = ta.ROCR100(data['close'], timeperiod=10)  
   if rocr100.iloc[-1] > 100:  
      return 'buy'  
   elif rocr100.iloc[-1] < 100:  
      return 'sell'  
   else:  
      return 'hold'  
  
def RSI_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Relative Strength Index (RSI) indicator."""  
      
   rsi = ta.RSI(data['close'], timeperiod=14)  
   if rsi.iloc[-1] > 70:  
      return 'sell'  
   elif rsi.iloc[-1] < 30:  
      return 'buy'  
   else:  
      return 'hold'  
  
def STOCH_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Stochastic (STOCH) indicator."""  
      
   slowk, slowd = ta.STOCH(data['high'], data['low'], data['close'], fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)  
   if slowk.iloc[-1] > 80:  
      return 'sell'  
   elif slowk.iloc[-1] < 20:  
      return 'buy'  
   else:  
      return 'hold'  
  
def STOCHF_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Stochastic Fast (STOCHF) indicator."""  
      
   fastk, fastd = ta.STOCHF(data['high'], data['low'], data['close'], fastk_period=5, fastd_period=3, fastd_matype=0)  
   if fastk.iloc[-1] > 80:  
      return 'sell'  
   elif fastk.iloc[-1] < 20:  
      return 'buy'  
   else:  
      return 'hold'  
  
def STOCHRSI_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Stochastic Relative Strength Index (STOCHRSI) indicator."""  
      
   fastk, fastd = ta.STOCHRSI(data['close'], timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)  
   if fastk.iloc[-1] > 80:  
      return 'sell'  
   elif fastk.iloc[-1] < 20:  
      return 'buy'  
   else:  
      return 'hold'  
  
def TRIX_indicator(ticker:str, data:pd.DataFrame)->str:  
   """1-day Rate-Of-Change (ROC) of a Triple Smooth EMA (TRIX) indicator."""  
      
   trix = ta.TRIX(data['close'], timeperiod=30)  
   if trix.iloc[-1] > 0:  
      return 'buy'  
   elif trix.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def ULTOSC_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Ultimate Oscillator (ULTOSC) indicator."""  
      
   ultosc = ta.ULTOSC(data['high'], data['low'], data['close'], timeperiod1=7, timeperiod2=14, timeperiod3=28)  
   if ultosc.iloc[-1] > 70:  
      return 'sell'  
   elif ultosc.iloc[-1] < 30:  
      return 'buy'  
   else:  
      return 'hold'  
  
def WILLR_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Williams' %R (WILLR) indicator."""  
      
   willr = ta.WILLR(data['high'], data['low'], data['close'], timeperiod=14)  
   if willr.iloc[-1] > -20:  
      return 'sell'  
   elif willr.iloc[-1] < -80:  
      return 'buy'  
   else:  
      return 'hold'  
  
# Volume Indicators  
  
def AD_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Chaikin A/D Line (AD) indicator."""  
      
   ad = ta.AD(data['high'], data['low'], data['close'], data['volume'])  
   if ad.iloc[-1] > 0:  
      return 'buy'  
   elif ad.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def ADOSC_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Chaikin A/D Oscillator (ADOSC) indicator."""  
      
   adosc = ta.ADOSC(data['high'], data['low'], data['close'], data['volume'], fastperiod=3, slowperiod=10)  
   if adosc.iloc[-1] > 0:  
      return 'buy'  
   elif adosc.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def OBV_indicator(ticker:str, data:pd.DataFrame)->str:  
   """On Balance Volume (OBV) indicator."""  
      
   obv = ta.OBV(data['close'], data['volume'])  
   if obv.iloc[-1] > 0:  
      return 'buy'  
   elif obv.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
# Cycle Indicators  
  
def HT_DCPERIOD_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Hilbert Transform - Dominant Cycle Period (HT_DCPERIOD) indicator."""  
      
   ht_dcperiod = ta.HT_DCPERIOD(data['close'])  
   if ht_dcperiod.iloc[-1] > 20:  
      return 'buy'  
   elif ht_dcperiod.iloc[-1] < 10:  
      return 'sell'  
   else:  
      return 'hold'  
  
def HT_DCPHASE_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Hilbert Transform - Dominant Cycle Phase (HT_DCPHASE) indicator."""  
      
   ht_dcphase = ta.HT_DCPHASE(data['close'])  
   if ht_dcphase.iloc[-1] > 0:  
      return 'buy'  
   elif ht_dcphase.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def HT_PHASOR_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Hilbert Transform - Phasor Components (HT_PHASOR) indicator."""  
      
   inphase, quadrature = ta.HT_PHASOR(data['close'])  
   if inphase.iloc[-1] > 0:  
      return 'buy'  
   elif inphase.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def HT_SINE_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Hilbert Transform - SineWave (HT_SINE) indicator."""  
      
   sine, leadsine = ta.HT_SINE(data['close'])  
   if sine.iloc[-1] > 0:  
      return 'buy'  
   elif sine.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def HT_TRENDMODE_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Hilbert Transform - Trend vs Cycle Mode (HT_TRENDMODE) indicator."""  
      
   ht_trendmode = ta.HT_TRENDMODE(data['close'])  
   if ht_trendmode.iloc[-1] > 0:  
      return 'buy'  
   elif ht_trendmode.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
# Price Transform  
  
def AVGPRICE_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Average Price (AVGPRICE) indicator."""  
      
   avgprice = ta.AVGPRICE(data['open'], data['high'], data['low'], data['close'])  
   if data['close'].iloc[-1] > avgprice.iloc[-1]:  
      return 'buy'  
   elif data['close'].iloc[-1] < avgprice.iloc[-1]:  
      return 'sell'  
   else:  
      return 'hold'  
  
def MEDPRICE_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Median Price (MEDPRICE) indicator."""  
      
   medprice = ta.MEDPRICE(data['high'], data['low'])  
   if data['close'].iloc[-1] > medprice.iloc[-1]:  
      return 'buy'  
   elif data['close'].iloc[-1] < medprice.iloc[-1]:  
      return 'sell'  
   else:  
      return 'hold'  
  
def TYPPRICE_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Typical Price (TYPPRICE) indicator."""  
      
   typprice = ta.TYPPRICE(data['high'], data['low'], data['close'])  
   if data['close'].iloc[-1] > typprice.iloc[-1]:  
      return 'buy'  
   elif data['close'].iloc[-1] < typprice.iloc[-1]:  
      return 'sell'  
   else:  
      return 'hold'  
  
def WCLPRICE_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Weighted close Price (WCLPRICE) indicator."""  
      
   wclprice = ta.WCLPRICE(data['high'], data['low'], data['close'])  
   if data['close'].iloc[-1] > wclprice.iloc[-1]:  
      return 'buy'  
   elif data['close'].iloc[-1] < wclprice.iloc[-1]:  
      return 'sell'  
   else:  
      return 'hold'  
  
# Volatility Indicators  
  
def ATR_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Average True Range (ATR) indicator."""  
      
   atr = ta.ATR(data['high'], data['low'], data['close'], timeperiod=14)  
   if atr.iloc[-1] > 20:  
      return 'buy'  
   elif atr.iloc[-1] < 10:  
      return 'sell'  
   else:  
      return 'hold'  
  
def NATR_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Normalized Average True Range (NATR) indicator."""  
      
   natr = ta.NATR(data['high'], data['low'], data['close'], timeperiod=14)  
   if natr.iloc[-1] > 20:  
      return 'buy'  
   elif natr.iloc[-1] < 10:  
      return 'sell'  
   else:  
      return 'hold'  
  
def TRANGE_indicator(ticker:str, data:pd.DataFrame)->str:  
   """True Range (TRANGE) indicator."""  
      
   trange = ta.TRANGE(data['high'], data['low'], data['close'])  
   if trange.iloc[-1] > 20:  
      return 'buy'  
   elif trange.iloc[-1] < 10:  
      return 'sell'  
   else:  
      return 'hold'  
  
# Pattern Recognition  
  
def CDL2CROWS_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Two Crows (CDL2CROWS) indicator."""  
      
   cdl2crows = ta.CDL2CROWS(data['open'], data['high'], data['low'], data['close'])  
   if cdl2crows.iloc[-1] > 0:  
      return 'buy'  
   elif cdl2crows.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def CDL3BLACKCROWS_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Three Black Crows (CDL3BLACKCROWS) indicator."""  
      
   cdl3blackcrows = ta.CDL3BLACKCROWS(data['open'], data['high'], data['low'], data['close'])  
   if cdl3blackcrows.iloc[-1] > 0:  
      return 'buy'  
   elif cdl3blackcrows.iloc[-1] < 0:  
      return 'sell'
   else:  
      return 'hold'  
  
def CDL3INSIDE_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Three Inside Up/Down (CDL3INSIDE) indicator."""  
      
   cdl3inside = ta.CDL3INSIDE(data['open'], data['high'], data['low'], data['close'])  
   if cdl3inside.iloc[-1] > 0:  
      return 'buy'  
   elif cdl3inside.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def CDL3LINESTRIKE_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Three-Line Strike (CDL3LINESTRIKE) indicator."""  
      
   cdl3linestrike = ta.CDL3LINESTRIKE(data['open'], data['high'], data['low'], data['close'])  
   if cdl3linestrike.iloc[-1] > 0:  
      return 'buy'  
   elif cdl3linestrike.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def CDL3OUTSIDE_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Three Outside Up/Down (CDL3OUTSIDE) indicator."""  
      
   cdl3outside = ta.CDL3OUTSIDE(data['open'], data['high'], data['low'], data['close'])  
   if cdl3outside.iloc[-1] > 0:  
      return 'buy'  
   elif cdl3outside.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def CDL3STARSINSOUTH_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Three Stars In The South (CDL3STARSINSOUTH) indicator."""  
      
   cdl3starsinsouth = ta.CDL3STARSINSOUTH(data['open'], data['high'], data['low'], data['close'])  
   if cdl3starsinsouth.iloc[-1] > 0:  
      return 'buy'  
   elif cdl3starsinsouth.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def CDL3WHITESOLDIERS_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Three Advancing White Soldiers (CDL3WHITESOLDIERS) indicator."""  
      
   cdl3whitesoldiers = ta.CDL3WHITESOLDIERS(data['open'], data['high'], data['low'], data['close'])  
   if cdl3whitesoldiers.iloc[-1] > 0:  
      return 'buy'  
   elif cdl3whitesoldiers.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def CDLABANDONEDBABY_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Abandoned Baby (CDLABANDONEDBABY) indicator."""  
      
   cdlabandonedbaby = ta.CDLABANDONEDBABY(data['open'], data['high'], data['low'], data['close'], penetration=0)  
   if cdlabandonedbaby.iloc[-1] > 0:  
      return 'buy'  
   elif cdlabandonedbaby.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def CDLADVANCEBLOCK_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Advance Block (CDLADVANCEBLOCK) indicator."""  
      
   cdladvanceblock = ta.CDLADVANCEBLOCK(data['open'], data['high'], data['low'], data['close'])  
   if cdladvanceblock.iloc[-1] > 0:  
      return 'buy'  
   elif cdladvanceblock.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def CDLBELTHOLD_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Belt-hold (CDLBELTHOLD) indicator."""  
      
   cdlbelthold = ta.CDLBELTHOLD(data['open'], data['high'], data['low'], data['close'])  
   if cdlbelthold.iloc[-1] > 0:  
      return 'buy'  
   elif cdlbelthold.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def CDLBREAKAWAY_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Breakaway (CDLBREAKAWAY) indicator."""  
      
   cdlbreakaway = ta.CDLBREAKAWAY(data['open'], data['high'], data['low'], data['close'])  
   if cdlbreakaway.iloc[-1] > 0:  
      return 'buy'  
   elif cdlbreakaway.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def CDLCLOSINGMARUBOZU_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Closing Marubozu (CDLCLOSINGMARUBOZU) indicator."""  
      
   cdlclosingmarubozu = ta.CDLCLOSINGMARUBOZU(data['open'], data['high'], data['low'], data['close'])  
   if cdlclosingmarubozu.iloc[-1] > 0:  
      return 'buy'  
   elif cdlclosingmarubozu.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def CDLCONCEALBABYSWALL_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Concealing Baby Swallow (CDLCONCEALBABYSWALL) indicator."""  
      
   cdlconcealbabyswall = ta.CDLCONCEALBABYSWALL(data['open'], data['high'], data['low'], data['close'])  
   if cdlconcealbabyswall.iloc[-1] > 0:  
      return 'buy'  
   elif cdlconcealbabyswall.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def CDLCOUNTERATTACK_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Counterattack (CDLCOUNTERATTACK) indicator."""  
      
   cdlcounterattack = ta.CDLCOUNTERATTACK(data['open'], data['high'], data['low'], data['close'])  
   if cdlcounterattack.iloc[-1] > 0:  
      return 'buy'  
   elif cdlcounterattack.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def CDLDARKCLOUDCOVER_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Dark Cloud Cover (CDLDARKCLOUDCOVER) indicator."""  
      
   cdldarkcloudcover = ta.CDLDARKCLOUDCOVER(data['open'], data['high'], data['low'], data['close'], penetration=0)  
   if cdldarkcloudcover.iloc[-1] > 0:  
      return 'buy'  
   elif cdldarkcloudcover.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def CDLDOJI_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Doji (CDLDOJI) indicator."""  
      
   cdldoji = ta.CDLDOJI(data['open'], data['high'], data['low'], data['close'])  
   if cdldoji.iloc[-1] > 0:  
      return 'buy'  
   elif cdldoji.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def CDLDOJISTAR_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Doji Star (CDLDOJISTAR) indicator."""  
      
   cdldojistar = ta.CDLDOJISTAR(data['open'], data['high'], data['low'], data['close'])  
   if cdldojistar.iloc[-1] > 0:  
      return 'buy'  
   elif cdldojistar.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def CDLDRAGONFLYDOJI_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Dragonfly Doji (CDLDRAGONFLYDOJI) indicator."""  
      
   cdldragonflydoji = ta.CDLDRAGONFLYDOJI(data['open'], data['high'], data['low'], data['close'])  
   if cdldragonflydoji.iloc[-1] > 0:  
      return 'buy'  
   elif cdldragonflydoji.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def CDLENGULFING_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Engulfing Pattern (CDLENGULFING) indicator."""  
      
   cdlengulfing = ta.CDLENGULFING(data['open'], data['high'], data['low'], data['close'])  
   if cdlengulfing.iloc[-1] > 0:  
      return 'buy'  
   elif cdlengulfing.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def CDLEVENINGDOJISTAR_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Evening Doji Star (CDLEVENINGDOJISTAR) indicator."""  
      
   cdlEveningDojiStar = ta.CDLEVENINGDOJISTAR(data['open'], data['high'], data['low'], data['close'], penetration=0)  
   if cdlEveningDojiStar.iloc[-1] > 0:  
      return 'buy'  
   elif cdlEveningDojiStar.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def CDLEVENINGSTAR_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Evening Star (CDLEVENINGSTAR) indicator."""  
      
   cdlEveningStar = ta.CDLEVENINGSTAR(data['open'], data['high'], data['low'], data['close'], penetration=0)  
   if cdlEveningStar.iloc[-1] > 0:  
      return 'buy'  
   elif cdlEveningStar.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def CDLGAPSIDESIDEWHITE_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Up/Down-gap side-by-side white lines (CDLGAPSIDESIDEWHITE) indicator."""  
      
   cdlgapsidesidewhite = ta.CDLGAPSIDESIDEWHITE(data['open'], data['high'], data['low'], data['close'])  
   if cdlgapsidesidewhite.iloc[-1] > 0:  
      return 'buy'  
   elif cdlgapsidesidewhite.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def CDLGRAVESTONEDOJI_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Gravestone Doji (CDLGRAVESTONEDOJI) indicator."""  
      
   cdlgravestonedoji = ta.CDLGRAVESTONEDOJI(data['open'], data['high'], data['low'], data['close'])  
   if cdlgravestonedoji.iloc[-1] > 0:  
      return 'buy'  
   elif cdlgravestonedoji.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def CDLHAMMER_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Hammer (CDLHAMMER) indicator."""  
      
   cdlhammer = ta.CDLHAMMER(data['open'], data['high'], data['low'], data['close'])  
   if cdlhammer.iloc[-1] > 0:  
      return 'buy'  
   elif cdlhammer.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def CDLHANGINGMAN_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Hanging Man (CDLHANGINGMAN) indicator."""  
      
   cdlhangingman = ta.CDLHANGINGMAN(data['open'], data['high'], data['low'], data['close'])  
   if cdlhangingman.iloc[-1] > 0:  
      return 'buy'  
   elif cdlhangingman.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def CDLHARAMI_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Harami Pattern (CDLHARAMI) indicator."""  
      
   cdlharami = ta.CDLHARAMI(data['open'], data['high'], data['low'], data['close'])  
   if cdlharami.iloc[-1] > 0:  
      return 'buy'  
   elif cdlharami.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def CDLHARAMICROSS_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Harami Cross Pattern (CDLHARAMICROSS) indicator."""  
      
   cdlharamicross = ta.CDLHARAMICROSS(data['open'], data['high'], data['low'], data['close'])  
   if cdlharamicross.iloc[-1] > 0:  
      return 'buy'  
   elif cdlharamicross.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def CDLHIGHWAVE_indicator(ticker:str, data:pd.DataFrame)->str:  
   """High-Wave Candle (CDLHIGHWAVE) indicator."""  
      
   cdlhighwave = ta.CDLHIGHWAVE(data['open'], data['high'], data['low'], data['close'])  
   if cdlhighwave.iloc[-1] > 0:  
      return 'buy'  
   elif cdlhighwave.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def CDLHIKKAKE_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Hikkake Pattern (CDLHIKKAKE) indicator."""  
      
   cdlhikkake = ta.CDLHIKKAKE(data['open'], data['high'], data['low'], data['close'])  
   if cdlhikkake.iloc[-1] > 0:  
      return 'buy'  
   elif cdlhikkake.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def CDLHIKKAKEMOD_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Modified Hikkake Pattern (CDLHIKKAKEMOD) indicator."""  
      
   cdlhikkakemod = ta.CDLHIKKAKEMOD(data['open'], data['high'], data['low'], data['close'])  
   if cdlhikkakemod.iloc[-1] > 0:  
      return 'buy'  
   elif cdlhikkakemod.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def CDLHOMINGPIGEON_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Homing Pigeon (CDLHOMINGPIGEON) indicator."""  
      
   cdlhomingpigeon = ta.CDLHOMINGPIGEON(data['open'], data['high'], data['low'], data['close'])  
   if cdlhomingpigeon.iloc[-1] > 0:  
      return 'buy'  
   elif cdlhomingpigeon.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def CDLIDENTICAL3CROWS_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Identical Three Crows (CDLIDENTICAL3CROWS) indicator."""  
      
   cdlidentical3crows = ta.CDLIDENTICAL3CROWS(data['open'], data['high'], data['low'], data['close'])  
   if cdlidentical3crows.iloc[-1] > 0:  
      return 'buy'  
   elif cdlidentical3crows.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def CDLINNECK_indicator(ticker:str, data:pd.DataFrame)->str:  
   """In-Neck Pattern (CDLINNECK) indicator."""  
      
   cdlInNeck = ta.CDLINNECK(data['open'], data['high'], data['low'], data['close'])  
   if cdlInNeck.iloc[-1] > 0:  
      return 'buy'  
   elif cdlInNeck.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def CDLINVERTEDHAMMER_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Inverted Hammer (CDLINVERTEDHAMMER) indicator."""  
      
   cdlInvertedHammer = ta.CDLINVERTEDHAMMER(data['open'], data['high'], data['low'], data['close'])  
   if cdlInvertedHammer.iloc[-1] > 0:  
      return 'buy'  
   elif cdlInvertedHammer.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def CDLKICKING_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Kicking (CDLKICKING) indicator."""  
      
   cdlkicking = ta.CDLKICKING(data['open'], data['high'], data['low'], data['close'])  
   if cdlkicking.iloc[-1] > 0:  
      return 'buy'  
   elif cdlkicking.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'

   
def CDLKICKINGBYLENGTH_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Kicking - bull/bear determined by the longer marubozu (CDLKICKINGBYLENGTH) indicator."""  
      
   cdlkickingbylength = ta.CDLKICKINGBYLENGTH(data['open'], data['high'], data['low'], data['close'])  
   if cdlkickingbylength.iloc[-1] > 0:  
      return 'buy'  
   elif cdlkickingbylength.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def CDLLADDERBOTTOM_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Ladder Bottom (CDLLADDERBOTTOM) indicator."""  
      
   cdlladderbottom = ta.CDLLADDERBOTTOM(data['open'], data['high'], data['low'], data['close'])  
   if cdlladderbottom.iloc[-1] > 0:  
      return 'buy'  
   elif cdlladderbottom.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def CDLLONGLEGGEDDOJI_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Long Legged Doji (CDLLONGLEGGEDDOJI) indicator."""  
      
   cdllongleggeddoji = ta.CDLLONGLEGGEDDOJI(data['open'], data['high'], data['low'], data['close'])  
   if cdllongleggeddoji.iloc[-1] > 0:  
      return 'buy'  
   elif cdllongleggeddoji.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def CDLLONGLINE_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Long Line Candle (CDLLONGLINE) indicator."""  
      
   cdllongline = ta.CDLLONGLINE(data['open'], data['high'], data['low'], data['close'])  
   if cdllongline.iloc[-1] > 0:  
      return 'buy'  
   elif cdllongline.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def CDLMARUBOZU_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Marubozu (CDLMARUBOZU) indicator."""  
      
   cdlmarubozu = ta.CDLMARUBOZU(data['open'], data['high'], data['low'], data['close'])  
   if cdlmarubozu.iloc[-1] > 0:  
      return 'buy'  
   elif cdlmarubozu.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def CDLMATCHINGLOW_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Matching Low (CDLMATCHINGLOW) indicator."""  
      
   cdlmatchinglow = ta.CDLMATCHINGLOW(data['open'], data['high'], data['low'], data['close'])  
   if cdlmatchinglow.iloc[-1] > 0:  
      return 'buy'  
   elif cdlmatchinglow.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def CDLMATHOLD_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Mat Hold (CDLMATHOLD) indicator."""  
      
   cdlmathold = ta.CDLMATHOLD(data['open'], data['high'], data['low'], data['close'], penetration=0)  
   if cdlmathold.iloc[-1] > 0:  
      return 'buy'  
   elif cdlmathold.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def CDLMORNINGDOJISTAR_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Morning Doji Star (CDLMORNINGDOJISTAR) indicator."""  
      
   cdlmorningdojistar = ta.CDLMORNINGDOJISTAR(data['open'], data['high'], data['low'], data['close'], penetration=0)  
   if cdlmorningdojistar.iloc[-1] > 0:  
      return 'buy'  
   elif cdlmorningdojistar.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def CDLMORNINGSTAR_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Morning Star (CDLMORNINGSTAR) indicator."""  
      
   cdlmorningstar = ta.CDLMORNINGSTAR(data['open'], data['high'], data['low'], data['close'], penetration=0)  
   if cdlmorningstar.iloc[-1] > 0:  
      return 'buy'  
   elif cdlmorningstar.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def CDLONNECK_indicator(ticker:str, data:pd.DataFrame)->str:  
   """On-Neck Pattern (CDLONNECK) indicator."""  
      
   cdlonneck = ta.CDLONNECK(data['open'], data['high'], data['low'], data['close'])  
   if cdlonneck.iloc[-1] > 0:  
      return 'buy'  
   elif cdlonneck.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def CDLPIERCING_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Piercing Pattern (CDLPIERCING) indicator."""  
      
   cdlpiercing = ta.CDLPIERCING(data['open'], data['high'], data['low'], data['close'])  
   if cdlpiercing.iloc[-1] > 0:  
      return 'buy'  
   elif cdlpiercing.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def CDLRICKSHAWMAN_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Rickshaw Man (CDLRICKSHAWMAN) indicator."""  
      
   cdlrickshawman = ta.CDLRICKSHAWMAN(data['open'], data['high'], data['low'], data['close'])  
   if cdlrickshawman.iloc[-1] > 0:  
      return 'buy'  
   elif cdlrickshawman.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def CDLRISEFALL3METHODS_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Rising/Falling Three Methods (CDLRISEFALL3METHODS) indicator."""  
      
   cdlrisefall3methods = ta.CDLRISEFALL3METHODS(data['open'], data['high'], data['low'], data['close'])  
   if cdlrisefall3methods.iloc[-1] > 0:  
      return 'buy'  
   elif cdlrisefall3methods.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def CDLSEPARATINGLINES_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Separating Lines (CDLSEPARATINGLINES) indicator."""  
      
   cdlseparatinglines = ta.CDLSEPARATINGLINES(data['open'], data['high'], data['low'], data['close'])  
   if cdlseparatinglines.iloc[-1] > 0:  
      return 'buy'  
   elif cdlseparatinglines.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def CDLSHOOTINGSTAR_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Shooting Star (CDLSHOOTINGSTAR) indicator."""  
      
   cdlshootingstar = ta.CDLSHOOTINGSTAR(data['open'], data['high'], data['low'], data['close'])  
   if cdlshootingstar.iloc[-1] > 0:  
      return 'buy'  
   elif cdlshootingstar.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def CDLSHORTLINE_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Short Line Candle (CDLSHORTLINE) indicator."""  
      
   cdlshortline = ta.CDLSHORTLINE(data['open'], data['high'], data['low'], data['close'])  
   if cdlshortline.iloc[-1] > 0:  
      return 'buy'  
   elif cdlshortline.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def CDLSPINNINGTOP_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Spinning Top (CDLSPINNINGTOP) indicator."""  
      
   cdlspinningtop = ta.CDLSPINNINGTOP(data['open'], data['high'], data['low'], data['close'])  
   if cdlspinningtop.iloc[-1] > 0:  
      return 'buy'  
   elif cdlspinningtop.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def CDLSTALLEDPATTERN_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Stalled Pattern (CDLSTALLEDPATTERN) indicator."""  
      
   cdlstalledpattern = ta.CDLSTALLEDPATTERN(data['open'], data['high'], data['low'], data['close'])  
   if cdlstalledpattern.iloc[-1] > 0:  
      return 'buy'  
   elif cdlstalledpattern.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def CDLSTICKSANDWICH_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Stick Sandwich (CDLSTICKSANDWICH) indicator."""  
      
   cdlsticksandwich = ta.CDLSTICKSANDWICH(data['open'], data['high'], data['low'], data['close'])  
   if cdlsticksandwich.iloc[-1] > 0:  
      return 'buy'  
   elif cdlsticksandwich.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def CDLTAKURI_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Takuri (Dragonfly Doji with very long lower shadow) (CDLTAKURI) indicator."""  
      
   cdltakuri = ta.CDLTAKURI(data['open'], data['high'], data['low'], data['close'])  
   if cdltakuri.iloc[-1] > 0:  
      return 'buy'  
   elif cdltakuri.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def CDLTASUKIGAP_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Tasuki Gap (CDLTASUKIGAP) indicator."""  
      
   cdltasukigap = ta.CDLTASUKIGAP(data['open'], data['high'], data['low'], data['close'])  
   if cdltasukigap.iloc[-1] > 0:  
      return 'buy'  
   elif cdltasukigap.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def CDLTHRUSTING_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Thrusting Pattern (CDLTHRUSTING) indicator."""  
      
   cdlthrusting = ta.CDLTHRUSTING(data['open'], data['high'], data['low'], data['close'])  
   if cdlthrusting.iloc[-1] > 0:  
      return 'buy'  
   elif cdlthrusting.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def CDLTRISTAR_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Tristar Pattern (CDLTRISTAR) indicator."""  
      
   cdltristar = ta.CDLTRISTAR(data['open'], data['high'], data['low'], data['close'])  
   if cdltristar.iloc[-1] > 0:  
      return 'buy'  
   elif cdltristar.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def CDLUNIQUE3RIVER_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Unique 3 River (CDLUNIQUE3RIVER) indicator."""  
      
   cdlunique3river = ta.CDLUNIQUE3RIVER(data['open'], data['high'], data['low'], data['close'])  
   if cdlunique3river.iloc[-1] > 0:  
      return 'buy'  
   elif cdlunique3river.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def CDLUPSIDEGAP2CROWS_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Upside Gap Two Crows (CDLUPSIDEGAP2CROWS) indicator."""  
      
   cdlupsidegap2crows = ta.CDLUPSIDEGAP2CROWS(data['open'], data['high'], data['low'], data['close'])  
   if cdlupsidegap2crows.iloc[-1] > 0:  
      return 'buy'  
   elif cdlupsidegap2crows.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def CDLXSIDEGAP3METHODS_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Upside/Downside Gap Three Methods (CDLXSIDEGAP3METHODS) indicator."""  
      
   cdlxsidegap3methods = ta.CDLXSIDEGAP3METHODS(data['open'], data['high'], data['low'], data['close'])  
   if cdlxsidegap3methods.iloc[-1] > 0:  
      return 'buy'  
   elif cdlxsidegap3methods.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
# Statistic Functions  
  
def BETA_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Beta (BETA) indicator."""  
      
   beta = ta.BETA(data['high'], data['low'], timeperiod=5)  
   if beta.iloc[-1] > 1:  
      return 'buy'  
   elif beta.iloc[-1] < 1:  
      return 'sell'  
   else:  
      return 'hold'  
  
def CORREL_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Pearson's Correlation Coefficient (r) (CORREL) indicator."""  
      
   correl = ta.CORREL(data['high'], data['low'], timeperiod=30)  
   if correl.iloc[-1] > 0.5:  
      return 'buy'  
   elif correl.iloc[-1] < -0.5:  
      return 'sell'  
   else:  
      return 'hold'  
  
def LINEARREG_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Linear Regression (LINEARREG) indicator."""  
      
   linearreg = ta.LINEARREG(data['close'], timeperiod=14)  
   if data['close'].iloc[-1] > linearreg.iloc[-1]:  
      return 'buy'  
   elif data['close'].iloc[-1] < linearreg.iloc[-1]:  
      return 'sell'  
   else:  
      return 'hold'  
  
def LINEARREG_ANGLE_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Linear Regression Angle (LINEARREG_ANGLE) indicator."""  
      
   linearreg_angle = ta.LINEARREG_ANGLE(data['close'], timeperiod=14)  
   if linearreg_angle.iloc[-1] > 0:  
      return 'buy'  
   elif linearreg_angle.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def LINEARREG_INTERCEPT_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Linear Regression Intercept (LINEARREG_INTERCEPT) indicator."""  
      
   linearreg_intercept = ta.LINEARREG_INTERCEPT(data['close'], timeperiod=14)  
   if data['close'].iloc[-1] > linearreg_intercept.iloc[-1]:  
      return 'buy'  
   elif data['close'].iloc[-1] < linearreg_intercept.iloc[-1]:  
      return 'sell'  
   else:  
      return 'hold'  
  
def LINEARREG_SLOPE_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Linear Regression Slope (LINEARREG_SLOPE) indicator."""  
      
   linearreg_slope = ta.LINEARREG_SLOPE(data['close'], timeperiod=14)  
   if linearreg_slope.iloc[-1] > 0:  
      return 'buy'  
   elif linearreg_slope.iloc[-1] < 0:  
      return 'sell'  
   else:  
      return 'hold'  
  
def STDDEV_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Standard Deviation (STDDEV) indicator."""  
      
   stddev = ta.STDDEV(data['close'], timeperiod=20, nbdev=1)  
   if stddev.iloc[-1] > 20:  
      return 'buy'  
   elif stddev.iloc[-1] < 10:  
      return 'sell'  
   else:  
      return 'hold'  
  
def TSF_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Time Series Forecast (TSF) indicator."""  

   tsf = ta.TSF(data['close'], timeperiod=14)  
   if data['close'].iloc[-1] > tsf.iloc[-1]:  
      return 'buy'  
   elif data['close'].iloc[-1] < tsf.iloc[-1]:  
      return 'sell'  
   else:  
      return 'hold'  
  
def VAR_indicator(ticker:str, data:pd.DataFrame)->str:  
   """Variance (VAR) indicator."""  
      
   var = ta.VAR(data['close'], timeperiod=5, nbdev=1)  
   if var.iloc[-1] > 20:  
      return 'buy'  
   elif var.iloc[-1] < 10:  
      return 'sell'  
   else:  
      return 'hold'

