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
   This function remains unchanged but will now work with indicator values
   rather than trading signals.
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
Modified interfaces for using TA-Lib indicators. 

The functions are named as the indicator name + _indicator.
The functions take a ticker and a pandas dataframe as input.
The dataframe must have a 'date' column and a 'close' column.
The functions now return the actual indicator value(s) instead of a trading signal.
"""

# Overlap Studies
def BBANDS_indicator(ticker:str, data:pd.DataFrame)->dict:  
   """Bollinger Bands (BBANDS) indicator."""  
   
   upper, middle, lower = ta.BBANDS(data['close'], timeperiod=20, matype=1)  
   return {
      "upper": upper.iloc[-1],
      "middle": middle.iloc[-1],
      "lower": lower.iloc[-1],
      "current_price": data['close'].iloc[-1],
      # Additional calculated metrics that might be useful
      "percent_b": (data['close'].iloc[-1] - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1]) if (upper.iloc[-1] - lower.iloc[-1]) != 0 else 0
   }

def DEMA_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Double Exponential Moving Average (DEMA) indicator."""  
      
   dema = ta.DEMA(data['close'], timeperiod=30)  
   return dema.iloc[-1]
  
def EMA_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Exponential Moving Average (EMA) indicator."""  
      
   ema = ta.EMA(data['close'], timeperiod=30)  
   return ema.iloc[-1]
  
def HT_TRENDLINE_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Hilbert Transform - Instantaneous Trendline (HT_TRENDLINE) indicator."""  
      
   ht_trendline = ta.HT_TRENDLINE(data['close'])  
   return ht_trendline.iloc[-1]
  
def KAMA_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Kaufman Adaptive Moving Average (KAMA) indicator."""  
      
   kama = ta.KAMA(data['close'], timeperiod=30)  
   return kama.iloc[-1]
  
def MA_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Moving average (MA) indicator."""  
      
   ma = ta.MA(data['close'], timeperiod=30, matype=0)  
   return ma.iloc[-1]

def MAMA_indicator(ticker:str, data:pd.DataFrame)->dict:
    """
    MESA Adaptive Moving Average (MAMA) indicator.
    
    Parameters:
    - ticker (str): Stock ticker symbol.

    Returns:
    - dict: Dictionary with MAMA and FAMA values.
    """
   
    close_prices = data['close'].values

    try:
        mama, fama = ta.MAMA(close_prices, fastlimit=0.5, slowlimit=0.05)
        return {
            "mama": mama[-1],
            "fama": fama[-1],
            "current_price": close_prices[-1]
        }
    except Exception as e:
        # Handle error and return NaN values
        return {
            "mama": float('nan'),
            "fama": float('nan'),
            "current_price": close_prices[-1] if len(close_prices) > 0 else float('nan')
        }
  
def MAVP_indicator(ticker:str, data:pd.DataFrame)->float:
    """
    Moving Average with Variable Period (MAVP) indicator.
    
    Parameters:
    - ticker (str): Stock ticker symbol.

    Returns:
    - float: MAVP value.
    """
     
    close_prices = data['close'].values
    # Define variable periods as a NumPy array
    variable_periods = np.full(len(close_prices), 30, dtype=np.float64)
    
    try:
        mavp = ta.MAVP(close_prices, periods=variable_periods)
        return mavp[-1]
    except Exception as e:
        return float('nan')  
  
def MIDPOINT_indicator(ticker:str, data:pd.DataFrame)->float:  
   """MidPoint over period (MIDPOINT) indicator."""  
      
   midpoint = ta.MIDPOINT(data['close'], timeperiod=14)  
   return midpoint.iloc[-1]
  
def MIDPRICE_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Midpoint Price over period (MIDPRICE) indicator."""  
   
   midprice = ta.MIDPRICE(data['high'], data['low'], timeperiod=14)  
   return midprice.iloc[-1]
  
def SAR_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Parabolic SAR (SAR) indicator."""  
      
   sar = ta.SAR(data['high'], data['low'], acceleration=0.02, maximum=0.2)  
   return sar.iloc[-1]
  
def SAREXT_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Parabolic SAR - Extended (SAREXT) indicator."""  
      
   sarext = ta.SAREXT(data['high'], data['low'], 
                      startvalue=0, offsetonreverse=0, 
                      accelerationinitlong=0.02, accelerationlong=0.02, 
                      accelerationmaxlong=0.2, accelerationinitshort=0.02, 
                      accelerationshort=0.02, accelerationmaxshort=0.2)  
   return sarext.iloc[-1]
  
def SMA_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Simple Moving Average (SMA) indicator."""  
      
   sma = ta.SMA(data['close'], timeperiod=30)  
   return sma.iloc[-1]
  
def T3_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Triple Exponential Moving Average (T3) indicator."""  
      
   t3 = ta.T3(data['close'], timeperiod=30, vfactor=0.7)  
   return t3.iloc[-1]
  
def TEMA_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Triple Exponential Moving Average (TEMA) indicator."""  
      
   tema = ta.TEMA(data['close'], timeperiod=30)  
   return tema.iloc[-1]
  
def TRIMA_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Triangular Moving Average (TRIMA) indicator."""  
      
   trima = ta.TRIMA(data['close'], timeperiod=30)  
   return trima.iloc[-1]
  
def WMA_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Weighted Moving Average (WMA) indicator."""  
      
   wma = ta.WMA(data['close'], timeperiod=30)  
   return wma.iloc[-1]
  
# Momentum Indicators  
  
def ADX_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Average Directional Movement Index (ADX) indicator."""  
      
   adx = ta.ADX(data['high'], data['low'], data['close'], timeperiod=14)  
   return adx.iloc[-1]
  
def ADXR_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Average Directional Movement Index Rating (ADXR) indicator."""  
      
   adxr = ta.ADXR(data['high'], data['low'], data['close'], timeperiod=14)  
   return adxr.iloc[-1]
  
def APO_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Absolute Price Oscillator (APO) indicator."""  
      
   apo = ta.APO(data['close'], fastperiod=12, slowperiod=26, matype=0)  
   return apo.iloc[-1]
  
def AROON_indicator(ticker:str, data:pd.DataFrame)->dict:  
   """Aroon (AROON) indicator."""  
      
   aroon_down, aroon_up = ta.AROON(data['high'], data['low'], timeperiod=14)  
   return {
       "aroon_down": aroon_down.iloc[-1],
       "aroon_up": aroon_up.iloc[-1],
       "aroon_oscillator": aroon_up.iloc[-1] - aroon_down.iloc[-1]
   }
  
def AROONOSC_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Aroon Oscillator (AROONOSC) indicator."""  
      
   aroonosc = ta.AROONOSC(data['high'], data['low'], timeperiod=14)  
   return aroonosc.iloc[-1]
  
def BOP_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Balance Of Power (BOP) indicator."""  
      
   bop = ta.BOP(data['open'], data['high'], data['low'], data['close'])  
   return bop.iloc[-1]
  
def CCI_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Commodity Channel Index (CCI) indicator."""  
      
   cci = ta.CCI(data['high'], data['low'], data['close'], timeperiod=14)  
   return cci.iloc[-1]
  
def CMO_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Chande Momentum Oscillator (CMO) indicator."""  
      
   cmo = ta.CMO(data['close'], timeperiod=14)  
   return cmo.iloc[-1]
  
def DX_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Directional Movement Index (DX) indicator."""  
      
   dx = ta.DX(data['high'], data['low'], data['close'], timeperiod=14)  
   return dx.iloc[-1]

def MACD_indicator(ticker:str, data:pd.DataFrame)->dict:  
   """Moving Average Convergence/Divergence (MACD) indicator."""  
      
   macd, macdsignal, macdhist = ta.MACD(data['close'], fastperiod=12, slowperiod=26, signalperiod=9)  
   return {
       "macd": macd.iloc[-1],
       "macdsignal": macdsignal.iloc[-1],
       "macdhist": macdhist.iloc[-1]
   }
  
def MACDEXT_indicator(ticker:str, data:pd.DataFrame)->dict:  
   """MACD with controllable MA type (MACDEXT) indicator."""  
      
   macd, macdsignal, macdhist = ta.MACDEXT(data['close'], fastperiod=12, fastmatype=0, 
                                          slowperiod=26, slowmatype=0, 
                                          signalperiod=9, signalmatype=0)  
   return {
       "macd": macd.iloc[-1],
       "macdsignal": macdsignal.iloc[-1],
       "macdhist": macdhist.iloc[-1]
   }
  
def MACDFIX_indicator(ticker:str, data:pd.DataFrame)->dict:  
   """Moving Average Convergence/Divergence Fix 12/26 (MACDFIX) indicator."""  
      
   macd, macdsignal, macdhist = ta.MACDFIX(data['close'], signalperiod=9)  
   return {
       "macd": macd.iloc[-1],
       "macdsignal": macdsignal.iloc[-1],
       "macdhist": macdhist.iloc[-1]
   }
  
def MFI_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Money Flow Index (MFI) indicator."""  
      
   mfi = ta.MFI(data['high'], data['low'], data['close'], data['volume'], timeperiod=14)  
   return mfi.iloc[-1]
  
def MINUS_DI_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Minus Directional Indicator (MINUS_DI) indicator."""  
      
   minus_di = ta.MINUS_DI(data['high'], data['low'], data['close'], timeperiod=14)  
   return minus_di.iloc[-1]
  
def MINUS_DM_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Minus Directional Movement (MINUS_DM) indicator."""  
   
   minus_dm = ta.MINUS_DM(data['high'], data['low'], timeperiod=14)  
   return minus_dm.iloc[-1]
  
def MOM_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Momentum (MOM) indicator."""  
      
   mom = ta.MOM(data['close'], timeperiod=10)  
   return mom.iloc[-1]
  
def PLUS_DI_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Plus Directional Indicator (PLUS_DI) indicator."""  
      
   plus_di = ta.PLUS_DI(data['high'], data['low'], data['close'], timeperiod=14)  
   return plus_di.iloc[-1]
  
def PLUS_DM_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Plus Directional Movement (PLUS_DM) indicator."""  
      
   plus_dm = ta.PLUS_DM(data['high'], data['low'], timeperiod=14)  
   return plus_dm.iloc[-1]
  
def PPO_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Percentage Price Oscillator (PPO) indicator."""  
      
   ppo = ta.PPO(data['close'], fastperiod=12, slowperiod=26, matype=0)  
   return ppo.iloc[-1]
  
def ROC_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Rate of change : ((price/prevPrice)-1)*100 (ROC) indicator."""  
      
   roc = ta.ROC(data['close'], timeperiod=10)  
   return roc.iloc[-1]
  
def ROCP_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Rate of change Percentage: (price-prevPrice)/prevPrice (ROCP) indicator."""  
      
   rocp = ta.ROCP(data['close'], timeperiod=10)  
   return rocp.iloc[-1]
  
def ROCR_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Rate of change ratio: (price/prevPrice) (ROCR) indicator."""  
      
   rocr = ta.ROCR(data['close'], timeperiod=10)  
   return rocr.iloc[-1]
  
def ROCR100_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Rate of change ratio 100 scale: (price/prevPrice)*100 (ROCR100) indicator."""  
      
   rocr100 = ta.ROCR100(data['close'], timeperiod=10)  
   return rocr100.iloc[-1]
  
def RSI_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Relative Strength Index (RSI) indicator."""  
      
   rsi = ta.RSI(data['close'], timeperiod=14)  
   return rsi.iloc[-1]
  
def STOCH_indicator(ticker:str, data:pd.DataFrame)->dict:  
   """Stochastic (STOCH) indicator."""  
      
   slowk, slowd = ta.STOCH(data['high'], data['low'], data['close'], 
                          fastk_period=5, slowk_period=3, slowk_matype=0, 
                          slowd_period=3, slowd_matype=0)  
   return {
       "slowk": slowk.iloc[-1],
       "slowd": slowd.iloc[-1]
   }
  
def STOCHF_indicator(ticker:str, data:pd.DataFrame)->dict:  
   """Stochastic Fast (STOCHF) indicator."""  
      
   fastk, fastd = ta.STOCHF(data['high'], data['low'], data['close'], 
                           fastk_period=5, fastd_period=3, fastd_matype=0)  
   return {
       "fastk": fastk.iloc[-1],
       "fastd": fastd.iloc[-1]
   }
  
def STOCHRSI_indicator(ticker:str, data:pd.DataFrame)->dict:  
   """Stochastic Relative Strength Index (STOCHRSI) indicator."""  
      
   fastk, fastd = ta.STOCHRSI(data['close'], timeperiod=14, 
                             fastk_period=5, fastd_period=3, fastd_matype=0)  
   return {
       "fastk": fastk.iloc[-1],
       "fastd": fastd.iloc[-1]
   }
  
def TRIX_indicator(ticker:str, data:pd.DataFrame)->float:  
   """1-day Rate-Of-Change (ROC) of a Triple Smooth EMA (TRIX) indicator."""  
      
   trix = ta.TRIX(data['close'], timeperiod=30)  
   return trix.iloc[-1]
  
def ULTOSC_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Ultimate Oscillator (ULTOSC) indicator."""  
      
   ultosc = ta.ULTOSC(data['high'], data['low'], data['close'], 
                     timeperiod1=7, timeperiod2=14, timeperiod3=28)  
   return ultosc.iloc[-1]
  
def WILLR_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Williams' %R (WILLR) indicator."""  
      
   willr = ta.WILLR(data['high'], data['low'], data['close'], timeperiod=14)  
   return willr.iloc[-1]
  
# Volume Indicators  
  
def AD_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Chaikin A/D Line (AD) indicator."""  
      
   ad = ta.AD(data['high'], data['low'], data['close'], data['volume'])  
   return ad.iloc[-1]
  
def ADOSC_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Chaikin A/D Oscillator (ADOSC) indicator."""  
      
   adosc = ta.ADOSC(data['high'], data['low'], data['close'], data['volume'], 
                   fastperiod=3, slowperiod=10)  
   return adosc.iloc[-1]
  
def OBV_indicator(ticker:str, data:pd.DataFrame)->float:  
   """On Balance Volume (OBV) indicator."""  
      
   obv = ta.OBV(data['close'], data['volume'])  
   return obv.iloc[-1]
  
# Cycle Indicators  
  
def HT_DCPERIOD_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Hilbert Transform - Dominant Cycle Period (HT_DCPERIOD) indicator."""  
      
   ht_dcperiod = ta.HT_DCPERIOD(data['close'])  
   return ht_dcperiod.iloc[-1]
  
def HT_DCPHASE_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Hilbert Transform - Dominant Cycle Phase (HT_DCPHASE) indicator."""  
      
   ht_dcphase = ta.HT_DCPHASE(data['close'])  
   return ht_dcphase.iloc[-1]
  
def HT_PHASOR_indicator(ticker:str, data:pd.DataFrame)->dict:  
   """Hilbert Transform - Phasor Components (HT_PHASOR) indicator."""  
      
   inphase, quadrature = ta.HT_PHASOR(data['close'])  
   return {
       "inphase": inphase.iloc[-1],
       "quadrature": quadrature.iloc[-1]
   }
  
def HT_SINE_indicator(ticker:str, data:pd.DataFrame)->dict:  
   """Hilbert Transform - SineWave (HT_SINE) indicator."""  
      
   sine, leadsine = ta.HT_SINE(data['close'])  
   return {
       "sine": sine.iloc[-1],
       "leadsine": leadsine.iloc[-1]
   }
  
def HT_TRENDMODE_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Hilbert Transform - Trend vs Cycle Mode (HT_TRENDMODE) indicator."""  
      
   ht_trendmode = ta.HT_TRENDMODE(data['close'])  
   return ht_trendmode.iloc[-1]
  
# Price Transform  
  
def AVGPRICE_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Average Price (AVGPRICE) indicator."""  
      
   avgprice = ta.AVGPRICE(data['open'], data['high'], data['low'], data['close'])  
   return avgprice.iloc[-1]
  
def MEDPRICE_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Median Price (MEDPRICE) indicator."""  
      
   medprice = ta.MEDPRICE(data['high'], data['low'])  
   return medprice.iloc[-1]
  
def TYPPRICE_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Typical Price (TYPPRICE) indicator."""  
      
   typprice = ta.TYPPRICE(data['high'], data['low'], data['close'])  
   return typprice.iloc[-1]
  
def WCLPRICE_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Weighted close Price (WCLPRICE) indicator."""  
      
   wclprice = ta.WCLPRICE(data['high'], data['low'], data['close'])  
   return wclprice.iloc[-1]
  
# Volatility Indicators  
  
def ATR_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Average True Range (ATR) indicator."""  
      
   atr = ta.ATR(data['high'], data['low'], data['close'], timeperiod=14)  
   return atr.iloc[-1]
  
def NATR_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Normalized Average True Range (NATR) indicator."""  
      
   natr = ta.NATR(data['high'], data['low'], data['close'], timeperiod=14)  
   return natr.iloc[-1]
  
def TRANGE_indicator(ticker:str, data:pd.DataFrame)->float:  
   """True Range (TRANGE) indicator."""  
      
   trange = ta.TRANGE(data['high'], data['low'], data['close'])  
   return trange.iloc[-1]
  
# Pattern Recognition  
# For pattern recognition, return the strength value directly (-100 to 100)

def CDL2CROWS_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Two Crows (CDL2CROWS) indicator."""  
      
   cdl2crows = ta.CDL2CROWS(data['open'], data['high'], data['low'], data['close'])  
   return cdl2crows.iloc[-1]
  
def CDL3BLACKCROWS_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Three Black Crows (CDL3BLACKCROWS) indicator."""  
      
   cdl3blackcrows = ta.CDL3BLACKCROWS(data['open'], data['high'], data['low'], data['close'])  
   return cdl3blackcrows.iloc[-1]
  
def CDL3INSIDE_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Three Inside Up/Down (CDL3INSIDE) indicator."""  
      
   cdl3inside = ta.CDL3INSIDE(data['open'], data['high'], data['low'], data['close'])  
   return cdl3inside.iloc[-1]
  
def CDL3LINESTRIKE_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Three-Line Strike (CDL3LINESTRIKE) indicator."""  
      
   cdl3linestrike = ta.CDL3LINESTRIKE(data['open'], data['high'], data['low'], data['close'])  
   return cdl3linestrike.iloc[-1]
  
def CDL3OUTSIDE_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Three Outside Up/Down (CDL3OUTSIDE) indicator."""  
      
   cdl3outside = ta.CDL3OUTSIDE(data['open'], data['high'], data['low'], data['close'])  
   return cdl3outside.iloc[-1]
  
def CDL3STARSINSOUTH_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Three Stars In The South (CDL3STARSINSOUTH) indicator."""  
      
   cdl3starsinsouth = ta.CDL3STARSINSOUTH(data['open'], data['high'], data['low'], data['close'])  
   return cdl3starsinsouth.iloc[-1]
  
def CDL3WHITESOLDIERS_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Three Advancing White Soldiers (CDL3WHITESOLDIERS) indicator."""  
      
   cdl3whitesoldiers = ta.CDL3WHITESOLDIERS(data['open'], data['high'], data['low'], data['close'])  
   return cdl3whitesoldiers.iloc[-1]
  
def CDLABANDONEDBABY_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Abandoned Baby (CDLABANDONEDBABY) indicator."""  
      
   cdlabandonedbaby = ta.CDLABANDONEDBABY(data['open'], data['high'], data['low'], data['close'], penetration=0)  
   return cdlabandonedbaby.iloc[-1]
  
def CDLADVANCEBLOCK_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Advance Block (CDLADVANCEBLOCK) indicator."""  
      
   cdladvanceblock = ta.CDLADVANCEBLOCK(data['open'], data['high'], data['low'], data['close'])  
   return cdladvanceblock.iloc[-1]
  
def CDLBELTHOLD_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Belt-hold (CDLBELTHOLD) indicator."""  
      
   cdlbelthold = ta.CDLBELTHOLD(data['open'], data['high'], data['low'], data['close'])  
   return cdlbelthold.iloc[-1]
  
def CDLBREAKAWAY_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Breakaway (CDLBREAKAWAY) indicator."""  
      
   cdlbreakaway = ta.CDLBREAKAWAY(data['open'], data['high'], data['low'], data['close'])  
   return cdlbreakaway.iloc[-1]
  
def CDLCLOSINGMARUBOZU_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Closing Marubozu (CDLCLOSINGMARUBOZU) indicator."""  
      
   cdlclosingmarubozu = ta.CDLCLOSINGMARUBOZU(data['open'], data['high'], data['low'], data['close'])  
   return cdlclosingmarubozu.iloc[-1]
  
def CDLCONCEALBABYSWALL_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Concealing Baby Swallow (CDLCONCEALBABYSWALL) indicator."""  
      
   cdlconcealbabyswall = ta.CDLCONCEALBABYSWALL(data['open'], data['high'], data['low'], data['close'])  
   return cdlconcealbabyswall.iloc[-1]
  
def CDLCOUNTERATTACK_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Counterattack (CDLCOUNTERATTACK) indicator."""  
      
   cdlcounterattack = ta.CDLCOUNTERATTACK(data['open'], data['high'], data['low'], data['close'])  
   return cdlcounterattack.iloc[-1]
  
def CDLDARKCLOUDCOVER_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Dark Cloud Cover (CDLDARKCLOUDCOVER) indicator."""  
      
   cdldarkcloudcover = ta.CDLDARKCLOUDCOVER(data['open'], data['high'], data['low'], data['close'], penetration=0)  
   return cdldarkcloudcover.iloc[-1]
  
def CDLDOJI_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Doji (CDLDOJI) indicator."""  
      
   cdldoji = ta.CDLDOJI(data['open'], data['high'], data['low'], data['close'])  
   return cdldoji.iloc[-1]
  
def CDLDOJISTAR_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Doji Star (CDLDOJISTAR) indicator."""  
      
   cdldojistar = ta.CDLDOJISTAR(data['open'], data['high'], data['low'], data['close'])  
   return cdldojistar.iloc[-1]
  
def CDLDRAGONFLYDOJI_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Dragonfly Doji (CDLDRAGONFLYDOJI) indicator."""  
      
   cdldragonflydoji = ta.CDLDRAGONFLYDOJI(data['open'], data['high'], data['low'], data['close'])  
   return cdldragonflydoji.iloc[-1]
  
def CDLENGULFING_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Engulfing Pattern (CDLENGULFING) indicator."""  
      
   cdlengulfing = ta.CDLENGULFING(data['open'], data['high'], data['low'], data['close'])  
   return cdlengulfing.iloc[-1]
  
def CDLEVENINGDOJISTAR_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Evening Doji Star (CDLEVENINGDOJISTAR) indicator."""  
      
   cdlEveningDojiStar = ta.CDLEVENINGDOJISTAR(data['open'], data['high'], data['low'], data['close'], penetration=0)  
   return cdlEveningDojiStar.iloc[-1]
  
def CDLEVENINGSTAR_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Evening Star (CDLEVENINGSTAR) indicator."""  
      
   cdlEveningStar = ta.CDLEVENINGSTAR(data['open'], data['high'], data['low'], data['close'], penetration=0)  
   return cdlEveningStar.iloc[-1]
  
def CDLGAPSIDESIDEWHITE_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Up/Down-gap side-by-side white lines (CDLGAPSIDESIDEWHITE) indicator."""  
      
   cdlgapsidesidewhite = ta.CDLGAPSIDESIDEWHITE(data['open'], data['high'], data['low'], data['close'])  
   return cdlgapsidesidewhite.iloc[-1]
  
def CDLGRAVESTONEDOJI_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Gravestone Doji (CDLGRAVESTONEDOJI) indicator."""  
      
   cdlgravestonedoji = ta.CDLGRAVESTONEDOJI(data['open'], data['high'], data['low'], data['close'])  
   return cdlgravestonedoji.iloc[-1]
  
def CDLHAMMER_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Hammer (CDLHAMMER) indicator."""  
      
   cdlhammer = ta.CDLHAMMER(data['open'], data['high'], data['low'], data['close'])  
   return cdlhammer.iloc[-1]
  
def CDLHANGINGMAN_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Hanging Man (CDLHANGINGMAN) indicator."""  
      
   cdlhangingman = ta.CDLHANGINGMAN(data['open'], data['high'], data['low'], data['close'])  
   return cdlhangingman.iloc[-1]
  
def CDLHARAMI_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Harami Pattern (CDLHARAMI) indicator."""  
      
   cdlharami = ta.CDLHARAMI(data['open'], data['high'], data['low'], data['close'])  
   return cdlharami.iloc[-1]
  
def CDLHARAMICROSS_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Harami Cross Pattern (CDLHARAMICROSS) indicator."""  
      
   cdlharamicross = ta.CDLHARAMICROSS(data['open'], data['high'], data['low'], data['close'])  
   return cdlharamicross.iloc[-1]
  
def CDLHIGHWAVE_indicator(ticker:str, data:pd.DataFrame)->float:  
   """High-Wave Candle (CDLHIGHWAVE) indicator."""  
      
   cdlhighwave = ta.CDLHIGHWAVE(data['open'], data['high'], data['low'], data['close'])  
   return cdlhighwave.iloc[-1]
  
def CDLHIKKAKE_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Hikkake Pattern (CDLHIKKAKE) indicator."""  
      
   cdlhikkake = ta.CDLHIKKAKE(data['open'], data['high'], data['low'], data['close'])  
   return cdlhikkake.iloc[-1]
  
def CDLHIKKAKEMOD_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Modified Hikkake Pattern (CDLHIKKAKEMOD) indicator."""  
      
   cdlhikkakemod = ta.CDLHIKKAKEMOD(data['open'], data['high'], data['low'], data['close'])  
   return cdlhikkakemod.iloc[-1]
  
def CDLHOMINGPIGEON_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Homing Pigeon (CDLHOMINGPIGEON) indicator."""  
      
   cdlhomingpigeon = ta.CDLHOMINGPIGEON(data['open'], data['high'], data['low'], data['close'])  
   return cdlhomingpigeon.iloc[-1]
  
def CDLIDENTICAL3CROWS_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Identical Three Crows (CDLIDENTICAL3CROWS) indicator."""  
      
   cdlidentical3crows = ta.CDLIDENTICAL3CROWS(data['open'], data['high'], data['low'], data['close'])  
   return cdlidentical3crows.iloc[-1]
  
def CDLINNECK_indicator(ticker:str, data:pd.DataFrame)->float:  
   """In-Neck Pattern (CDLINNECK) indicator."""  
      
   cdlInNeck = ta.CDLINNECK(data['open'], data['high'], data['low'], data['close'])  
   return cdlInNeck.iloc[-1]
  
def CDLINVERTEDHAMMER_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Inverted Hammer (CDLINVERTEDHAMMER) indicator."""  
      
   cdlInvertedHammer = ta.CDLINVERTEDHAMMER(data['open'], data['high'], data['low'], data['close'])  
   return cdlInvertedHammer.iloc[-1]
  
def CDLKICKING_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Kicking (CDLKICKING) indicator."""  
      
   cdlkicking = ta.CDLKICKING(data['open'], data['high'], data['low'], data['close'])  
   return cdlkicking.iloc[-1]

def CDLKICKINGBYLENGTH_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Kicking - bull/bear determined by the longer marubozu (CDLKICKINGBYLENGTH) indicator."""  
      
   cdlkickingbylength = ta.CDLKICKINGBYLENGTH(data['open'], data['high'], data['low'], data['close'])  
   return cdlkickingbylength.iloc[-1]
  
def CDLLADDERBOTTOM_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Ladder Bottom (CDLLADDERBOTTOM) indicator."""  
      
   cdlladderbottom = ta.CDLLADDERBOTTOM(data['open'], data['high'], data['low'], data['close'])  
   return cdlladderbottom.iloc[-1]
  
def CDLLONGLEGGEDDOJI_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Long Legged Doji (CDLLONGLEGGEDDOJI) indicator."""  
      
   cdllongleggeddoji = ta.CDLLONGLEGGEDDOJI(data['open'], data['high'], data['low'], data['close'])  
   return cdllongleggeddoji.iloc[-1]
  
def CDLLONGLINE_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Long Line Candle (CDLLONGLINE) indicator."""  
      
   cdllongline = ta.CDLLONGLINE(data['open'], data['high'], data['low'], data['close'])  
   return cdllongline.iloc[-1]
  
def CDLMARUBOZU_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Marubozu (CDLMARUBOZU) indicator."""  
      
   cdlmarubozu = ta.CDLMARUBOZU(data['open'], data['high'], data['low'], data['close'])  
   return cdlmarubozu.iloc[-1]
  
def CDLMATCHINGLOW_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Matching Low (CDLMATCHINGLOW) indicator."""  
      
   cdlmatchinglow = ta.CDLMATCHINGLOW(data['open'], data['high'], data['low'], data['close'])  
   return cdlmatchinglow.iloc[-1]
  
def CDLMATHOLD_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Mat hold (CDLMATHOLD) indicator."""  
      
   cdlmathold = ta.CDLMATHOLD(data['open'], data['high'], data['low'], data['close'], penetration=0)  
   return cdlmathold.iloc[-1]
  
def CDLMORNINGDOJISTAR_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Morning Doji Star (CDLMORNINGDOJISTAR) indicator."""  
      
   cdlmorningdojistar = ta.CDLMORNINGDOJISTAR(data['open'], data['high'], data['low'], data['close'], penetration=0)  
   return cdlmorningdojistar.iloc[-1]
  
def CDLMORNINGSTAR_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Morning Star (CDLMORNINGSTAR) indicator."""  
      
   cdlmorningstar = ta.CDLMORNINGSTAR(data['open'], data['high'], data['low'], data['close'], penetration=0)  
   return cdlmorningstar.iloc[-1]
  
def CDLONNECK_indicator(ticker:str, data:pd.DataFrame)->float:  
   """On-Neck Pattern (CDLONNECK) indicator."""  
      
   cdlonneck = ta.CDLONNECK(data['open'], data['high'], data['low'], data['close'])  
   return cdlonneck.iloc[-1]
  
def CDLPIERCING_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Piercing Pattern (CDLPIERCING) indicator."""  
      
   cdlpiercing = ta.CDLPIERCING(data['open'], data['high'], data['low'], data['close'])  
   return cdlpiercing.iloc[-1]
  
def CDLRICKSHAWMAN_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Rickshaw Man (CDLRICKSHAWMAN) indicator."""  
      
   cdlrickshawman = ta.CDLRICKSHAWMAN(data['open'], data['high'], data['low'], data['close'])  
   return cdlrickshawman.iloc[-1]
  
def CDLRISEFALL3METHODS_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Rising/Falling Three Methods (CDLRISEFALL3METHODS) indicator."""  
      
   cdlrisefall3methods = ta.CDLRISEFALL3METHODS(data['open'], data['high'], data['low'], data['close'])  
   return cdlrisefall3methods.iloc[-1]
  
def CDLSEPARATINGLINES_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Separating Lines (CDLSEPARATINGLINES) indicator."""  
      
   cdlseparatinglines = ta.CDLSEPARATINGLINES(data['open'], data['high'], data['low'], data['close'])  
   return cdlseparatinglines.iloc[-1]
  
def CDLSHOOTINGSTAR_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Shooting Star (CDLSHOOTINGSTAR) indicator."""  
      
   cdlshootingstar = ta.CDLSHOOTINGSTAR(data['open'], data['high'], data['low'], data['close'])  
   return cdlshootingstar.iloc[-1]
  
def CDLSHORTLINE_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Short Line Candle (CDLSHORTLINE) indicator."""  
      
   cdlshortline = ta.CDLSHORTLINE(data['open'], data['high'], data['low'], data['close'])  
   return cdlshortline.iloc[-1]
  
def CDLSPINNINGTOP_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Spinning Top (CDLSPINNINGTOP) indicator."""  
      
   cdlspinningtop = ta.CDLSPINNINGTOP(data['open'], data['high'], data['low'], data['close'])  
   return cdlspinningtop.iloc[-1]
  
def CDLSTALLEDPATTERN_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Stalled Pattern (CDLSTALLEDPATTERN) indicator."""  
      
   cdlstalledpattern = ta.CDLSTALLEDPATTERN(data['open'], data['high'], data['low'], data['close'])  
   return cdlstalledpattern.iloc[-1]
  
def CDLSTICKSANDWICH_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Stick Sandwich (CDLSTICKSANDWICH) indicator."""  
      
   cdlsticksandwich = ta.CDLSTICKSANDWICH(data['open'], data['high'], data['low'], data['close'])  
   return cdlsticksandwich.iloc[-1]
  
def CDLTAKURI_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Takuri (Dragonfly Doji with very long lower shadow) (CDLTAKURI) indicator."""  
      
   cdltakuri = ta.CDLTAKURI(data['open'], data['high'], data['low'], data['close'])  
   return cdltakuri.iloc[-1]
  
def CDLTASUKIGAP_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Tasuki Gap (CDLTASUKIGAP) indicator."""  
      
   cdltasukigap = ta.CDLTASUKIGAP(data['open'], data['high'], data['low'], data['close'])  
   return cdltasukigap.iloc[-1]
  
def CDLTHRUSTING_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Thrusting Pattern (CDLTHRUSTING) indicator."""  
      
   cdlthrusting = ta.CDLTHRUSTING(data['open'], data['high'], data['low'], data['close'])  
   return cdlthrusting.iloc[-1]
  
def CDLTRISTAR_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Tristar Pattern (CDLTRISTAR) indicator."""  
      
   cdltristar = ta.CDLTRISTAR(data['open'], data['high'], data['low'], data['close'])  
   return cdltristar.iloc[-1]
  
def CDLUNIQUE3RIVER_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Unique 3 River (CDLUNIQUE3RIVER) indicator."""  
      
   cdlunique3river = ta.CDLUNIQUE3RIVER(data['open'], data['high'], data['low'], data['close'])  
   return cdlunique3river.iloc[-1]
  
def CDLUPSIDEGAP2CROWS_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Upside Gap Two Crows (CDLUPSIDEGAP2CROWS) indicator."""  
      
   cdlupsidegap2crows = ta.CDLUPSIDEGAP2CROWS(data['open'], data['high'], data['low'], data['close'])  
   return cdlupsidegap2crows.iloc[-1]
  
def CDLXSIDEGAP3METHODS_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Upside/Downside Gap Three Methods (CDLXSIDEGAP3METHODS) indicator."""  
      
   cdlxsidegap3methods = ta.CDLXSIDEGAP3METHODS(data['open'], data['high'], data['low'], data['close'])  
   return cdlxsidegap3methods.iloc[-1]
  
# Statistic Functions  
  
def BETA_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Beta (BETA) indicator."""  
      
   beta = ta.BETA(data['high'], data['low'], timeperiod=5)  
   return beta.iloc[-1]
  
def CORREL_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Pearson's Correlation Coefficient (r) (CORREL) indicator."""  
      
   correl = ta.CORREL(data['high'], data['low'], timeperiod=30)  
   return correl.iloc[-1]
  
def LINEARREG_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Linear Regression (LINEARREG) indicator."""  
      
   linearreg = ta.LINEARREG(data['close'], timeperiod=14)  
   return linearreg.iloc[-1]
  
def LINEARREG_ANGLE_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Linear Regression Angle (LINEARREG_ANGLE) indicator."""  
      
   linearreg_angle = ta.LINEARREG_ANGLE(data['close'], timeperiod=14)  
   return linearreg_angle.iloc[-1]
  
def LINEARREG_INTERCEPT_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Linear Regression Intercept (LINEARREG_INTERCEPT) indicator."""  
      
   linearreg_intercept = ta.LINEARREG_INTERCEPT(data['close'], timeperiod=14)  
   return linearreg_intercept.iloc[-1]
  
def LINEARREG_SLOPE_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Linear Regression Slope (LINEARREG_SLOPE) indicator."""  
      
   linearreg_slope = ta.LINEARREG_SLOPE(data['close'], timeperiod=14)  
   return linearreg_slope.iloc[-1]
  
def STDDEV_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Standard Deviation (STDDEV) indicator."""  
      
   stddev = ta.STDDEV(data['close'], timeperiod=20, nbdev=1)  
   return stddev.iloc[-1]
  
def TSF_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Time Series Forecast (TSF) indicator."""  

   tsf = ta.TSF(data['close'], timeperiod=14)  
   return tsf.iloc[-1]
  
def VAR_indicator(ticker:str, data:pd.DataFrame)->float:  
   """Variance (VAR) indicator."""  
      
   var = ta.VAR(data['close'], timeperiod=5, nbdev=1)  
   return var.iloc[-1]