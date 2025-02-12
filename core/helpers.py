from core.talib_indicators import (BBANDS_indicator, DEMA_indicator, EMA_indicator, HT_TRENDLINE_indicator, KAMA_indicator, MA_indicator, MAMA_indicator, MAVP_indicator, MIDPOINT_indicator, MIDPRICE_indicator, SAR_indicator, SAREXT_indicator, SMA_indicator, T3_indicator, TEMA_indicator, TRIMA_indicator, WMA_indicator, ADX_indicator, ADXR_indicator, APO_indicator, AROON_indicator, AROONOSC_indicator, BOP_indicator, CCI_indicator, CMO_indicator, DX_indicator, MACD_indicator, MACDEXT_indicator, MACDFIX_indicator, MFI_indicator, MINUS_DI_indicator, MINUS_DM_indicator, MOM_indicator, PLUS_DI_indicator, PLUS_DM_indicator, PPO_indicator, ROC_indicator, ROCP_indicator, ROCR_indicator, ROCR100_indicator, RSI_indicator, STOCH_indicator, STOCHF_indicator, STOCHRSI_indicator, TRIX_indicator, ULTOSC_indicator, WILLR_indicator, AD_indicator, ADOSC_indicator, OBV_indicator, HT_DCPERIOD_indicator, HT_DCPHASE_indicator, HT_PHASOR_indicator, HT_SINE_indicator, HT_TRENDMODE_indicator, AVGPRICE_indicator, MEDPRICE_indicator, TYPPRICE_indicator, WCLPRICE_indicator, ATR_indicator, NATR_indicator, TRANGE_indicator, CDL2CROWS_indicator, CDL3BLACKCROWS_indicator, CDL3INSIDE_indicator, CDL3LINESTRIKE_indicator, CDL3OUTSIDE_indicator, CDL3STARSINSOUTH_indicator, CDL3WHITESOLDIERS_indicator, CDLABANDONEDBABY_indicator, CDLADVANCEBLOCK_indicator, CDLBELTHOLD_indicator, CDLBREAKAWAY_indicator, CDLCLOSINGMARUBOZU_indicator, CDLCONCEALBABYSWALL_indicator, CDLCOUNTERATTACK_indicator, CDLDARKCLOUDCOVER_indicator, CDLDOJI_indicator, CDLDOJISTAR_indicator, CDLDRAGONFLYDOJI_indicator, CDLENGULFING_indicator, CDLEVENINGDOJISTAR_indicator, CDLEVENINGSTAR_indicator, CDLGAPSIDESIDEWHITE_indicator, CDLGRAVESTONEDOJI_indicator, CDLHAMMER_indicator, CDLHANGINGMAN_indicator, CDLHARAMI_indicator, CDLHARAMICROSS_indicator, CDLHIGHWAVE_indicator, CDLHIKKAKE_indicator, CDLHIKKAKEMOD_indicator, CDLHOMINGPIGEON_indicator, CDLIDENTICAL3CROWS_indicator, CDLINNECK_indicator, CDLINVERTEDHAMMER_indicator, CDLKICKING_indicator, CDLKICKINGBYLENGTH_indicator, CDLLADDERBOTTOM_indicator, CDLLONGLEGGEDDOJI_indicator, CDLLONGLINE_indicator, CDLMARUBOZU_indicator, CDLMATCHINGLOW_indicator, CDLMATHOLD_indicator, CDLMORNINGDOJISTAR_indicator, CDLMORNINGSTAR_indicator, CDLONNECK_indicator, CDLPIERCING_indicator, CDLRICKSHAWMAN_indicator, CDLRISEFALL3METHODS_indicator, CDLSEPARATINGLINES_indicator, CDLSHOOTINGSTAR_indicator, CDLSHORTLINE_indicator, CDLSPINNINGTOP_indicator, CDLSTALLEDPATTERN_indicator, CDLSTICKSANDWICH_indicator, CDLTAKURI_indicator, CDLTASUKIGAP_indicator, CDLTHRUSTING_indicator, CDLTRISTAR_indicator, CDLUNIQUE3RIVER_indicator, CDLUPSIDEGAP2CROWS_indicator, CDLXSIDEGAP3METHODS_indicator, BETA_indicator, CORREL_indicator, LINEARREG_indicator, LINEARREG_ANGLE_indicator, LINEARREG_INTERCEPT_indicator, LINEARREG_SLOPE_indicator, STDDEV_indicator, TSF_indicator, VAR_indicator)



overlap_studies = [BBANDS_indicator, DEMA_indicator, EMA_indicator, HT_TRENDLINE_indicator, KAMA_indicator, MA_indicator, MAMA_indicator, MAVP_indicator, MIDPOINT_indicator, MIDPRICE_indicator, SAR_indicator, SAREXT_indicator, SMA_indicator, T3_indicator, TEMA_indicator, TRIMA_indicator, WMA_indicator]
momentum_indicators = [ADX_indicator, ADXR_indicator, APO_indicator, AROON_indicator, AROONOSC_indicator, BOP_indicator, CCI_indicator, CMO_indicator, DX_indicator, MACD_indicator, MACDEXT_indicator, MACDFIX_indicator, MFI_indicator, MINUS_DI_indicator, MINUS_DM_indicator, MOM_indicator, PLUS_DI_indicator, PLUS_DM_indicator, PPO_indicator, ROC_indicator, ROCP_indicator, ROCR_indicator, ROCR100_indicator, RSI_indicator, STOCH_indicator, STOCHF_indicator, STOCHRSI_indicator, TRIX_indicator, ULTOSC_indicator, WILLR_indicator]
volume_indicators = [AD_indicator, ADOSC_indicator, OBV_indicator]
cycle_indicators = [HT_DCPERIOD_indicator, HT_DCPHASE_indicator, HT_PHASOR_indicator, HT_SINE_indicator, HT_TRENDMODE_indicator]
price_transforms = [AVGPRICE_indicator, MEDPRICE_indicator, TYPPRICE_indicator, WCLPRICE_indicator]
volatility_indicators = [ATR_indicator, NATR_indicator, TRANGE_indicator]
pattern_recognition = [CDL2CROWS_indicator, CDL3BLACKCROWS_indicator, CDL3INSIDE_indicator, CDL3LINESTRIKE_indicator, CDL3OUTSIDE_indicator, CDL3STARSINSOUTH_indicator, CDL3WHITESOLDIERS_indicator, CDLABANDONEDBABY_indicator, CDLADVANCEBLOCK_indicator, CDLBELTHOLD_indicator, CDLBREAKAWAY_indicator, CDLCLOSINGMARUBOZU_indicator, CDLCONCEALBABYSWALL_indicator, CDLCOUNTERATTACK_indicator, CDLDARKCLOUDCOVER_indicator, CDLDOJI_indicator, CDLDOJISTAR_indicator, CDLDRAGONFLYDOJI_indicator, CDLENGULFING_indicator, CDLEVENINGDOJISTAR_indicator, CDLEVENINGSTAR_indicator, CDLGAPSIDESIDEWHITE_indicator, CDLGRAVESTONEDOJI_indicator, CDLHAMMER_indicator, CDLHANGINGMAN_indicator, CDLHARAMI_indicator, CDLHARAMICROSS_indicator, CDLHIGHWAVE_indicator, CDLHIKKAKE_indicator, CDLHIKKAKEMOD_indicator, CDLHOMINGPIGEON_indicator, CDLIDENTICAL3CROWS_indicator, CDLINNECK_indicator, CDLINVERTEDHAMMER_indicator, CDLKICKING_indicator, CDLKICKINGBYLENGTH_indicator, CDLLADDERBOTTOM_indicator, CDLLONGLEGGEDDOJI_indicator, CDLLONGLINE_indicator, CDLMARUBOZU_indicator, CDLMATCHINGLOW_indicator, CDLMATHOLD_indicator, CDLMORNINGDOJISTAR_indicator, CDLMORNINGSTAR_indicator, CDLONNECK_indicator, CDLPIERCING_indicator, CDLRICKSHAWMAN_indicator, CDLRISEFALL3METHODS_indicator, CDLSEPARATINGLINES_indicator, CDLSHOOTINGSTAR_indicator, CDLSHORTLINE_indicator, CDLSPINNINGTOP_indicator, CDLSTALLEDPATTERN_indicator, CDLSTICKSANDWICH_indicator, CDLTAKURI_indicator, CDLTASUKIGAP_indicator, CDLTHRUSTING_indicator, CDLTRISTAR_indicator, CDLUNIQUE3RIVER_indicator, CDLUPSIDEGAP2CROWS_indicator, CDLXSIDEGAP3METHODS_indicator]
statistical_functions = [BETA_indicator, CORREL_indicator, LINEARREG_indicator, LINEARREG_ANGLE_indicator, LINEARREG_INTERCEPT_indicator, LINEARREG_SLOPE_indicator, STDDEV_indicator, TSF_indicator, VAR_indicator]

strategies = overlap_studies + momentum_indicators + volume_indicators + cycle_indicators + price_transforms + volatility_indicators + pattern_recognition + statistical_functions
