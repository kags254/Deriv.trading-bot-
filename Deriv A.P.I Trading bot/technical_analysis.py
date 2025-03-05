import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import talib

class TechnicalAnalyzer:
    def __init__(self):
        self.required_points = 100
        
    def analyze(self, prices, volumes=None):
        if len(prices) < self.required_points:
            return None
            
        df = pd.DataFrame(prices, columns=['price'])
        
        # Basic indicators
        df['sma_20'] = talib.SMA(df['price'], timeperiod=20)
        df['sma_50'] = talib.SMA(df['price'], timeperiod=50)
        df['ema_12'] = talib.EMA(df['price'], timeperiod=12)
        df['ema_26'] = talib.EMA(df['price'], timeperiod=26)
        
        # RSI
        df['rsi'] = talib.RSI(df['price'], timeperiod=14)
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
            df['price'], fastperiod=12, slowperiod=26, signalperiod=9
        )
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
            df['price'], timeperiod=20, nbdevup=2, nbdevdn=2
        )
        
        # Stochastic
        df['slowk'], df['slowd'] = talib.STOCH(
            df['price'], df['price'], df['price'],
            fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3
        )
        
        # ADX
        df['adx'] = talib.ADX(
            df['price'], df['price'], df['price'], timeperiod=14
        )
        
        # Ichimoku Cloud
        df['tenkan_sen'] = self.ichimoku_conversion_line(df['price'])
        df['kijun_sen'] = self.ichimoku_base_line(df['price'])
        df['senkou_span_a'] = self.ichimoku_leading_span_a(df['price'])
        df['senkou_span_b'] = self.ichimoku_leading_span_b(df['price'])
        
        # Support and Resistance
        df['support'], df['resistance'] = self.calculate_support_resistance(df['price'])
        
        # Trend Analysis
        df['trend'] = self.analyze_trend(df)
        
        # Pattern Recognition
        self.add_candlestick_patterns(df)
        
        return df
        
    def ichimoku_conversion_line(self, prices, period=9):
        """Calculate Ichimoku Conversion Line (Tenkan-sen)"""
        high_prices = pd.Series(prices).rolling(window=period).max()
        low_prices = pd.Series(prices).rolling(window=period).min()
        return (high_prices + low_prices) / 2

    def ichimoku_base_line(self, prices, period=26):
        """Calculate Ichimoku Base Line (Kijun-sen)"""
        high_prices = pd.Series(prices).rolling(window=period).max()
        low_prices = pd.Series(prices).rolling(window=period).min()
        return (high_prices + low_prices) / 2

    def ichimoku_leading_span_a(self, prices):
        """Calculate Ichimoku Leading Span A (Senkou Span A)"""
        conversion = self.ichimoku_conversion_line(prices)
        base = self.ichimoku_base_line(prices)
        return ((conversion + base) / 2).shift(26)

    def ichimoku_leading_span_b(self, prices, period=52):
        """Calculate Ichimoku Leading Span B (Senkou Span B)"""
        high_prices = pd.Series(prices).rolling(window=period).max()
        low_prices = pd.Series(prices).rolling(window=period).min()
        return ((high_prices + low_prices) / 2).shift(26)

    def calculate_support_resistance(self, prices, window=20):
        """Calculate dynamic support and resistance levels"""
        rolling_min = prices.rolling(window=window).min()
        rolling_max = prices.rolling(window=window).max()
        
        # Find local minima and maxima
        peaks, _ = find_peaks(prices)
        troughs, _ = find_peaks(-prices)
        
        support = rolling_min
        resistance = rolling_max
        
        return support, resistance

    def analyze_trend(self, df):
        """Determine trend direction and strength"""
        trend = pd.Series(index=df.index, dtype='str')
        
        # Strong uptrend conditions
        strong_uptrend = (
            (df['sma_20'] > df['sma_50']) &
            (df['price'] > df['bb_upper']) &
            (df['rsi'] > 60) &
            (df['macd'] > df['macd_signal'])
        )
        
        # Strong downtrend conditions
        strong_downtrend = (
            (df['sma_20'] < df['sma_50']) &
            (df['price'] < df['bb_lower']) &
            (df['rsi'] < 40) &
            (df['macd'] < df['macd_signal'])
        )
        
        # Weak uptrend conditions
        weak_uptrend = (
            (df['sma_20'] > df['sma_50']) &
            (df['price'] > df['sma_20'])
        )
        
        # Weak downtrend conditions
        weak_downtrend = (
            (df['sma_20'] < df['sma_50']) &
            (df['price'] < df['sma_20'])
        )
        
        trend[strong_uptrend] = 'strong_up'
        trend[strong_downtrend] = 'strong_down'
        trend[weak_uptrend] = 'weak_up'
        trend[weak_downtrend] = 'weak_down'
        trend[trend.isna()] = 'sideways'
        
        return trend

    def add_candlestick_patterns(self, df):
        """Add candlestick pattern recognition"""
        patterns = {
            'doji': talib.CDLDOJI,
            'hammer': talib.CDLHAMMER,
            'engulfing': talib.CDLENGULFING,
            'morning_star': talib.CDLMORNINGSTAR,
            'evening_star': talib.CDLEVENINGSTAR,
            'three_white_soldiers': talib.CDL3WHITESOLDIERS,
            'three_black_crows': talib.CDL3BLACKCROWS
        }
        
        for pattern_name, pattern_func in patterns.items():
            df[f'pattern_{pattern_name}'] = pattern_func(
                df['price'], df['price'], df['price'], df['price']
            )

    def get_trade_signals(self, df):
        """Generate trade signals based on technical analysis"""
        signals = pd.DataFrame(index=df.index)
        
        # Trend-following signals
        signals['trend_signal'] = 0
        signals.loc[df['trend'] == 'strong_up', 'trend_signal'] = 1
        signals.loc[df['trend'] == 'strong_down', 'trend_signal'] = -1
        
        # RSI signals
        signals['rsi_signal'] = 0
        signals.loc[df['rsi'] < 30, 'rsi_signal'] = 1  # Oversold
        signals.loc[df['rsi'] > 70, 'rsi_signal'] = -1  # Overbought
        
        # MACD signals
        signals['macd_signal'] = 0
        signals.loc[df['macd'] > df['macd_signal'], 'macd_signal'] = 1
        signals.loc[df['macd'] < df['macd_signal'], 'macd_signal'] = -1
        
        # Bollinger Bands signals
        signals['bb_signal'] = 0
        signals.loc[df['price'] < df['bb_lower'], 'bb_signal'] = 1
        signals.loc[df['price'] > df['bb_upper'], 'bb_signal'] = -1
        
        # Combined signal
        signals['combined_signal'] = (
            signals['trend_signal'] +
            signals['rsi_signal'] +
            signals['macd_signal'] +
            signals['bb_signal']
        )
        
        return signals

    def get_entry_points(self, df, signals):
        """Determine optimal entry points for trades"""
        entry_points = pd.DataFrame(index=df.index)
        
        # Long entry conditions
        long_entry = (
            (signals['combined_signal'] >= 2) &  # Strong bullish signals
            (df['adx'] > 25) &  # Strong trend
            (df['slowk'] < 80) &  # Not overbought
            (df['price'] > df['sma_50'])  # Above major MA
        )
        
        # Short entry conditions
        short_entry = (
            (signals['combined_signal'] <= -2) &  # Strong bearish signals
            (df['adx'] > 25) &  # Strong trend
            (df['slowk'] > 20) &  # Not oversold
            (df['price'] < df['sma_50'])  # Below major MA
        )
        
        entry_points['position'] = 0
        entry_points.loc[long_entry, 'position'] = 1
        entry_points.loc[short_entry, 'position'] = -1
        
        return entry_points

    def calculate_target_and_stop(self, df, entry_points):
        """Calculate take profit and stop loss levels"""
        targets = pd.DataFrame(index=df.index)
        
        # ATR for dynamic stop loss and take profit
        atr = talib.ATR(df['price'], df['price'], df['price'], timeperiod=14)
        
        # Calculate levels
        targets['take_profit_long'] = df['price'] + (atr * 2)
        targets['stop_loss_long'] = df['price'] - (atr * 1)
        targets['take_profit_short'] = df['price'] - (atr * 2)
        targets['stop_loss_short'] = df['price'] + (atr * 1)
        
        return targets 