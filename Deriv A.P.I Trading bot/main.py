import json
import asyncio
import websockets
import os
import numpy as np
from dotenv import load_dotenv
from collections import Counter, deque
from datetime import datetime, timedelta
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import logging
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import threading
from virtual_trader import VirtualTrader

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)

# Load environment variables
load_dotenv()

class WebDashboard:
    def __init__(self, bot):
        self.bot = bot
        self.app = dash.Dash(__name__)
        self.setup_layout()
        
    def setup_layout(self):
        self.app.layout = html.Div([
            html.H1('Deriv Trading Bot Dashboard'),
            dcc.Interval(id='interval-component', interval=5000),
            html.Div([
                html.H2('Active Markets'),
                html.Div(id='active-markets')
            ]),
            html.Div([
                html.H2('Performance Metrics'),
                html.Div(id='performance-metrics')
            ]),
            dcc.Graph(id='profit-chart'),
            dcc.Graph(id='win-rate-chart')
        ])
        
        @self.app.callback(
            [Output('active-markets', 'children'),
             Output('performance-metrics', 'children'),
             Output('profit-chart', 'figure'),
             Output('win-rate-chart', 'figure')],
            Input('interval-component', 'n_intervals')
        )
        def update_dashboard(_):
            return self.update_dashboard_data()
    
    def update_dashboard_data(self):
        active_markets = html.Ul([html.Li(market) for market in self.bot.active_markets])
        
        metrics = self.bot.metrics.get_strategy_metrics()
        performance = html.Table([
            html.Tr([html.Th('Strategy'), html.Th('Win Rate'), html.Th('Profit')])
        ] + [
            html.Tr([
                html.Td(strategy),
                html.Td(f"{data['win_rate']:.2f}%"),
                html.Td(f"${data['profit']:.2f}")
            ]) for strategy, data in metrics.items()
        ])
        
        profit_fig = go.Figure()
        for market in self.bot.active_markets:
            market_data = self.bot.market_analyzer.market_metrics.get(market, {})
            if 'trades' in market_data:
                trades = market_data['trades']
                profit_fig.add_trace(go.Scatter(
                    x=[t['timestamp'] for t in trades],
                    y=[t['profit'] for t in trades],
                    name=market
                ))
        
        win_rate_fig = go.Figure()
        for market in self.bot.active_markets:
            metrics = self.bot.market_analyzer.get_market_win_rates(market)
            if metrics:
                win_rate_fig.add_trace(go.Scatter(
                    x=list(metrics.keys()),
                    y=list(metrics.values()),
                    name=market
                ))
        
        return active_markets, performance, profit_fig, win_rate_fig
    
    def run(self):
        self.app.run_server(debug=True, port=8050)

class TradingMetrics:
    def __init__(self):
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.profit_loss = 0
        self.trade_history = []
        self.strategy_performance = {}
        
    def update(self, strategy, profit, won):
        self.total_trades += 1
        self.profit_loss += profit
        
        if won:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
            
        if strategy not in self.strategy_performance:
            self.strategy_performance[strategy] = {
                'wins': 0, 'losses': 0, 'profit': 0
            }
            
        self.strategy_performance[strategy]['profit'] += profit
        if won:
            self.strategy_performance[strategy]['wins'] += 1
        else:
            self.strategy_performance[strategy]['losses'] += 1
            
        self.trade_history.append({
            'timestamp': datetime.now(),
            'strategy': strategy,
            'profit': profit,
            'won': won
        })
        
    def get_win_rate(self):
        return (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        
    def get_strategy_metrics(self):
        metrics = {}
        for strategy, data in self.strategy_performance.items():
            total = data['wins'] + data['losses']
            win_rate = (data['wins'] / total * 100) if total > 0 else 0
            metrics[strategy] = {
                'win_rate': win_rate,
                'profit': data['profit'],
                'total_trades': total
            }
        return metrics

class RiskManager:
    def __init__(self, initial_balance, max_risk_per_trade=0.02):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.max_risk_per_trade = max_risk_per_trade
        self.daily_loss_limit = initial_balance * 0.1  # 10% daily loss limit
        self.daily_loss = 0
        self.last_reset = datetime.now().date()
        
    def can_trade(self):
        current_date = datetime.now().date()
        if current_date > self.last_reset:
            self.daily_loss = 0
            self.last_reset = current_date
            
        return self.daily_loss < self.daily_loss_limit
        
    def calculate_stake(self, win_rate):
        if win_rate < 40:  # If win rate is too low, reduce stake
            return self.current_balance * 0.01
        return self.current_balance * self.max_risk_per_trade * (win_rate / 100)
        
    def update_balance(self, profit_loss):
        self.current_balance += profit_loss
        if profit_loss < 0:
            self.daily_loss -= profit_loss

class MLModel:
    def __init__(self, strategy_name):
        self.strategy_name = strategy_name
        self.model = RandomForestClassifier(n_estimators=100)
        self.features_history = deque(maxlen=1000)
        self.labels_history = deque(maxlen=1000)
        self.min_samples_train = 50
        
    def prepare_features(self, digits):
        features = []
        features.extend([
            np.mean(digits),
            np.std(digits),
            sum(1 for d in digits if d % 2 == 0) / len(digits),
            max(digits),
            min(digits)
        ])
        return features
        
    def add_training_data(self, features, won):
        self.features_history.append(features)
        self.labels_history.append(1 if won else 0)
        
    def should_trade(self, features):
        if len(self.features_history) < self.min_samples_train:
            return True
            
        self.model.fit(
            list(self.features_history),
            list(self.labels_history)
        )
        prediction = self.model.predict_proba([features])[0]
        return prediction[1] > 0.6  # Only trade if 60% confidence of winning

class MarketAnalyzer:
    def __init__(self):
        self.markets = [
            "R_10",     # Volatility 10 Index
            "R_10S",    # Volatility 10 (1s) Index
            "R_25",     # Volatility 25 Index
            "R_25S",    # Volatility 25 (1s) Index
            "R_50",     # Volatility 50 Index
            "R_50S",    # Volatility 50 (1s) Index
            "R_75",     # Volatility 75 Index
            "R_75S",    # Volatility 75 (1s) Index
            "R_100",    # Volatility 100 Index
            "R_100S"    # Volatility 100 (1s) Index
        ]
        self.market_metrics = {}
        self.min_trades_threshold = int(os.getenv("MIN_TRADES_THRESHOLD", "10"))
        self.analysis_window = timedelta(hours=int(os.getenv("ANALYSIS_WINDOW_HOURS", "4")))
        self.correlation_threshold = float(os.getenv("CORRELATION_THRESHOLD", "0.7"))
        self.market_data = {market: pd.DataFrame() for market in self.markets}
        
    def update_market_metrics(self, market, won, profit):
        if market not in self.market_metrics:
            self.market_metrics[market] = {
                'trades': [],
                'wins': 0,
                'total_trades': 0,
                'total_profit': 0
            }
            
        self.market_metrics[market]['trades'].append({
            'timestamp': datetime.now(),
            'won': won,
            'profit': profit
        })
        
        if won:
            self.market_metrics[market]['wins'] += 1
        self.market_metrics[market]['total_trades'] += 1
        self.market_metrics[market]['total_profit'] += profit
        
        # Remove old trades
        cutoff_time = datetime.now() - self.analysis_window
        self.market_metrics[market]['trades'] = [
            trade for trade in self.market_metrics[market]['trades']
            if trade['timestamp'] > cutoff_time
        ]
        
    def get_best_markets(self, top_n=3):
        current_metrics = {}
        for market, data in self.market_metrics.items():
            recent_trades = [t for t in data['trades'] 
                           if t['timestamp'] > datetime.now() - self.analysis_window]
            
            if len(recent_trades) >= self.min_trades_threshold:
                wins = sum(1 for t in recent_trades if t['won'])
                total = len(recent_trades)
                win_rate = (wins / total) * 100
                profit = sum(t['profit'] for t in recent_trades)
                
                current_metrics[market] = {
                    'win_rate': win_rate,
                    'profit': profit,
                    'trade_count': total
                }
        
        sorted_markets = sorted(
            current_metrics.items(),
            key=lambda x: (x[1]['win_rate'], x[1]['profit']),
            reverse=True
        )
        
        return [market for market, _ in sorted_markets[:top_n]]

    async def analyze_market(self, websocket, market, num_samples=100):
        request = {
            "ticks_history": market,
            "count": num_samples,
            "end": "latest",
            "style": "ticks"
        }
        
        await websocket.send(json.dumps(request))
        response = await websocket.recv()
        history = json.loads(response)
        
        if "error" in history:
            logging.error(f"Error analyzing market {market}: {history['error']['message']}")
            return None
            
        prices = [float(price['quote']) for price in history['history']['prices']]
        volatility = np.std(prices)
        trend = np.mean(np.diff(prices))
        
        return {
            'volatility': volatility,
            'trend': trend,
            'last_price': prices[-1]
        }

    async def analyze_correlations(self, websocket):
        """Analyze correlations between markets"""
        price_data = {}
        
        for market in self.markets:
            history = await self.get_market_history(websocket, market, 100)
            if history:
                price_data[market] = pd.Series(history)
        
        if len(price_data) > 1:
            df = pd.DataFrame(price_data)
            correlations = df.corr()
            
            # Find highly correlated pairs
            correlated_pairs = []
            for i in range(len(correlations.columns)):
                for j in range(i+1, len(correlations.columns)):
                    if abs(correlations.iloc[i, j]) > self.correlation_threshold:
                        correlated_pairs.append((
                            correlations.columns[i],
                            correlations.columns[j],
                            correlations.iloc[i, j]
                        ))
            
            return correlated_pairs
        return []

    async def get_market_history(self, websocket, market, count=100):
        request = {
            "ticks_history": market,
            "count": count,
            "end": "latest",
            "style": "ticks"
        }
        
        await websocket.send(json.dumps(request))
        response = await websocket.recv()
        history = json.loads(response)
        
        if "error" in history:
            return None
            
        return [float(price['quote']) for price in history['history']['prices']]

    def calculate_technical_indicators(self, prices):
        """Calculate technical indicators for market analysis"""
        df = pd.DataFrame(prices, columns=['price'])
        
        # Calculate RSI
        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        exp1 = df['price'].ewm(span=12, adjust=False).mean()
        exp2 = df['price'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # Calculate Bollinger Bands
        df['sma'] = df['price'].rolling(window=20).mean()
        df['std'] = df['price'].rolling(window=20).std()
        df['upper_band'] = df['sma'] + (df['std'] * 2)
        df['lower_band'] = df['sma'] - (df['std'] * 2)
        
        return df

    def get_market_win_rates(self, market, window=None):
        """Get historical win rates for a market"""
        if market not in self.market_metrics:
            return {}
            
        trades = self.market_metrics[market]['trades']
        if not trades:
            return {}
            
        if window is None:
            window = self.analysis_window
            
        current_time = datetime.now()
        win_rates = {}
        
        # Calculate win rates for each hour in the window
        for hour in range(int(window.total_seconds() / 3600)):
            start_time = current_time - timedelta(hours=hour+1)
            end_time = current_time - timedelta(hours=hour)
            
            period_trades = [t for t in trades 
                           if start_time <= t['timestamp'] <= end_time]
            
            if period_trades:
                wins = sum(1 for t in period_trades if t['won'])
                win_rates[end_time] = (wins / len(period_trades)) * 100
        
        return win_rates

    def get_optimal_trading_hours(self, market):
        """Determine the best hours to trade for each market"""
        if market not in self.market_metrics:
            return []
            
        trades = self.market_metrics[market]['trades']
        if not trades:
            return []
            
        # Group trades by hour and calculate win rates
        hourly_performance = {}
        for trade in trades:
            hour = trade['timestamp'].hour
            if hour not in hourly_performance:
                hourly_performance[hour] = {'wins': 0, 'total': 0}
            
            hourly_performance[hour]['total'] += 1
            if trade['won']:
                hourly_performance[hour]['wins'] += 1
        
        # Calculate win rates and find best hours
        best_hours = []
        for hour, stats in hourly_performance.items():
            if stats['total'] >= self.min_trades_threshold:
                win_rate = (stats['wins'] / stats['total']) * 100
                if win_rate > 50:  # Only include hours with >50% win rate
                    best_hours.append((hour, win_rate))
        
        return sorted(best_hours, key=lambda x: x[1], reverse=True)

class DerivTradingBot:
    def __init__(self, api_token, initial_balance=1000.0):
        self.api_url = "wss://ws.binaryws.com/websockets/v3"
        self.api_token = api_token
        self.port = int(os.environ.get("PORT", 8050))
        self.host = os.environ.get("HOST", "0.0.0.0")
        self.debug = os.environ.get("DEBUG", "False").lower() == "true"
        
        # Add memory management
        self.max_history_size = 1000  # Limit history size for free tier memory constraints
        self.cleanup_interval = 3600  # Cleanup every hour
        self.last_cleanup = datetime.now()
        
        # Add keep-alive mechanism
        self.last_ping = datetime.now()
        self.ping_interval = 30  # Send ping every 30 seconds
        
        self.websocket = None
        self.tick_history = {}  # Separate history for each market
        
        # Trading parameters
        self.stake_amount = float(os.getenv("STAKE_AMOUNT", "1.0"))
        self.take_profit = float(os.getenv("TAKE_PROFIT", "19.0"))
        self.active_markets = []
        self.max_active_markets = 3
        
        # Initialize components
        self.metrics = TradingMetrics()
        self.risk_manager = RiskManager(initial_balance)
        self.market_analyzer = MarketAnalyzer()
        
        # Initialize ML models for each market
        self.ml_models = {}
        self.initialize_ml_models()
        self.last_report_time = datetime.now()
        
        # Add new parameters from environment
        self.max_correlation = float(os.getenv("MAX_CORRELATION", "0.7"))
        self.min_volatility = float(os.getenv("MIN_VOLATILITY", "0.001"))
        self.max_volatility = float(os.getenv("MAX_VOLATILITY", "0.05"))
        self.min_win_rate = float(os.getenv("MIN_WIN_RATE", "55.0"))
        
        # Initialize dashboard
        self.dashboard = WebDashboard(self)
        self.dashboard_thread = threading.Thread(target=self.run_dashboard)
        self.dashboard_thread.daemon = True
        
        self.virtual_trader = VirtualTrader(initial_balance)
        self.backtesting_mode = False
        
    def initialize_ml_models(self):
        strategies = ['even_odd', 'differs', 'matches', 'over_under', 'higher_lower']
        for market in self.market_analyzer.markets:
            self.ml_models[market] = {
                strategy: MLModel(f"{market}_{strategy}")
                for strategy in strategies
            }
        self.load_ml_models()

    async def analyze_all_markets(self):
        market_analysis = {}
        for market in self.market_analyzer.markets:
            analysis = await self.market_analyzer.analyze_market(self.websocket, market)
            if analysis:
                market_analysis[market] = analysis
                logging.info(f"Market {market} analysis: {analysis}")
        
        best_markets = self.market_analyzer.get_best_markets()
        if best_markets:
            self.active_markets = best_markets
            logging.info(f"Selected markets for trading: {best_markets}")
        else:
            self.active_markets = self.market_analyzer.markets[:self.max_active_markets]
        
        return market_analysis

    async def start(self):
        while True:
            try:
                if await self.connect():
                    if await self.authenticate():
                        logging.info("Bot is ready to trade!")
                        
                        while True:
                            try:
                                # Memory management
                                if (datetime.now() - self.last_cleanup).seconds > self.cleanup_interval:
                                    self.cleanup_old_data()
                                
                                # Keep-alive mechanism
                                if (datetime.now() - self.last_ping).seconds > self.ping_interval:
                                    await self.send_ping()
                                
                                # Regular bot operations
                                await self.run_trading_cycle()
                                
                            except Exception as e:
                                logging.error(f"Error in trading cycle: {e}")
                                await asyncio.sleep(5)
                                
            except Exception as e:
                logging.error(f"Connection error: {e}")
                await asyncio.sleep(30)  # Wait before reconnecting
    
    def cleanup_old_data(self):
        """Clean up old data to manage memory usage"""
        for market in self.market_analyzer.market_metrics:
            if len(self.market_analyzer.market_metrics[market]['trades']) > self.max_history_size:
                self.market_analyzer.market_metrics[market]['trades'] = \
                    self.market_analyzer.market_metrics[market]['trades'][-self.max_history_size:]
        
        self.last_cleanup = datetime.now()
        logging.info("Performed data cleanup")
    
    async def send_ping(self):
        """Send ping to keep connection alive"""
        try:
            ping_request = {"ping": 1}
            await self.websocket.send(json.dumps(ping_request))
            self.last_ping = datetime.now()
        except Exception as e:
            logging.error(f"Error sending ping: {e}")
    
    async def run_trading_cycle(self):
        """Run one cycle of trading operations"""
        # Analyze markets and correlations
        market_analysis = await self.analyze_all_markets()
        correlations = await self.market_analyzer.analyze_correlations(self.websocket)
        
        # Filter out highly correlated markets
        filtered_markets = self.filter_correlated_markets(
            self.active_markets, correlations
        )
        
        # Update active markets based on performance and correlations
        self.active_markets = [
            market for market in filtered_markets
            if self.should_trade_market(market, market_analysis)
        ]
        
        # Trade only during optimal hours for each market
        for market in self.active_markets:
            optimal_hours = self.market_analyzer.get_optimal_trading_hours(market)
            current_hour = datetime.now().hour
            
            if any(hour[0] == current_hour for hour in optimal_hours):
                await self.trade_market(market)
        
        # Regular reporting and model saving
        if (datetime.now() - self.last_report_time).seconds > 3600:
            self.print_performance_report()
            self.save_ml_models()
            self.last_report_time = datetime.now()

    def filter_correlated_markets(self, markets, correlations):
        """Filter out highly correlated markets, keeping the better performing one"""
        if not correlations:
            return markets
            
        filtered = set(markets)
        for market1, market2, corr in correlations:
            if market1 in filtered and market2 in filtered:
                # Keep the better performing market
                m1_metrics = self.market_analyzer.market_metrics.get(market1, {})
                m2_metrics = self.market_analyzer.market_metrics.get(market2, {})
                
                m1_profit = sum(t['profit'] for t in m1_metrics.get('trades', []))
                m2_profit = sum(t['profit'] for t in m2_metrics.get('trades', []))
                
                if m1_profit > m2_profit:
                    filtered.remove(market2)
                else:
                    filtered.remove(market1)
        
        return list(filtered)

    def should_trade_market(self, market, market_analysis):
        """Determine if a market should be traded based on analysis"""
        if market not in market_analysis:
            return False
            
        analysis = market_analysis[market]
        metrics = self.market_analyzer.market_metrics.get(market, {})
        
        # Check volatility
        if not (self.min_volatility <= analysis['volatility'] <= self.max_volatility):
            return False
            
        # Check win rate
        recent_trades = [t for t in metrics.get('trades', [])
                        if t['timestamp'] > datetime.now() - self.market_analyzer.analysis_window]
        
        if recent_trades:
            win_rate = (sum(1 for t in recent_trades if t['won']) / len(recent_trades)) * 100
            if win_rate < self.min_win_rate:
                return False
        
        return True

    async def trade_market(self, market):
        """Execute trades for a specific market"""
        strategies = [
            self.analyze_even_odd,
            self.analyze_differs,
            self.analyze_matches,
            self.analyze_over_under,
            self.analyze_higher_lower
        ]
        
        for strategy in strategies:
            result = await strategy(market)
            if result:
                if "error" in result:
                    logging.error(f"Trade error on {market}: {result['error']['message']}")
                else:
                    logging.info(f"Trade placed on {market}: {result}")
            await asyncio.sleep(5)

    async def get_last_digit_stats(self, market, count=10):
        request = {
            "ticks_history": market,
            "count": count,
            "end": "latest",
            "style": "ticks"
        }
        await self.websocket.send(json.dumps(request))
        response = await self.websocket.recv()
        history = json.loads(response)
        return [int(str(tick['quote'])[-1]) for tick in history['history']['prices']]

    async def place_trade(self, market, contract_type, stake, parameters):
        """Place either a real or virtual trade based on mode"""
        if self.backtesting_mode:
            trade_id = self.virtual_trader.place_virtual_trade(
                market, contract_type, stake, parameters
            )
            logging.info(f"Placed virtual trade {trade_id} on {market}")
            return trade_id
        else:
            # Original live trading logic
            return self.api.buy(parameters)

    async def analyze_even_odd(self, market):
        digits = await self.get_last_digit_stats(market, 10)
        features = self.ml_models[market]['even_odd'].prepare_features(digits)
        
        if not self.ml_models[market]['even_odd'].should_trade(features):
            return None
            
        even_count = sum(1 for d in digits if d % 2 == 0)
        odd_count = len(digits) - even_count
        
        prediction = "EVEN" if odd_count > even_count else "ODD"
        
        parameters = {
            "contract_type": f"DIGIT_{prediction}",
            "duration": 1,
            "duration_unit": "t"
        }
        return await self.place_trade(market, "digit", self.stake_amount, parameters)

    async def analyze_differs(self, market):
        digits = await self.get_last_digit_stats(market, 10)
        features = self.ml_models[market]['differs'].prepare_features(digits)
        
        if not self.ml_models[market]['differs'].should_trade(features):
            return None
            
        digit_counts = Counter(digits)
        least_common = digit_counts.most_common()[-1][0]
        
        parameters = {
            "contract_type": "DIGIT_DIFF",
            "symbol": market,
            "duration": 1,
            "duration_unit": "t",
            "barrier": least_common
        }
        return await self.place_trade(market, "digit", self.stake_amount, parameters)

    async def analyze_matches(self, market):
        digits = await self.get_last_digit_stats(market, 5)
        features = self.ml_models[market]['matches'].prepare_features(digits)
        
        if not self.ml_models[market]['matches'].should_trade(features):
            return None
            
        if len(set(digits[-2:])) == 1:
            prediction = digits[-1]
            
            parameters = {
                "contract_type": "DIGIT_MATCH",
                "symbol": market,
                "duration": 1,
                "duration_unit": "t",
                "barrier": prediction
            }
            return await self.place_trade(market, "digit", self.stake_amount, parameters)
        return None

    async def analyze_over_under(self, market):
        digits = await self.get_last_digit_stats(market, 10)
        features = self.ml_models[market]['over_under'].prepare_features(digits)
        
        if not self.ml_models[market]['over_under'].should_trade(features):
            return None
            
        avg_digit = sum(digits) / len(digits)
        
        if avg_digit < 4.5:
            contract_type = "DIGIT_OVER"
            barrier = 5
        else:
            contract_type = "DIGIT_UNDER"
            barrier = 4
            
        parameters = {
            "contract_type": contract_type,
            "symbol": market,
            "duration": 1,
            "duration_unit": "t",
            "barrier": barrier
        }
        return await self.place_trade(market, "digit", self.stake_amount, parameters)

    async def analyze_higher_lower(self, market):
        digits = await self.get_last_digit_stats(market, 5)
        features = self.ml_models[market]['higher_lower'].prepare_features(digits)
        
        if not self.ml_models[market]['higher_lower'].should_trade(features):
            return None
            
        request = {
            "proposal": 1,
            "subscribe": 1,
            "amount": self.stake_amount,
            "basis": "stake",
            "contract_type": "CALL",
            "currency": "USD",
            "symbol": market,
            "duration": 5,
            "duration_unit": "m"
        }
        
        await self.websocket.send(json.dumps(request))
        response = await self.websocket.recv()
        data = json.loads(response)
        
        if "error" in data:
            logging.error(f"Error getting proposal: {data['error']['message']}")
            return None
            
        current_spot = data["proposal"]["spot"]
        
        # Calculate barriers for take profit percentage
        higher_barrier = current_spot * (1 + self.take_profit/100)
        lower_barrier = current_spot * (1 - self.take_profit/100)
        
        trend_up = sum(digits[i] < digits[i+1] for i in range(len(digits)-1)) > len(digits)/2
        
        parameters = {
            "contract_type": "CALL" if trend_up else "PUT",
            "symbol": market,
            "duration": 5,
            "duration_unit": "m",
            "barrier": higher_barrier if trend_up else lower_barrier
        }
        return await self.place_trade(market, "vanilla", self.stake_amount, parameters)

    def print_performance_report(self):
        logging.info("\n=== Performance Report ===")
        logging.info(f"Total Trades: {self.metrics.total_trades}")
        logging.info(f"Overall Win Rate: {self.metrics.get_win_rate():.2f}%")
        logging.info(f"Total Profit/Loss: {self.metrics.profit_loss:.2f}")
        logging.info(f"Current Balance: {self.risk_manager.current_balance:.2f}")
        
        logging.info("\nMarket Performance:")
        for market in self.active_markets:
            metrics = self.market_analyzer.market_metrics.get(market, {})
            if metrics:
                recent_trades = [t for t in metrics['trades'] 
                               if t['timestamp'] > datetime.now() - self.market_analyzer.analysis_window]
                if recent_trades:
                    wins = sum(1 for t in recent_trades if t['won'])
                    total = len(recent_trades)
                    win_rate = (wins / total) * 100
                    profit = sum(t['profit'] for t in recent_trades)
                    
                    logging.info(f"\n{market}:")
                    logging.info(f"  Win Rate: {win_rate:.2f}%")
                    logging.info(f"  Profit: {profit:.2f}")
                    logging.info(f"  Recent Trades: {total}")

    async def connect(self):
        try:
            self.websocket = await websockets.connect(self.api_url)
            logging.info("Connected to Deriv WebSocket API")
            return True
        except Exception as e:
            logging.error(f"Error connecting to WebSocket: {e}")
            return False

    async def authenticate(self):
        if not self.api_token:
            logging.error("API token not found. Please set DERIV_API_TOKEN in .env file")
            return False
            
        auth_request = {
            "authorize": self.api_token,
        }
        
        try:
            await self.websocket.send(json.dumps(auth_request))
            response = await self.websocket.recv()
            auth_response = json.loads(response)
            
            if "error" in auth_response:
                logging.error(f"Authentication error: {auth_response['error']['message']}")
                return False
                
            self.risk_manager.current_balance = float(auth_response['authorize']['balance'])
            logging.info(f"Successfully authenticated with Deriv API")
            logging.info(f"Account balance: {auth_response['authorize']['balance']} {auth_response['authorize']['currency']}")
            return True
            
        except Exception as e:
            logging.error(f"Error during authentication: {e}")
            return False

    async def monitor_trade(self, contract_id):
        while True:
            try:
                response = await self.websocket.recv()
                data = json.loads(response)
                
                if "proposal_open_contract" in data:
                    contract = data["proposal_open_contract"]
                    
                    if contract["is_sold"]:
                        return float(contract["profit"])
                        
            except Exception as e:
                logging.error(f"Error monitoring trade: {e}")
                return 0

    def load_ml_models(self):
        for market, strategies in self.ml_models.items():
            for strategy, model in strategies.items():
                model_path = f"models/{market}_{strategy}_model.pkl"
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        model.model = pickle.load(f)
                    logging.info(f"Loaded saved model for {market}_{strategy}")
                
    def save_ml_models(self):
        os.makedirs("models", exist_ok=True)
        for market, strategies in self.ml_models.items():
            for strategy, model in strategies.items():
                with open(f"models/{market}_{strategy}_model.pkl", 'wb') as f:
                    pickle.dump(model.model, f)
        logging.info("Saved ML models")

    def toggle_backtesting_mode(self, enabled=True):
        """Toggle between live and backtesting mode"""
        self.backtesting_mode = enabled
        if enabled:
            logging.info("Switched to backtesting mode")
        else:
            logging.info("Switched to live trading mode")
            
    def update_virtual_trades(self, current_prices):
        """Update virtual trades with current market prices"""
        if not self.backtesting_mode:
            return
            
        current_time = datetime.now()
        for market, price in current_prices.items():
            for trade_index in range(len(self.virtual_trader.virtual_trades)):
                result = self.virtual_trader.update_virtual_trade(
                    trade_index, price, current_time
                )
                if result:
                    logging.info(f"Virtual trade {trade_index} closed with outcome: {result}")
                    
    def get_backtesting_report(self):
        """Get detailed backtesting performance report"""
        if not self.backtesting_mode:
            return None
        return self.virtual_trader.get_performance_report()
        
    def run_backtest(self, market_data, strategy_params):
        """Run backtest on historical market data"""
        self.toggle_backtesting_mode(True)
        self.virtual_trader.reset()
        
        for timestamp, data in market_data.items():
            # Update market analyzer with historical data
            self.market_analyzer.update_market_metrics(data)
            
            # Generate trading signals based on strategy
            signals = self.generate_trading_signals(data, strategy_params)
            
            # Place virtual trades based on signals
            for signal in signals:
                self.place_trade(
                    signal['market'],
                    signal['contract_type'],
                    signal['stake'],
                    signal['parameters']
                )
                
            # Update virtual trades with current prices
            self.update_virtual_trades(data['prices'])
            
        # Generate and return backtest report
        report = self.get_backtesting_report()
        self.toggle_backtesting_mode(False)
        return report
        
    def generate_trading_signals(self, market_data, strategy_params):
        """Generate trading signals based on market data and strategy parameters"""
        signals = []
        
        for market, data in market_data.items():
            # Apply technical analysis
            analysis = self.technical_analyzer.analyze(data['prices'])
            
            # Check for trading opportunities based on strategy
            if self.should_enter_trade(analysis, strategy_params):
                signal = {
                    'market': market,
                    'contract_type': self.determine_contract_type(analysis),
                    'stake': self.calculate_position_size(market),
                    'parameters': self.generate_trade_parameters(analysis, market)
                }
                signals.append(signal)
                
        return signals
        
    def should_enter_trade(self, analysis, strategy_params):
        """Determine if we should enter a trade based on technical analysis"""
        # Example strategy logic using multiple indicators
        rsi = analysis.get('rsi', 50)
        macd = analysis.get('macd', 0)
        bb_position = analysis.get('bb_position', 0)
        
        # RSI conditions
        rsi_oversold = rsi < strategy_params.get('rsi_oversold', 30)
        rsi_overbought = rsi > strategy_params.get('rsi_overbought', 70)
        
        # MACD conditions
        macd_bullish = macd > strategy_params.get('macd_threshold', 0)
        
        # Bollinger Bands conditions
        bb_buy = bb_position < strategy_params.get('bb_lower_threshold', -0.8)
        bb_sell = bb_position > strategy_params.get('bb_upper_threshold', 0.8)
        
        # Combined signals
        buy_signal = rsi_oversold and macd_bullish and bb_buy
        sell_signal = rsi_overbought and not macd_bullish and bb_sell
        
        return buy_signal or sell_signal
        
    def determine_contract_type(self, analysis):
        """Determine the best contract type based on analysis"""
        trend = analysis.get('trend', 'neutral')
        volatility = analysis.get('volatility', 'medium')
        
        if trend == 'strong_uptrend':
            return 'CALL'
        elif trend == 'strong_downtrend':
            return 'PUT'
        elif volatility == 'high':
            # Use digit trades in high volatility
            return 'DIGIT_EVEN' if random.random() > 0.5 else 'DIGIT_ODD'
        else:
            # Default to rise/fall
            return 'CALL' if random.random() > 0.5 else 'PUT'
            
    def generate_trade_parameters(self, analysis, market):
        """Generate trade parameters based on technical analysis"""
        current_price = analysis.get('current_price', 0)
        volatility = analysis.get('volatility', 1.0)
        
        # Adjust take profit and stop loss based on volatility
        take_profit_pips = self.base_take_profit * volatility
        stop_loss_pips = self.base_stop_loss * volatility
        
        parameters = {
            'market': market,
            'entry_price': current_price,
            'take_profit': current_price + take_profit_pips,
            'stop_loss': current_price - stop_loss_pips,
            'duration': self.calculate_optimal_duration(analysis)
        }
        
        return parameters
        
    def calculate_optimal_duration(self, analysis):
        """Calculate optimal trade duration based on market conditions"""
        volatility = analysis.get('volatility', 1.0)
        trend_strength = analysis.get('trend_strength', 0.5)
        
        # Base duration in seconds
        base_duration = 60  # 1 minute
        
        # Adjust duration based on market conditions
        if volatility > 1.5:  # High volatility
            duration = base_duration * 0.5  # Shorter duration
        elif trend_strength > 0.7:  # Strong trend
            duration = base_duration * 2  # Longer duration
        else:
            duration = base_duration
            
        return max(30, min(300, duration))  # Keep between 30s and 5min

    def run_dashboard(self):
        """Run the dashboard on Heroku"""
        self.dashboard.run(host='0.0.0.0', port=self.port)

async def main():
    # Load configuration from environment variables
    api_token = os.environ.get("DERIV_API_TOKEN")
    initial_balance = float(os.environ.get("INITIAL_BALANCE", "1000.0"))
    
    if not api_token:
        raise ValueError("DERIV_API_TOKEN environment variable is required")
    
    bot = DerivTradingBot(api_token, initial_balance)
    
    # Start the dashboard in a separate thread
    import threading
    dashboard_thread = threading.Thread(target=bot.run_dashboard)
    dashboard_thread.daemon = True
    dashboard_thread.start()
    
    # Start the trading bot
    await bot.start()

if __name__ == "__main__":
    # Create a .env file if it doesn't exist and we're not on Heroku
    if not os.environ.get("DYNO") and not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write("""DERIV_API_TOKEN=your_api_token_here
STAKE_AMOUNT=1.0
TAKE_PROFIT=19.0
INITIAL_BALANCE=1000.0
TRADING_SYMBOL=R_100
MIN_TRADES_THRESHOLD=10
ANALYSIS_WINDOW_HOURS=4
CORRELATION_THRESHOLD=0.7
MAX_CORRELATION=0.7
MIN_VOLATILITY=0.001
MAX_VOLATILITY=0.05
MIN_WIN_RATE=55.0
""")
        logging.info("Created .env file. Please fill in your API token and trading parameters.")
    
    # Load environment variables
    load_dotenv()
    
    # Run the application
    asyncio.run(main()) 