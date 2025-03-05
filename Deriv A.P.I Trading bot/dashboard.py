import dash
from dash import html, dcc
import plotly.graph_objs as go
from dash.dependencies import Input, Output
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
from plotly.subplots import make_subplots

class WebDashboard:
    def __init__(self, bot):
        self.bot = bot
        self.app = dash.Dash(__name__)
        self.server = self.app.server  # Add this line for Heroku
        self.setup_layout()
        
    def setup_layout(self):
        self.app.layout = html.Div([
            html.H1("Deriv Trading Bot Dashboard", style={'textAlign': 'center', 'color': '#2c3e50'}),
            
            # Mode Toggle
            html.Div([
                html.H3("Trading Mode", style={'color': '#2c3e50'}),
                dcc.RadioItems(
                    id='mode-toggle',
                    options=[
                        {'label': 'Live Trading', 'value': 'live'},
                        {'label': 'Backtesting', 'value': 'backtest'}
                    ],
                    value='live',
                    style={'margin': '10px'}
                )
            ], style={'margin': '20px', 'padding': '20px', 'backgroundColor': '#ecf0f1', 'borderRadius': '5px'}),
            
            # Performance Metrics
            html.Div([
                html.H3("Performance Metrics", style={'color': '#2c3e50'}),
                html.Div(id='performance-metrics', style={'display': 'flex', 'justifyContent': 'space-around'})
            ], style={'margin': '20px', 'padding': '20px', 'backgroundColor': '#ecf0f1', 'borderRadius': '5px'}),
            
            # Active Markets
            html.Div([
                html.H3("Active Markets", style={'color': '#2c3e50'}),
                html.Div(id='active-markets')
            ], style={'margin': '20px', 'padding': '20px', 'backgroundColor': '#ecf0f1', 'borderRadius': '5px'}),
            
            # Trade History
            html.Div([
                html.H3("Trade History", style={'color': '#2c3e50'}),
                dcc.Graph(id='trade-history-graph')
            ], style={'margin': '20px', 'padding': '20px', 'backgroundColor': '#ecf0f1', 'borderRadius': '5px'}),
            
            # Market Analysis
            html.Div([
                html.H3("Market Analysis", style={'color': '#2c3e50'}),
                dcc.Dropdown(
                    id='market-selector',
                    style={'margin': '10px'}
                ),
                dcc.Graph(id='market-analysis-graph')
            ], style={'margin': '20px', 'padding': '20px', 'backgroundColor': '#ecf0f1', 'borderRadius': '5px'}),
            
            # Strategy Performance
            html.Div([
                html.H3("Strategy Performance", style={'color': '#2c3e50'}),
                html.Div(id='strategy-performance')
            ], style={'margin': '20px', 'padding': '20px', 'backgroundColor': '#ecf0f1', 'borderRadius': '5px'}),
            
            # Backtesting Controls (visible only in backtest mode)
            html.Div([
                html.H3("Backtesting Controls", style={'color': '#2c3e50'}),
                html.Div([
                    html.Label("Date Range"),
                    dcc.DatePickerRange(
                        id='backtest-date-range',
                        start_date=datetime.now() - timedelta(days=30),
                        end_date=datetime.now(),
                        style={'margin': '10px'}
                    ),
                    html.Label("Strategy Parameters"),
                    dcc.Input(
                        id='rsi-oversold',
                        type='number',
                        placeholder='RSI Oversold',
                        value=30,
                        style={'margin': '5px'}
                    ),
                    dcc.Input(
                        id='rsi-overbought',
                        type='number',
                        placeholder='RSI Overbought',
                        value=70,
                        style={'margin': '5px'}
                    ),
                    html.Button(
                        'Run Backtest',
                        id='run-backtest-button',
                        style={'margin': '10px', 'backgroundColor': '#3498db', 'color': 'white'}
                    )
                ])
            ], id='backtest-controls', style={'margin': '20px', 'padding': '20px', 'backgroundColor': '#ecf0f1', 'borderRadius': '5px'}),
            
            dcc.Interval(
                id='interval-component',
                interval=5*1000,  # Update every 5 seconds
                n_intervals=0
            )
        ])
        
        self.setup_callbacks()
        
    def setup_callbacks(self):
        @self.app.callback(
            [Output('performance-metrics', 'children'),
             Output('active-markets', 'children'),
             Output('trade-history-graph', 'figure'),
             Output('market-analysis-graph', 'figure'),
             Output('strategy-performance', 'children'),
             Output('backtest-controls', 'style')],
            [Input('interval-component', 'n_intervals'),
             Input('mode-toggle', 'value'),
             Input('market-selector', 'value')]
        )
        def update_dashboard(n, mode, selected_market):
            # Get performance metrics
            if mode == 'live':
                metrics = self.bot.metrics.get_overall_metrics()
                backtest_style = {'display': 'none'}
            else:
                metrics = self.bot.virtual_trader.get_performance_report()
                backtest_style = {'margin': '20px', 'padding': '20px', 'backgroundColor': '#ecf0f1', 'borderRadius': '5px'}
                
            # Create metrics cards
            metrics_cards = [
                self.create_metric_card("Win Rate", f"{metrics['win_rate']:.2f}%"),
                self.create_metric_card("Profit Factor", f"{metrics['profit_factor']:.2f}"),
                self.create_metric_card("Net Profit", f"${metrics['net_profit']:.2f}"),
                self.create_metric_card("Max Drawdown", f"{metrics['max_drawdown']:.2f}%")
            ]
            
            # Get active markets
            active_markets = self.bot.market_analyzer.get_active_markets()
            markets_table = self.create_markets_table(active_markets)
            
            # Create trade history graph
            trade_history = self.create_trade_history_graph(mode)
            
            # Create market analysis graph
            market_analysis = self.create_market_analysis_graph(selected_market)
            
            # Create strategy performance table
            strategy_table = self.create_strategy_table(mode)
            
            return metrics_cards, markets_table, trade_history, market_analysis, strategy_table, backtest_style
            
        @self.app.callback(
            Output('market-selector', 'options'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_market_selector(n):
            markets = self.bot.market_analyzer.get_active_markets()
            return [{'label': market, 'value': market} for market in markets]
            
        @self.app.callback(
            Output('trade-history-graph', 'figure'),
            [Input('run-backtest-button', 'n_clicks')],
            [dash.dependencies.State('backtest-date-range', 'start_date'),
             dash.dependencies.State('backtest-date-range', 'end_date'),
             dash.dependencies.State('rsi-oversold', 'value'),
             dash.dependencies.State('rsi-overbought', 'value')]
        )
        def run_backtest(n_clicks, start_date, end_date, rsi_oversold, rsi_overbought):
            if n_clicks is None:
                return dash.no_update
                
            # Prepare strategy parameters
            strategy_params = {
                'rsi_oversold': rsi_oversold,
                'rsi_overbought': rsi_overbought
            }
            
            # Get historical data for backtesting
            market_data = self.bot.get_historical_data(start_date, end_date)
            
            # Run backtest
            results = self.bot.run_backtest(market_data, strategy_params)
            
            # Create and return updated trade history graph
            return self.create_trade_history_graph('backtest')
            
    def create_metric_card(self, title, value):
        return html.Div([
            html.H4(title, style={'textAlign': 'center', 'color': '#2c3e50'}),
            html.H2(value, style={'textAlign': 'center', 'color': '#27ae60'})
        ], style={'flex': '1', 'margin': '10px', 'padding': '20px', 'backgroundColor': 'white', 'borderRadius': '5px'})
        
    def create_markets_table(self, markets):
        return html.Table([
            html.Thead([
                html.Tr([
                    html.Th("Market"),
                    html.Th("Win Rate"),
                    html.Th("Profit"),
                    html.Th("Trades"),
                    html.Th("Status")
                ])
            ]),
            html.Tbody([
                html.Tr([
                    html.Td(market['symbol']),
                    html.Td(f"{market['win_rate']:.2f}%"),
                    html.Td(f"${market['profit']:.2f}"),
                    html.Td(str(market['trades'])),
                    html.Td(market['status'])
                ]) for market in markets
            ])
        ], style={'width': '100%', 'textAlign': 'center'})
        
    def create_trade_history_graph(self, mode):
        if mode == 'live':
            trades = self.bot.metrics.get_trade_history()
        else:
            trades = self.bot.virtual_trader.trade_history
            
        df = pd.DataFrame(trades)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['cumulative_profit'],
            mode='lines+markers',
            name='Cumulative Profit'
        ))
        
        fig.update_layout(
            title='Trade History',
            xaxis_title='Time',
            yaxis_title='Cumulative Profit ($)',
            template='plotly_white'
        )
        
        return fig
        
    def create_market_analysis_graph(self, market):
        if not market:
            return go.Figure()
            
        analysis = self.bot.technical_analyzer.analyze(
            self.bot.get_market_data(market)
        )
        
        df = pd.DataFrame(analysis)
        
        fig = make_subplots(rows=3, cols=1)
        
        # Price and Moving Averages
        fig.add_trace(
            go.Scatter(x=df.index, y=df['price'], name='Price'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['sma'], name='SMA'),
            row=1, col=1
        )
        
        # RSI
        fig.add_trace(
            go.Scatter(x=df.index, y=df['rsi'], name='RSI'),
            row=2, col=1
        )
        
        # MACD
        fig.add_trace(
            go.Scatter(x=df.index, y=df['macd'], name='MACD'),
            row=3, col=1
        )
        
        fig.update_layout(height=800, title=f'Market Analysis - {market}')
        return fig
        
    def create_strategy_table(self, mode):
        if mode == 'live':
            strategies = self.bot.metrics.get_strategy_performance()
        else:
            strategies = self.bot.virtual_trader.get_strategy_performance()
            
        return html.Table([
            html.Thead([
                html.Tr([
                    html.Th("Strategy"),
                    html.Th("Win Rate"),
                    html.Th("Profit"),
                    html.Th("Trades"),
                    html.Th("Avg Duration")
                ])
            ]),
            html.Tbody([
                html.Tr([
                    html.Td(strategy['name']),
                    html.Td(f"{strategy['win_rate']:.2f}%"),
                    html.Td(f"${strategy['profit']:.2f}"),
                    html.Td(str(strategy['trades'])),
                    html.Td(f"{strategy['avg_duration']:.1f}s")
                ]) for strategy in strategies
            ])
        ], style={'width': '100%', 'textAlign': 'center'})
        
    def run(self, debug=False, port=8050, host='0.0.0.0'):
        """Run the dashboard server"""
        self.app.run_server(debug=debug, port=port, host=host) 