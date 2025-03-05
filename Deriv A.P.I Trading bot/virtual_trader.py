import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

class VirtualTrader:
    def __init__(self, initial_balance=1000.0):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions = []
        self.trade_history = []
        self.virtual_trades = []
        self.win_rate = 0
        self.profit_factor = 0
        self.max_drawdown = 0
        
    def place_virtual_trade(self, market, contract_type, stake, parameters, timestamp=None):
        """Place a virtual trade for backtesting"""
        if timestamp is None:
            timestamp = datetime.now()
            
        trade = {
            'market': market,
            'contract_type': contract_type,
            'stake': stake,
            'parameters': parameters,
            'entry_time': timestamp,
            'status': 'open',
            'entry_price': parameters.get('entry_price', 0),
            'take_profit': parameters.get('take_profit', 0),
            'stop_loss': parameters.get('stop_loss', 0)
        }
        
        self.virtual_trades.append(trade)
        return len(self.virtual_trades) - 1  # Return trade index
        
    def update_virtual_trade(self, trade_index, current_price, current_time):
        """Update virtual trade status based on price movement"""
        if trade_index >= len(self.virtual_trades):
            return None
            
        trade = self.virtual_trades[trade_index]
        if trade['status'] != 'open':
            return None
            
        result = self.check_trade_outcome(trade, current_price)
        if result:
            trade['status'] = 'closed'
            trade['exit_time'] = current_time
            trade['exit_price'] = current_price
            trade['profit'] = result['profit']
            trade['outcome'] = result['outcome']
            
            self.balance += result['profit']
            self.trade_history.append(trade)
            
            self.update_metrics()
            
            return result
            
        return None
        
    def check_trade_outcome(self, trade, current_price):
        """Check if trade has hit take profit or stop loss"""
        if trade['contract_type'] in ['DIGIT_EVEN', 'DIGIT_ODD', 'DIGIT_MATCH', 'DIGIT_DIFF']:
            # For digit trades, use the last digit
            current_digit = int(str(current_price)[-1])
            if trade['contract_type'] == 'DIGIT_EVEN':
                won = current_digit % 2 == 0
            elif trade['contract_type'] == 'DIGIT_ODD':
                won = current_digit % 2 != 0
            elif trade['contract_type'] == 'DIGIT_MATCH':
                won = current_digit == trade['parameters']['barrier']
            elif trade['contract_type'] == 'DIGIT_DIFF':
                won = current_digit != trade['parameters']['barrier']
                
            profit = trade['stake'] * (1.95 if won else -1)
            return {'profit': profit, 'outcome': 'win' if won else 'loss'}
            
        elif trade['contract_type'] in ['CALL', 'PUT']:
            # For rise/fall trades
            if trade['contract_type'] == 'CALL':
                if current_price >= trade['take_profit']:
                    return {'profit': trade['stake'] * 0.95, 'outcome': 'win'}
                elif current_price <= trade['stop_loss']:
                    return {'profit': -trade['stake'], 'outcome': 'loss'}
            else:  # PUT
                if current_price <= trade['take_profit']:
                    return {'profit': trade['stake'] * 0.95, 'outcome': 'win'}
                elif current_price >= trade['stop_loss']:
                    return {'profit': -trade['stake'], 'outcome': 'loss'}
                    
        return None
        
    def update_metrics(self):
        """Update performance metrics"""
        if not self.trade_history:
            return
            
        # Calculate win rate
        wins = sum(1 for trade in self.trade_history if trade['outcome'] == 'win')
        self.win_rate = (wins / len(self.trade_history)) * 100
        
        # Calculate profit factor
        gross_profit = sum(trade['profit'] for trade in self.trade_history if trade['profit'] > 0)
        gross_loss = abs(sum(trade['profit'] for trade in self.trade_history if trade['profit'] < 0))
        self.profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
        
        # Calculate maximum drawdown
        balance_curve = [self.initial_balance]
        for trade in self.trade_history:
            balance_curve.append(balance_curve[-1] + trade['profit'])
        
        peak = balance_curve[0]
        max_dd = 0
        
        for balance in balance_curve:
            if balance > peak:
                peak = balance
            dd = (peak - balance) / peak * 100
            max_dd = max(max_dd, dd)
        
        self.max_drawdown = max_dd
        
    def get_performance_report(self):
        """Generate detailed performance report"""
        return {
            'total_trades': len(self.trade_history),
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'max_drawdown': self.max_drawdown,
            'net_profit': self.balance - self.initial_balance,
            'roi': ((self.balance - self.initial_balance) / self.initial_balance) * 100,
            'average_profit': sum(t['profit'] for t in self.trade_history) / len(self.trade_history) if self.trade_history else 0,
            'largest_win': max((t['profit'] for t in self.trade_history), default=0),
            'largest_loss': min((t['profit'] for t in self.trade_history), default=0),
            'average_trade_duration': self.calculate_average_duration()
        }
        
    def calculate_average_duration(self):
        """Calculate average trade duration"""
        if not self.trade_history:
            return 0
            
        durations = [
            (trade['exit_time'] - trade['entry_time']).total_seconds()
            for trade in self.trade_history
        ]
        return sum(durations) / len(durations)
        
    def reset(self):
        """Reset virtual trader state"""
        self.balance = self.initial_balance
        self.positions = []
        self.trade_history = []
        self.virtual_trades = []
        self.win_rate = 0
        self.profit_factor = 0
        self.max_drawdown = 0 