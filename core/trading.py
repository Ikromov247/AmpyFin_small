import time

class Trade:
    """
    A class that represents a single buy, sell or hold trade.
    """
    MAX_SINGLE_BUY = 0.2  # Maximum percentage of cash to use in a single buy trade
    MAX_SINGLE_TICKER_HOLDING = 0.5  # Maximum percentage of portfolio value for a single ticker

    def __init__(self, strategy_name, ticker, action, current_price, account_cash, portfolio_qty, total_portfolio_value):
        self.strategy_name = strategy_name
        self.ticker = ticker
        self.action = action
        self.current_price = current_price
        self.account_cash = account_cash
        self.portfolio_qty = portfolio_qty
        self.total_portfolio_value = total_portfolio_value
        self.net_worth = self.account_cash + self.total_portfolio_value
        self.trading_time = time.time()  # Unix time in seconds

        self.trade_quantity = self._calculate_quantity()

    def _calculate_quantity(self) -> int:
        """Calculates trade quantity, ensuring it adheres to all constraints."""

        if self.action == 'hold':
            return 0

        if self.action not in ('buy', 'sell'):
            raise ValueError(f"Invalid trade action: {self.action}")

        # Calculate the maximum amount of money that can be invested in a single ticker
        max_ticker_value = self.net_worth * self.MAX_SINGLE_TICKER_HOLDING

        if self.action == 'buy':
            max_shares_investment = int(max_ticker_value // self.current_price - self.portfolio_qty)
            # maximum number of shares that can be bought with the available cash in the account
            max_shares_cash = int(self.account_cash * self.MAX_SINGLE_BUY // self.current_price)
            return max(0, min(max_shares_investment, max_shares_cash))  # Returns 0 if constraints not met.

        elif self.action == 'sell':
            half_qty = max(1, int(self.portfolio_qty * 0.5)) # Sell at least 1, up to half
            return min(self.portfolio_qty, half_qty)  # Returns 0 if portfolio_qty is 0.


    def execute_trade(self):
        """Executes the trade and updates portfolio values. Assumes trade is valid."""
        
        trade_value = self.current_price * self.trade_quantity

        if self.action == 'buy':
            # we are using our cash to buy shares
            self.account_cash -= trade_value
            self.portfolio_qty += self.trade_quantity
        
        elif self.action == 'sell':
            # we are selling our shares to get cash
            self.account_cash += trade_value
            self.portfolio_qty -= self.trade_quantity
        
        # update the total portfolio value
        self.net_worth = self.account_cash + self.portfolio_qty * self.current_price
        self.total_portfolio_value = self.net_worth - self.account_cash

        return self.account_cash, self.portfolio_qty, self.total_portfolio_value, self.net_worth