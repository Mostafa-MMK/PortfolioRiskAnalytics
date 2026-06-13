

# Portfolio Analysis without (General View)
"""
What it does ======
1. Download the tickers directly from the YahooFinance
2. Wrangle the data and calculate log_returns
3. Select weighting of the asset -- EW = Equal Weighted, MC = Market Cap, RP = Risk Parity, CW = Customized, Else --> random weights!
4. Calculate portfolio Returns, Standard Deviation, and Sharpe Ratio
5. De-mystify the risk attributions accross the assets
"""


def portfolio_analytics(tickers, risk_free_rate, n_trading_days, confidence_level,weights = None, port_type = None):
    import yfinance as yf
    import pandas as pd
    import numpy as np
    rng = np.random.default_rng(1000)
    
    # Loading the data 
    df_close = yf.download(tickers=tickers, start='2020-01-01', auto_adjust=True)['Close']
    df_close.columns = tickers
    df_vol = yf.download(tickers=tickers, start='2020-01-01', auto_adjust=True)['Volume']
    df_vol.columns = tickers
    
    # Market Cap data
    market_caps = []

    for ticker in tickers:
        t = yf.Ticker(ticker)
        market_caps.append(t.info['marketCap'])

    # Wrangling the data
    drop_10na_cols = df_close.columns[df_close.isna().sum() > 200]
    df_close = df_close.drop(columns=drop_10na_cols)
    df_mcap = df_close * df_vol
    
    if len(drop_10na_cols) == 0:
        print(f'... Less than 10% NA Values!')
    else:
        print(f' More than 10% NA in the columns: {drop_10na_cols}')

    # Log returns
    df_returns = np.log(df_close / df_close.shift(1)).dropna()
    n_assets = df_returns.shape[1]
    
    # Weighting techniques
    if port_type == 'EW':
        weights = np.ones(n_assets) / n_assets
    
    elif port_type == 'MC':
        weights = np.array(market_caps)
        weights = weights / weights.sum()
        # weights = df_mcap.iloc[-1].values (proxy for MCap)
        # weights = weights / weights.sum()

    elif port_type == 'RP':
        weights = 1 / df_returns.std().values
        weights /= weights.sum()
    
    elif port_type == 'CW':
    
        if weights is None:
            raise ValueError("For port_type='CW', you must provide weights parameter")
    
        weights = np.array(weights).flatten()
        weights = weights / weights.sum()

    else:
        weights = rng.random(size=n_assets)
        weights /= weights.sum()


    # Portfolio Returns
    cov_mat = df_returns.cov()
    corr_mat = df_returns.corr()
    port_returns =  df_returns @ weights
    port_variance = weights.T @ cov_mat @ weights

    # Annualized Returns
    annual_returns = port_returns.mean() * n_trading_days
    annual_std = np.sqrt(port_variance * n_trading_days)
    annual_sharpe = (annual_returns - risk_free_rate) / annual_std

    # Calculating VaR and ES (annualized assuming IID and Gaussian)
    daily_var = np.percentile(port_returns, (1 - confidence_level) * 100) # remember: np.quantile(df_returns, 1 - confidence_level) as it takes 0.95 format.
    annual_var = daily_var * np.sqrt(n_trading_days)
    daily_es = port_returns[port_returns <= daily_var].mean()
    annual_es = daily_es * np.sqrt(n_trading_days)

    annual_results = pd.DataFrame({
        'Annual Port Return': [annual_returns],
        'Annual Port Std': [annual_std],
        'Annual Port Sharpe': [annual_sharpe],
        f'Annual VaR_{confidence_level}': [annual_var],
        f'Annual ES_{confidence_level}': [annual_es]
    })

    # Risk Attribution
    overall_port_std = np.sqrt(weights.T @ cov_mat @ weights)
    marginal_risk_contribution = (cov_mat @ weights) / overall_port_std
    component_risk_contribution = weights * marginal_risk_contribution
    pct_risk_contribution = component_risk_contribution / component_risk_contribution.sum()
    
    risk_attribution = pd.DataFrame({
        'Asset': df_returns.columns,
        'Weight': weights,
        'Marginal Risk Contribution': marginal_risk_contribution,
        'Component Risk Contribution': component_risk_contribution,
        '% of Total Risk': pct_risk_contribution
    })


    return annual_results, risk_attribution, corr_mat




class Portfolio_Analyis:
    def __init__(self, tickers, ticker_name, risk_free_rate, start_date, n_trading_days, confidence_level, seed = None, weights = None, port_type = None):
        self.tickers = tickers
        self.ticker_name = ticker_name
        self.risk_free_rate = risk_free_rate
        self.start_date = start_date
        self.n_trading_days = n_trading_days
        self.confidence_level = confidence_level
        self.weights = weights
        self.port_type = port_type
        self.seed = seed

    def Data_Wrangling_Func(self):
        import pandas as pd
        import numpy as np
        from scipy.stats import norm
        import yfinance as yf
        rng = np.random.default_rng(seed=self.seed)


        # Downloading Tickers
        df_close = yf.download(tickers=self.tickers, start=self.start_date, auto_adjust=True)['Close']
        df_close.columns = self.ticker_name

        na_cols_200 = df_close.columns[df_close.isna().sum() > 200]
        df_close = df_close.drop(columns=na_cols_200)

        log_returns = np.log(df_close / df_close.shift(1)).dropna()
        n_assets = log_returns.shape[1]


        market_caps = []
        for ticker in self.tickers:
            t = yf.Ticker(ticker)
            market_caps.append(t.info['marketCap'])
        market_caps = np.array(market_caps)

        # Weights Measuring
        if self.port_type == 'EW':
            self.weights = np.ones(n_assets) / n_assets

        elif self.port_type == 'MC':
            self.weights = market_caps / market_caps.sum()

        elif self.port_type == 'RP':
            self.weights = 1 / log_returns.std().values
            self.weights /= self.weights.sum()

        elif self.port_type == 'RM':
            self.weights = rng.random(size=n_assets)
            self.weights /= self.weights.sum()

        else:
            if self.weights is None:
                raise ValueError('Please insert weights!')

            self.weights = np.array(self.weights).flatten()
            self.weights /= self.weights.sum() # for ensuring sum is 1.0      

        # Calculating Portfolio
        corr_mat = log_returns.corr()
        cov_mat = log_returns.cov()

        port_returns = log_returns @ self.weights
        port_std = np.sqrt(self.weights.T @ cov_mat @ self.weights)


        # Annualized Portfolio Params
        annual_port_returns = (log_returns @ self.weights).mean() * self.n_trading_days
        annual_port_std = np.sqrt((self.weights.T @ cov_mat @ self.weights) * self.n_trading_days)
        annual_sharpe_ratio = (annual_port_returns - self.risk_free_rate) / annual_port_std
        daily_VaR = np.percentile(port_returns, (1 - self.confidence_level) * 100)
        daily_ES = port_returns[port_returns <= daily_VaR].mean()
        annual_VaR = daily_VaR * np.sqrt(self.n_trading_days)
        annual_ES = daily_ES * np.sqrt(self.n_trading_days)

        annual_results = pd.DataFrame({
            'Annual Port Returns': [annual_port_returns],
            'Annual Port Std.': [annual_port_std],
            'Annual Sharpe Ratio': [annual_sharpe_ratio],
            'Annual VaR': [annual_VaR],
            'Annual ES': [annual_ES]
        })


        # Risk Breakdown
        avg_daily_returns = log_returns.mean()
        avg_daily_std = log_returns.std()
        daily_risk_free_rate = self.risk_free_rate / self.n_trading_days
        avg_sharpe_ratios = ((log_returns - daily_risk_free_rate) / log_returns.std()).mean()

        marginal_risk_contribution = (cov_mat @ self.weights) / np.sqrt(self.weights.T @ cov_mat @ self.weights)
        component_risk_contribution = self.weights * marginal_risk_contribution
        pct_risk_contribution = component_risk_contribution / component_risk_contribution.sum()

        risk_breakdown = pd.DataFrame({
            'Asset': log_returns.columns,
            'Weight': self.weights,
            'Avg. Daily Returns': avg_daily_returns,
            'Avg. Daily Std.': avg_daily_std,
            'Avg. Sharpe Ratio': avg_sharpe_ratios,
            'Marginal Risk Contribution': marginal_risk_contribution,
            'component Risk Contribution': component_risk_contribution,
            '% of Total Risk': pct_risk_contribution
        })
        risk_breakdown = risk_breakdown.sort_values('% of Total Risk', ascending=False)


        return risk_breakdown, annual_results

