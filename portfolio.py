# Import required libraries
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

class PortfolioAnalyzer:
    def __init__(self, tickers, weights=None):
        """
        Initialize portfolio analyzer
        
        Parameters:
            tickers (list): List of stock tickers, e.g., ['AAPL', 'GOOGL', 'MSFT']
            weights (list): Weight for each stock, if not specified, weights will be equally distributed
        """
        self.tickers = tickers
        if weights is None:
            self.weights = np.array([1/len(tickers)] * len(tickers))
        else:
            self.weights = np.array(weights)
            
        # Verify that weights sum to 1
        if not np.isclose(sum(self.weights), 1.0):
            raise ValueError("Sum of weights must equal 1")

    def fetch_data(self, start_date, end_date):
        """
        Fetch historical stock data
        """
        data = {}
        for ticker in self.tickers:
            try:
                stock = yf.Ticker(ticker)
                data[ticker] = stock.history(start=start_date, end=end_date)['Close']
            except Exception as e:
                print(f"Error fetching data for {ticker}: {str(e)}")
                return None
        
        # Merge all stock data into a single DataFrame
        return pd.DataFrame(data)

    def calculate_portfolio_metrics(self, df):
        """
        Calculate portfolio returns and risk metrics
        """
        # Calculate daily returns
        returns = df.pct_change()
        
        # Calculate portfolio daily returns
        portfolio_returns = returns.dot(self.weights)
        
        # Calculate annualized return
        annual_return = np.mean(portfolio_returns) * 252
        
        # Calculate annualized volatility
        annual_volatility = np.std(portfolio_returns) * np.sqrt(252)
        
        # Calculate Sharpe ratio (assuming risk-free rate of 2%)
        risk_free_rate = 0.02
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
        
        return {
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio
        }

    def monte_carlo_simulation(self, df, n_simulations=1000, time_horizon=252):
        """
        Perform Monte Carlo simulation
        
        Parameters:
            df: Historical price data
            n_simulations: Number of simulations
            time_horizon: Forecast time horizon (in days)
        """
        # Calculate daily returns
        returns = df.pct_change().dropna()
        
        # Calculate covariance matrix
        cov_matrix = returns.cov()
        
        # Calculate mean returns
        mean_returns = returns.mean()
        
        # Generate simulation paths
        simulated_returns = np.zeros((n_simulations, time_horizon))
        
        for i in range(n_simulations):
            # Generate random return sequences
            daily_returns = np.random.multivariate_normal(
                mean_returns,
                cov_matrix,
                time_horizon
            )
            
            # Calculate portfolio returns
            portfolio_returns = daily_returns.dot(self.weights)
            
            # Calculate cumulative returns
            simulated_returns[i] = np.cumprod(1 + portfolio_returns)
        
        return simulated_returns

    def plot_simulation_results(self, simulated_returns, initial_investment=10000):
        """
        Visualize simulation results
        """
        plt.figure(figsize=(12, 8))
        
        # Calculate portfolio value
        portfolio_values = simulated_returns * initial_investment
        
        # Plot all simulation paths
        plt.plot(portfolio_values.T, color='blue', alpha=0.1)
        
        # Calculate and plot mean path
        mean_path = portfolio_values.mean(axis=0)
        plt.plot(mean_path, 'r--', linewidth=2, label='Mean Path')
        
        # Calculate and plot confidence intervals
        percentiles = np.percentile(portfolio_values, [5, 95], axis=0)
        plt.fill_between(range(len(mean_path)), 
                        percentiles[0], 
                        percentiles[1], 
                        color='gray', 
                        alpha=0.3, 
                        label='90% Confidence Interval')
        
        plt.title('Portfolio Monte Carlo Simulation')
        plt.xlabel('Trading Days')
        plt.ylabel('Portfolio Value')
        plt.legend()
        
        return plt.gcf()

    def generate_report(self, metrics, simulated_returns, initial_investment=10000):
        """
        Generate analysis report
        """
        final_values = simulated_returns[:, -1] * initial_investment
        
        report = f"""
        Portfolio Analysis Report
        ================
        
        Portfolio Allocation:
        {'-' * 20}
        """
        
        for ticker, weight in zip(self.tickers, self.weights):
            report += f"\n{ticker}: {weight*100:.1f}%"
            
        report += f"""
        
        Historical Performance:
        {'-' * 20}
        Annual Return: {metrics['annual_return']*100:.2f}%
        Annual Volatility: {metrics['annual_volatility']*100:.2f}%
        Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
        
        Simulation Results:
        {'-' * 20}
        Number of Simulations: {len(simulated_returns)}
        Time Horizon: {len(simulated_returns[0])} days
        
        Expected Investment Results:
        Initial Investment: ${initial_investment:,.2f}
        Average Final Value: ${np.mean(final_values):,.2f}
        Worst Case (5th percentile): ${np.percentile(final_values, 5):,.2f}
        Best Case (95th percentile): ${np.percentile(final_values, 95):,.2f}
        
        Risk Assessment:
        Probability of Loss: {(final_values < initial_investment).mean()*100:.1f}%
        Probability of Doubling Investment: {(final_values > initial_investment*2).mean()*100:.1f}%
        """
        
        return report

def main():
    # Set portfolio parameters
    tickers = ['AAPL', 'GOOGL', 'MSFT']  # Apple, Google, Microsoft
    weights = [0.4, 0.3, 0.3]  # Investment weights
    
    # Create analyzer instance
    analyzer = PortfolioAnalyzer(tickers, weights)
    
    # Get data for the past year
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    df = analyzer.fetch_data(start_date, end_date)
    
    if df is not None:
        # Calculate historical metrics
        metrics = analyzer.calculate_portfolio_metrics(df)
        
        # Run Monte Carlo simulation
        simulated_returns = analyzer.monte_carlo_simulation(df)
        
        # Generate visualization
        fig = analyzer.plot_simulation_results(simulated_returns)
        fig.savefig('portfolio_simulation.png')
        
        # Generate report
        report = analyzer.generate_report(metrics, simulated_returns)
        
        # Save report
        with open('portfolio_report.txt', 'w') as f:
            f.write(report)
            
        print("Analysis complete! Please check the generated report and chart.")
        print("\nReport Preview:")
        print(report)

if __name__ == "__main__":
    main()