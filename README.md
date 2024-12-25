# portfolio-analysis-new

A Python-based portfolio analysis tool that helps investors analyze and simulate investment portfolios using historical data and Monte Carlo simulation.

## Features

- **Historical Data Analysis**: Fetches and analyzes historical stock data using the Yahoo Finance API
- **Portfolio Metrics Calculation**: 
  - Annual Returns
  - Annual Volatility
  - Sharpe Ratio
- **Monte Carlo Simulation**: Simulates possible future portfolio performance scenarios
- **Visualization**: Generates visual representations of simulation results
- **Risk Assessment**: Provides probability analysis of investment outcomes
- **Automated Reporting**: Generates comprehensive portfolio analysis reports

## Requirements

- Python 3.6+
- Required packages:
  ```
  yfinance
  pandas
  numpy
  matplotlib
  seaborn
  ```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ZhongyuXie921/portfolio-analysis.git
   cd portfolio-analysis
   ```

## Usage

1. Import the PortfolioAnalyzer class:
   ```python
   from portfolio_analyzer import PortfolioAnalyzer
   ```

2. Create an analyzer instance:
   ```python
   # Define your portfolio
   tickers = ['AAPL', 'GOOGL', 'MSFT']
   weights = [0.4, 0.3, 0.3]
   
   analyzer = PortfolioAnalyzer(tickers, weights)
   ```

3. Run analysis:
   ```python
   # Get historical data
   df = analyzer.fetch_data(start_date, end_date)
   
   # Calculate metrics
   metrics = analyzer.calculate_portfolio_metrics(df)
   
   # Run simulation
   simulated_returns = analyzer.monte_carlo_simulation(df)
   
   # Generate report
   report = analyzer.generate_report(metrics, simulated_returns)
   ```

## Output

The tool generates:
- A portfolio simulation chart (`portfolio_simulation.png`)
- A detailed analysis report (`portfolio_report.txt`) including:
  - Portfolio allocation
  - Historical performance metrics
  - Simulation results
  - Risk assessment

## Example Report Output

```
Portfolio Analysis Report
================

Portfolio Allocation:
--------------------
AAPL: 40.0%
GOOGL: 30.0%
MSFT: 30.0%

Historical Performance:
--------------------
Annual Return: XX.XX%
Annual Volatility: XX.XX%
Sharpe Ratio: X.XX
```

## Contributing

Feel free to fork the repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.


## Contact

Zhongyu Xie - [Your Contact Information]
Project Link: https://github.com/ZhongyuXie921/portfolio-analysis
