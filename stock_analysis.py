# ============================================================
#  STOCK PRICE FORECASTING - Full Project
#  Portfolio Project | Time Series + AI Forecasting
# ============================================================
#
#  HOW TO RUN:
#  C:/Users/nitis/AppData/Local/Python/pythoncore-3.14-64/python.exe stock_analysis.py
#
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

import yfinance as yf
from prophet import Prophet

print("=" * 55)
print("   STOCK PRICE FORECASTING PROJECT")
print("=" * 55)

# ============================================================
#  STEP 1: DOWNLOAD LIVE STOCK DATA
# ============================================================

# You can change this to any stock ticker:
# AAPL = Apple | MSFT = Microsoft | GOOGL = Google
# RELIANCE.NS = Reliance | TCS.NS = TCS | INFY.NS = Infosys

STOCK = "AAPL"
START = "2022-01-01"
END   = "2025-01-01"

print(f"\nDownloading {STOCK} stock data from Yahoo Finance...")
df = yf.download(STOCK, start=START, end=END, auto_adjust=True)
df = df.reset_index()

# Flatten column names if multi-level
if isinstance(df.columns, pd.MultiIndex):
    df.columns = [col[0] for col in df.columns]

# Rename Date column if needed
df.columns = [str(c).strip() for c in df.columns]
if 'Datetime' in df.columns:
    df = df.rename(columns={'Datetime': 'Date'})
if 'index' in df.columns:
    df = df.rename(columns={'index': 'Date'})

print("Columns found:", df.columns.tolist())
print(f"Downloaded {len(df)} trading days of data")
print(f"Date range: {df['Date'].min().date()} → {df['Date'].max().date()}")
print(f"Latest closing price: ${df['Close'].iloc[-1]:.2f}")

# ============================================================
#  STEP 2: CALCULATE TECHNICAL INDICATORS
# ============================================================

# Moving Averages
df['MA_20']  = df['Close'].rolling(window=20).mean()
df['MA_50']  = df['Close'].rolling(window=50).mean()
df['MA_200'] = df['Close'].rolling(window=200).mean()

# Daily Return
df['Daily_Return'] = df['Close'].pct_change() * 100

# Volatility (20-day rolling std)
df['Volatility'] = df['Daily_Return'].rolling(window=20).std()

# Price change
price_change = df['Close'].iloc[-1] - df['Close'].iloc[0]
pct_change   = (price_change / df['Close'].iloc[0]) * 100

print("\n--- Stock Performance Summary ---")
print(f"Starting price : ${df['Close'].iloc[0]:.2f}")
print(f"Current price  : ${df['Close'].iloc[-1]:.2f}")
print(f"Total change   : ${price_change:.2f} ({pct_change:.1f}%)")
print(f"Highest price  : ${df['High'].max():.2f}")
print(f"Lowest price   : ${df['Low'].min():.2f}")
print(f"Avg daily return: {df['Daily_Return'].mean():.3f}%")
print(f"Avg volatility : {df['Volatility'].mean():.2f}%")

# Best and worst days
best_day  = df.loc[df['Daily_Return'].idxmax()]
worst_day = df.loc[df['Daily_Return'].idxmin()]
print(f"\nBest day  : {str(best_day['Date'])[:10]} → +{best_day['Daily_Return']:.2f}%")
print(f"Worst day : {str(worst_day['Date'])[:10]} → {worst_day['Daily_Return']:.2f}%")

# ============================================================
#  STEP 3: PROPHET FORECASTING (30-day prediction)
# ============================================================

print("\n--- Training Prophet Forecasting Model ---")

# Prophet needs columns named 'ds' and 'y'
prophet_df = df[['Date', 'Close']].copy()
prophet_df.columns = ['ds', 'y']
prophet_df['ds'] = pd.to_datetime(prophet_df['ds']).dt.tz_localize(None)

# Train the model
model = Prophet(
    daily_seasonality=False,
    weekly_seasonality=True,
    yearly_seasonality=True,
    changepoint_prior_scale=0.05
)
model.fit(prophet_df)

# Predict next 30 days
future   = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# Get forecast values
future_forecast = forecast[forecast['ds'] > prophet_df['ds'].max()]
pred_price = future_forecast['yhat'].iloc[-1]
pred_low   = future_forecast['yhat_lower'].iloc[-1]
pred_high  = future_forecast['yhat_upper'].iloc[-1]

print(f"30-day price forecast : ${pred_price:.2f}")
print(f"Forecast range        : ${pred_low:.2f} — ${pred_high:.2f}")
current = df['Close'].iloc[-1]
direction = "UP" if pred_price > current else "DOWN"
print(f"Predicted direction   : {direction} ({abs(pred_price-current)/current*100:.1f}%)")

# ============================================================
#  STEP 4: VISUALIZATIONS
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle(f'{STOCK} Stock Analysis & 30-Day Forecast',
             fontsize=16, fontweight='bold')
plt.subplots_adjust(hspace=0.4, wspace=0.3)

# Chart 1: Price with Moving Averages
ax = axes[0, 0]
ax.plot(df['Date'], df['Close'], color='#2563EB', linewidth=1.5,
        label='Close Price', alpha=0.9)
ax.plot(df['Date'], df['MA_20'],  color='#F59E0B', linewidth=1,
        label='20-day MA', linestyle='--')
ax.plot(df['Date'], df['MA_50'],  color='#10B981', linewidth=1,
        label='50-day MA', linestyle='--')
ax.plot(df['Date'], df['MA_200'], color='#EF4444', linewidth=1,
        label='200-day MA', linestyle='--')
ax.set_title(f'{STOCK} Price + Moving Averages', fontweight='bold')
ax.set_ylabel('Price ($)')
ax.legend(fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

# Chart 2: 30-Day Prophet Forecast
ax = axes[0, 1]
# Historical (last 6 months)
last6 = df.tail(180)
ax.plot(last6['Date'], last6['Close'], color='#2563EB',
        linewidth=2, label='Actual Price')
# Forecast
ax.plot(future_forecast['ds'], future_forecast['yhat'],
        color='#F59E0B', linewidth=2, linestyle='--', label='Forecast')
ax.fill_between(future_forecast['ds'],
                future_forecast['yhat_lower'],
                future_forecast['yhat_upper'],
                alpha=0.2, color='#F59E0B', label='Confidence Range')
ax.axvline(x=prophet_df['ds'].max(), color='gray',
           linestyle=':', linewidth=1, label='Forecast Start')
ax.set_title('30-Day Price Forecast (Prophet)', fontweight='bold')
ax.set_ylabel('Price ($)')
ax.legend(fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

# Chart 3: Daily Returns Distribution
ax = axes[1, 0]
returns = df['Daily_Return'].dropna()
ax.hist(returns, bins=50, color='#2563EB', alpha=0.7, edgecolor='white')
ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
ax.axvline(x=returns.mean(), color='#F59E0B', linestyle='--',
           linewidth=1.5, label=f'Mean: {returns.mean():.3f}%')
ax.set_title('Daily Returns Distribution', fontweight='bold')
ax.set_xlabel('Daily Return (%)')
ax.set_ylabel('Frequency')
ax.legend(fontsize=9)

# Chart 4: Volume Bar Chart
ax = axes[1, 1]
colors = ['#10B981' if r >= 0 else '#EF4444'
          for r in df['Daily_Return'].fillna(0)]
ax.bar(df['Date'], df['Volume']/1e6, color=colors, alpha=0.7, width=1)
ax.set_title('Trading Volume (Green=Up Day, Red=Down Day)',
             fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Volume (Millions)')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

plt.tight_layout()
plt.savefig('stock_dashboard.png', dpi=150, bbox_inches='tight')
print("\nDashboard saved as: stock_dashboard.png")

# ============================================================
#  STEP 5: BUSINESS INSIGHTS
# ============================================================

print("\n" + "="*55)
print("  INVESTMENT INSIGHTS & ANALYSIS")
print("="*55)

print(f"\n1. PERFORMANCE: {STOCK} moved {pct_change:+.1f}% over the period.")
if pct_change > 0:
    print(f"   A $10,000 investment would now be worth ${10000*(1+pct_change/100):,.0f}")

print(f"\n2. FORECAST: Prophet model predicts price will go {direction}")
print(f"   in the next 30 days to around ${pred_price:.2f}")
print(f"   (Range: ${pred_low:.2f} — ${pred_high:.2f})")

ma20  = df['MA_20'].iloc[-1]
ma50  = df['MA_50'].iloc[-1]
close = df['Close'].iloc[-1]
if close > ma20 > ma50:
    signal = "BULLISH — price above both 20 and 50-day MA"
elif close < ma20 < ma50:
    signal = "BEARISH — price below both moving averages"
else:
    signal = "NEUTRAL — mixed moving average signals"

print(f"\n3. TREND SIGNAL: {signal}")
print(f"   Current: ${close:.2f} | 20MA: ${ma20:.2f} | 50MA: ${ma50:.2f}")

print(f"\n4. RISK: Average daily volatility of {df['Volatility'].mean():.2f}%")
if df['Volatility'].mean() > 2:
    print("   HIGH RISK stock — large daily price swings")
else:
    print("   MODERATE RISK stock — relatively stable")

print("\n  DISCLAIMER: This is for educational purposes only.")
print("  Never make real investment decisions based on ML models alone.")
print("="*55)
