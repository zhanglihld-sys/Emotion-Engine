import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fredapi import Fred
from datetime import datetime
import os

# =========================
# CONFIG
# =========================

fred = Fred(api_key=os.getenv("FRED_API_KEY"))

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# DATA DOWNLOAD
# =========================

tickers = [
    "SPY",
    "^VIX",
    "^VVIX",
    "^VIX9D",
    "^VIX3M",
    "^MOVE",
    "QQQ",
    "SMH",
    "IWM"
]

data = yf.download(tickers, period="1y", interval="1d", progress=False)["Close"]

spy = data["SPY"]
vix = data["^VIX"]
vvix = data["^VVIX"]
vix9d = data["^VIX9D"]
vix3m = data["^VIX3M"]
move = data["^MOVE"]
qqq = data["QQQ"]
smh = data["SMH"]
iwm = data["IWM"]

# =========================
# FRED DATA
# =========================

fred = Fred(api_key=FRED_API_KEY)

hy_oas = fred.get_series("BAMLH0A0HYM2")
bbb_oas = fred.get_series("BAMLC0A4CBBB")
anfci = fred.get_series("ANFCI")

# =========================
# LAST VALUES
# =========================

spy_last = spy.iloc[-1]
vix_last = vix.iloc[-1]
vvix_last = vvix.iloc[-1]
vix9d_last = vix9d.iloc[-1]
vix3m_last = vix3m.iloc[-1]
move_last = move.iloc[-1]

hy_last = hy_oas.iloc[-1]
bbb_last = bbb_oas.iloc[-1]
anfci_last = anfci.iloc[-1]

# =========================
# SENTIMENT SCORE
# =========================

score = 0

if spy_last > spy.rolling(50).mean().iloc[-1]:
    score += 10
else:
    score -= 10

if vix_last > 25:
    score -= 20
elif vix_last > 20:
    score -= 10

if vix9d_last > vix_last:
    score -= 5

if vix_last > vix3m_last:
    score -= 10

if vvix_last > 120:
    score -= 10

if move_last > 120:
    score -= 10
elif move_last > 100:
    score -= 5

if hy_last > hy_oas.tail(20).mean():
    score -= 8

if bbb_last > bbb_oas.tail(20).mean():
    score -= 4

if smh.iloc[-1] > smh.rolling(50).mean().iloc[-1]:
    score += 5

if iwm.iloc[-1] > iwm.rolling(50).mean().iloc[-1]:
    score += 5

if anfci_last < 0:
    score += 5

sentiment = 50 + score

sentiment = max(0, min(100, sentiment))

# =========================
# REGIME
# =========================

if sentiment >= 70:
    regime = "RISK_ON"
elif sentiment >= 55:
    regime = "NEUTRAL"
elif sentiment >= 40:
    regime = "RISK_OFF"
else:
    regime = "PANIC"

# =========================
# STRESS SCORE
# =========================

stress = 0

stress += max(0, (vix_last - 15) * 2)
stress += max(0, (vvix_last - 90) * 0.3)
stress += max(0, (move_last - 70) * 0.5)
stress += max(0, (hy_last - 3) * 20)

stress = min(100, stress)

# =========================
# PLOT
# =========================

fig = plt.figure(figsize=(12,6))

plt.plot(spy.tail(120))
plt.title("SPY Trend")

plt.savefig(f"{OUTPUT_DIR}/dashboard.png")

# =========================
# TEXT OUTPUT
# =========================

now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

text = f"""
Regime: {regime}

Sentiment Index: {sentiment}

SPY {spy_last:.2f}
VIX {vix_last:.2f}
VVIX {vvix_last:.2f}
MOVE {move_last:.2f}

HY_OAS {hy_last:.2f}
BBB_OAS {bbb_last:.2f}
ANFCI {anfci_last:.3f}

Market Stress Score: {stress:.1f}

RUN_TIME: {now}
"""

with open(f"{OUTPUT_DIR}/latest.txt","w") as f:
    f.write(text)

print(text)
print("dashboard saved:", f"{OUTPUT_DIR}/dashboard.png")
