# ==========================================================
# EMOTION ENGINE v5.4 — GITHUB VERSION
# ==========================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from fredapi import Fred
from datetime import datetime
from matplotlib.gridspec import GridSpec

EXPORT_DIR = "output"

ENGINE_VERSION = "Emotion Engine v5.4 GITHUB + LIQUIDITY(ANFCI)"

PANIC_TH   = 25
RISKOFF_TH = 45
NEUTRAL_TH = 60
RISKON_TH  = 75
GREED_TH   = 80

PANIC_VIX_LEVEL = 28
PANIC_VIX_MIN   = 24
PANIC_SI_MAX    = 30

STRESS_EXTREME = 80
STRESS_HIGH    = 62
STRESS_ELEV    = 45
STRESS_MOD     = 30

TERM_HALF = 0.5
VIX_TIER = {"28": 10, "24": 4, "20": 1}

ANFCI_EASY_STRONG = -0.50
ANFCI_EASY_MILD   =  0.00
ANFCI_TIGHT_MILD  =  0.25
ANFCI_TIGHT_STRONG=  0.50

os.makedirs(EXPORT_DIR, exist_ok=True)

LATEST_TXT  = os.path.join(EXPORT_DIR, "latest.txt")
PUSH_TXT    = os.path.join(EXPORT_DIR, "push.txt")
DASH_PNG    = os.path.join(EXPORT_DIR, "dashboard.png")
VERSION_TXT = os.path.join(EXPORT_DIR, "version.txt")

tickers = [
    "SPY","QQQ","IWM","DIA",
    "^VIX","^VIX3M","^VIX9D","^VVIX",
    "^MOVE","TLT","HYG","XLF","XLI","XLK","SMH"
]

data = yf.download(
    tickers,
    period="12mo",
    interval="1d",
    auto_adjust=True,
    progress=False
)

close = data["Close"].copy()

spy = close["SPY"].dropna()
idx = spy.index

def align(sym):
    s = close.get(sym)
    if s is None:
        return pd.Series(index=idx)
    return s.reindex(idx).ffill()

qqq   = align("QQQ")
iwm   = align("IWM")
dia   = align("DIA")
vix   = align("^VIX")
vix3m = align("^VIX3M")
vix9d = align("^VIX9D")
vvix  = align("^VVIX")
move  = align("^MOVE")
tlt   = align("TLT")
hyg   = align("HYG")
xlf   = align("XLF")
xli   = align("XLI")
xlk   = align("XLK")
smh   = align("SMH")

def ma(s,n):
    return s.rolling(n).mean()

FRED_API_KEY = os.getenv("FRED_API_KEY","").strip()

fred=None
if FRED_API_KEY:
    try:
        fred=Fred(api_key=FRED_API_KEY)
    except:
        fred=None

def fred_series(series):
    if fred is None:
        return pd.Series(index=idx)
    try:
        s=fred.get_series(series)
        s.index=pd.to_datetime(s.index)
        return s.reindex(idx).ffill()
    except:
        return pd.Series(index=idx)

hy_oas  = fred_series("BAMLH0A0HYM2")
bbb_oas = fred_series("BAMLC0A4CBBB")
anfci   = fred_series("ANFCI")

def liquidity_score(i):
    a=anfci.iloc[i]
    if not pd.notna(a):
        return 0
    if a<=ANFCI_EASY_STRONG:
        return 8
    elif a<ANFCI_EASY_MILD:
        return 4
    elif a>=ANFCI_TIGHT_STRONG:
        return -8
    elif a>=ANFCI_TIGHT_MILD:
        return -4
    else:
        return 0

def compute_components(i):

    trend=0
    if spy.iloc[i]>ma(spy,50).iloc[i]: trend+=10
    if spy.iloc[i]>ma(spy,200).iloc[i]: trend+=10
    if qqq.iloc[i]>ma(qqq,50).iloc[i]: trend+=10

    momentum=0
    if i>=20:
        mom=(spy.iloc[i]/spy.iloc[i-20]-1)*100
        if mom>5: momentum+=10
        elif mom<-5: momentum-=10

    vix_score=0
    v=vix.iloc[i]
    if v>25: vix_score-=20
    elif v>20: vix_score-=10
    elif v<15: vix_score+=10

    liq=liquidity_score(i)

    components={
        "Trend":trend,
        "Momentum":momentum,
        "VIX":vix_score,
        "Liquidity":liq
    }

    total=sum(components.values())
    sentiment=float(np.clip(50+total,0,100))

    return components,total,sentiment

i_last=len(idx)-1
today=idx[i_last].date()

components,score,sentiment=compute_components(i_last)

if sentiment<=PANIC_TH:
    regime="PANIC"
elif sentiment<=RISKOFF_TH:
    regime="RISK_OFF"
elif sentiment<=NEUTRAL_TH:
    regime="NEUTRAL"
elif sentiment<=RISKON_TH:
    regime="RISK_ON"
else:
    regime="GREED"

icon={"PANIC":"🔴","RISK_OFF":"🟠","NEUTRAL":"🟡","RISK_ON":"🟢","GREED":"🟣"}[regime]

spy_px=float(spy.iloc[i_last])
vix_last=float(vix.iloc[i_last])
vvix_last=float(vvix.iloc[i_last])
move_last=float(move.iloc[i_last])
vix9_last=float(vix9d.iloc[i_last])
vix3_last=float(vix3m.iloc[i_last])

hy_last=float(hy_oas.iloc[i_last])
bbb_last=float(bbb_oas.iloc[i_last])
anfci_last=float(anfci.iloc[i_last])

panic_window = regime=="PANIC"
greed_window = regime=="GREED"

stress_score=100 if panic_window else 30
stress_level="EXTREME" if panic_window else "LOW"

RUN_ID=datetime.now().strftime("%Y-%m-%d %H:%M:%S ET")

msg=f"""
{icon} {regime}

Date: {today}

Sentiment Index: {sentiment:.1f}
Score: {score:.1f}

SPY {spy_px:.2f}
VIX {vix_last:.2f}
VVIX {vvix_last:.2f}
MOVE {move_last:.2f}
VIX9D {vix9_last:.2f}
VIX3M {vix3_last:.2f}

HY_OAS {hy_last:.2f}
BBB_OAS {bbb_last:.2f}
ANFCI {anfci_last:.3f}

PANIC_WINDOW_OPEN = {panic_window}
GREED_WINDOW_OPEN = {greed_window}

Market Stress Score: {stress_score:.1f}  |  Level: {stress_level}

RUN_ID: {RUN_ID}
""".strip()

print(msg)

with open(LATEST_TXT,"w") as f:
    f.write(msg)

if panic_window or greed_window:
    with open(PUSH_TXT,"w") as f:
        f.write(msg)

with open(VERSION_TXT,"w") as f:
    f.write(ENGINE_VERSION)

plt.figure(figsize=(12,5))
plt.plot(spy.index,spy.values)
plt.title("SPY")
plt.savefig(DASH_PNG)
import requests

TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if TOKEN and CHAT_ID:
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    requests.post(url, data={
        "chat_id": CHAT_ID,
        "text": msg
    })

    url_photo = f"https://api.telegram.org/bot{TOKEN}/sendPhoto"
    with open(DASH_PNG, "rb") as photo:
        requests.post(url_photo, data={"chat_id": CHAT_ID}, files={"photo": photo})
