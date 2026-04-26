# Stock Prediction Reinforcement Learning

## Installation

Creating virtual environment:

```shell
uv venv --python python=3.12
```

Install Packages:

```shell
uv pip install -r requirements.txt
```

## RL Model Feaure

We picked 10 highest market cap companies, along with their 10 features.

- Bias (均線乖離率) $BIAS = \frac{\text{Close Price} - \text{N day Moving Average}}{N Day Moving Average} \times 100%$
    - 正乖離（+BIAS）：股價高於均線。數值越大，代表短線漲幅過大，有回檔修正的壓力。
    - 負乖離（-BIAS）：股價低於均線。數值越小（負越多），代表短線跌幅過深，有反彈向上的機率。
- SMA (簡單移動平均線) $\frac{\sum \limits_{i=1}^{N} P_i}{N}$
    - 常用作支撐線與壓力線，例如 20 日均線（月線）被視為中短期多空分界。


## Train RL Model

## Running Backtest
