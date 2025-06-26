import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import requests
import json
import numpy as np
import io
from PIL import Image
import pickle
import torch
import cv2
import os
import pandas_ta as pta
import random
from datetime import datetime, timedelta
from data_preprocess.get_ibkr_ohlc import get_ohlc
import asyncio
from pathlib import Path
from config.settings import (
    DAILY_MAINWINDOW,
    MONTHLY_MAINWINDOW,
    SPY_MAINWINDOW,
    VOLUME_SPIKEWINDOW,
    ATR_RETURN_HORIZON_DAILY,
    ATR_RETURN_HORIZON_SPY,
    BASE_DATA_DIR,
    DEFAULT_START_DATE,
    DEFAULT_END_DATE,
    TORCH_FLOAT32
)
from data_preprocess.chart_utils import ChartGenerator

current_dir = os.path.dirname(os.path.abspath(__file__))

class DataPipeline:
    def __init__(self):
        self.daily_mainwindow = DAILY_MAINWINDOW
        self.monthly_mainwindow = MONTHLY_MAINWINDOW
        self.spy_mainwindow = SPY_MAINWINDOW
        self.volume_spikewindow = VOLUME_SPIKEWINDOW
        self.chart_generator = ChartGenerator()

    def format_price_data(self, ohlc_data, atr_return_horizon=None):
        ohlc_data.reset_index(inplace=True, drop=False)
        expected_caps = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        if all(col in ohlc_data.columns for col in expected_caps):
            ohlc_data.rename(columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }, inplace=True)

        ohlc_data['date'] = pd.to_datetime(ohlc_data['date']).dt.date
        ohlc_data['date'] = ohlc_data['date'].astype(str)
        if atr_return_horizon:
            ohlc_data["ATR"] = pta.atr(high=ohlc_data["high"], low=ohlc_data["low"], close=ohlc_data["close"], length=14)
            ohlc_data['chg_perc_ATR'] = ((ohlc_data['close'].shift(-atr_return_horizon) - ohlc_data['close']) / ohlc_data['ATR'])
        return ohlc_data

    def get_volumespike_signals(self, ohlc_data):
        ohlc_data['volume_spike'] = 0
        ohlc_data.loc[(ohlc_data['volume'] > ohlc_data['volume'].rolling(window=self.volume_spikewindow).max().shift(self.volume_spikewindow)) & (ohlc_data['close'] > ohlc_data['open']), 'volume_spike'] = 1
        ohlc_data.dropna(inplace=True)
        ohlc_data.reset_index(inplace=True, drop=True)
        return ohlc_data

    def generate_temporal_subsets(self, daily_data, monthly_data, spy_data, i):
        sub_daily_data = daily_data[i:i + self.daily_mainwindow]
        sub_daily_data.reset_index(inplace=True, drop=True)
        date = sub_daily_data['date'].iloc[-1]

        sub_monthly_data = monthly_data.loc[monthly_data['date'] < date]
        sub_monthly_data.reset_index(inplace=True, drop=True)

        sub_spy_data = spy_data.loc[spy_data['date'] < date]
        sub_spy_data = sub_spy_data.tail(self.spy_mainwindow)
        sub_spy_data.reset_index(inplace=True, drop=True)

        return sub_daily_data, sub_monthly_data, sub_spy_data    def format_price_data(self,ohlc_data,atr_return_horizon=None):
        ohlc_data.reset_index(inplace=True, drop=False)
        expected_caps = ['Date','Open', 'High', 'Low', 'Close', 'Volume']
        if all(col in ohlc_data.columns for col in expected_caps):
            ohlc_data.rename(columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }, inplace=True)
          
        ohlc_data['date'] = pd.to_datetime(ohlc_data['date']).dt.date
        ohlc_data['date'] = ohlc_data['date'].astype(str)
        if atr_return_horizon:
            ohlc_data["ATR"] = pta.atr(high=ohlc_data["high"], low=ohlc_data["low"], close=ohlc_data["close"], length=14)
            ohlc_data['chg_perc_ATR'] = ((ohlc_data['close'].shift(-atr_return_horizon) - ohlc_data['close']) / ohlc_data['ATR'])
        return ohlc_data
    def get_volumespike_signals(self,ohlc_data):
        ohlc_data['volume_spike']=0
        ohlc_data.loc[(ohlc_data['volume']>ohlc_data['volume'].rolling(window=self.volume_spikewindow).max().shift(self.volume_spikewindow)) & (ohlc_data['close']>ohlc_data['open']),'volume_spike']=1
        ohlc_data.dropna(inplace=True)
        ohlc_data.reset_index(inplace=True,drop=True)
        return ohlc_data
    def generate_temporal_subsets(self,daily_data, monthly_data, spy_data, i):
        sub_daily_data = daily_data[i:i+self.daily_mainwindow]
        sub_daily_data.reset_index(inplace=True, drop=True)
        date = sub_daily_data['date'].iloc[-1]

        sub_monthly_data = monthly_data.loc[monthly_data['date'] < date]
        sub_monthly_data.reset_index(inplace=True, drop=True)

    def process_ticker_data(self, ticker_info, spy_data, mode):
        ticker = ticker_info["ticker"]
        start_date_str = ticker_info["entry"]
        end_date_str = ticker_info["exit"]

        start_date = datetime.strptime(start_date_str, "%Y-%m-%d") if start_date_str != 'infinity' else DEFAULT_START_DATE
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d") if end_date_str != 'infinity' else DEFAULT_END_DATE

        if np.busday_count(start_date.date(), end_date.date()) > self.daily_mainwindow:
            daily_data = asyncio.run(get_ohlc({
                "ticker": ticker,
                "start_date": start_date,
                "end_date": end_date,
                "barSizeSetting": "1 day"
            }))
            print(f"\033[92m{ticker}\033[0m \033[93m{len(daily_data)}\033[0m")

            if len(daily_data) > self.daily_mainwindow:
                monthly_data = asyncio.run(get_ohlc({
                    "ticker": ticker,
                    "start_date": DEFAULT_START_DATE,
                    "end_date": end_date,
                    "barSizeSetting": "1 month"
                }))

                daily_data = self.format_price_data(ohlc_data=daily_data, atr_return_horizon=ATR_RETURN_HORIZON_DAILY)
                monthly_data = self.format_price_data(ohlc_data=monthly_data)
                daily_data = self.get_volumespike_signals(ohlc_data=daily_data)

                print(ticker)
                print(start_date_str)
                print(end_date_str)

                entries = []
                for i in range(len(daily_data) - self.daily_mainwindow):
                    sub_daily_data, sub_monthly_data, sub_spy_data = self.generate_temporal_subsets(
                        daily_data=daily_data, monthly_data=monthly_data, spy_data=spy_data, i=i
                    )

                    if sub_daily_data['volume_spike'].iloc[-1] == 1 and len(sub_spy_data) == self.spy_mainwindow:
                        chart_1d = self.chart_generator.img_preprocess(self.chart_generator.create_chart(sub_daily_data))
                        chart_1mo = self.chart_generator.img_preprocess(self.chart_generator.create_chart(sub_monthly_data))
                        spy_sequential = torch.from_numpy(sub_spy_data['chg_perc_ATR'].values).to(TORCH_FLOAT32)
                        target = torch.tensor([sub_daily_data['chg_perc_ATR'].iloc[-1]]).float()

                        print(f"\033[92m{sub_daily_data['chg_perc_ATR'].iloc[-1]}\033[0m")
                        print(f"\032[92m{sub_daily_data['volume_spike'].iloc[-1]}\032[0m")

                        sample_id = f"{ticker}_{start_date_str}_{end_date_str}_{i}"
                        
                        data_path_1d = os.path.join(current_dir, BASE_DATA_DIR, mode, "chart_1d", f"{sample_id}.pt")
                        data_path_1mo = os.path.join(current_dir, BASE_DATA_DIR, mode, "chart_1mo", f"{sample_id}.pt")
                        data_path_spy = os.path.join(current_dir, BASE_DATA_DIR, mode, "spy_seq", f"{sample_id}.pt")
                        data_path_target = os.path.join(current_dir, BASE_DATA_DIR, mode, "target", f"{sample_id}.pt")

                        torch.save(chart_1d, data_path_1d)
                        torch.save(chart_1mo, data_path_1mo)
                        torch.save(spy_sequential, data_path_spy)
                        torch.save(target, data_path_target)

                        entries.append({
                            "id": sample_id,
                            "chart_1d": data_path_1d,
                            "chart_1mo": data_path_1mo,
                            "spy_seq": data_path_spy,
                            "target": data_path_target,
                        })
                return entries
        return []

if __name__ == "__main__":
    mode = 'validation'
    pipeline = DataPipeline()

    # Create necessary directories
    for sub_dir in ["chart_1d", "chart_1mo", "spy_seq", "target"]:
        Path(os.path.join(current_dir, BASE_DATA_DIR, mode, sub_dir)).mkdir(parents=True, exist_ok=True)

    df_tickers = pd.read_csv(f"s&p_tickers_{mode}.csv")

    # Fetch SPY data once
    spy_data_raw = asyncio.run(get_ohlc({
        "ticker": "SPY",
        "start_date": DEFAULT_START_DATE,
        "end_date": DEFAULT_END_DATE,
        "barSizeSetting": "1 day"
    }))
    spy_data = pipeline.format_price_data(ohlc_data=spy_data_raw, atr_return_horizon=ATR_RETURN_HORIZON_SPY)

    all_entries = []
    for _, ticker_info in df_tickers.iterrows():
        ticker_entries = pipeline.process_ticker_data(ticker_info, spy_data, mode)
        all_entries.extend(ticker_entries)

    # Save the index file
    index_file_path = os.path.join(current_dir, BASE_DATA_DIR, mode, "index.json")
    with open(index_file_path, "w") as f:
        json.dump(all_entries, f, indent=2)
                            "chart_1mo": os.path.join(current_dir, "..","..","data",mode,"chart_1mo",f"{sample_id}.pt"),
                            "spy_seq": os.path.join(current_dir, "..","..","data",mode,"spy_seq",f"{sample_id}.pt"),
                            "target": os.path.join(current_dir, "..","..","data",mode,"target",f"{sample_id}.pt"),
                        })
                        with open(os.path.join(current_dir, "..","..","data",mode,"index.json"), "w") as f:
                            json.dump(entries, f, indent=2)
