import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import requests
import json
import numpy as np
import plotly.graph_objects as go
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
from config.settings import *
current_dir = os.path.dirname(os.path.abspath(__file__))
class DataPipeline:
    def __init__(self,daily_mainwindow,monthly_mainwindow,spy_mainwindow,volume_spikewindow):
        self.daily_mainwindow=daily_mainwindow
        self.monthly_mainwindow=monthly_mainwindow
        self.spy_mainwindow=spy_mainwindow
        self.volume_spikewindow=volume_spikewindow
        pass
    def create_chart(self,data):
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'], 
            low=data['low'],
            close=data['close']
        ))
        fig.update_layout(
            plot_bgcolor='black',  
            paper_bgcolor='black',  
            xaxis=dict(
                showgrid=False,      
                zeroline=False,      
                showticklabels=False 
            ),
            yaxis=dict(
                showgrid=False,      
                zeroline=False,      
                showticklabels=False 
            ),
            font=dict(color='white'),
            showlegend=False
        )

        fig.update_layout(
            xaxis=dict(
                rangeslider=dict(visible=False)
            )
        )
        fig.update_layout(yaxis=dict(range=[data['low'].min(), data['high'].max()]))
        fig.update_layout(xaxis=dict(range=[data.index.min(), data.index.max()]))
        img_bytes = fig.to_image(format='jpg',width=CHART_WIDTH,height=CHART_HEIGHT,scale=1)
        img = Image.open(io.BytesIO(img_bytes))
        return img

    def img_preprocess(self,img):
        img = np.array(img)
        img = img[90:375, 50:400]
        img = Image.fromarray(img)
        img = img.resize(CHART_RESIZE_DIMS)
        img=np.array(img)
        img = (img / 255.0)  # Keep as float for processing
        # img = (img * 255).astype(np.uint8)  # Convert to uint8 for storage
        img = torch.from_numpy(img).to(torch.float16)
        img = img.permute(2, 0, 1)
        return img
    def format_price_data(self,ohlc_data,atr_return_horizon=None):
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

        sub_spy_data = spy_data.loc[spy_data['date'] < date]
        sub_spy_data = sub_spy_data.tail(self.spy_mainwindow)
        sub_spy_data.reset_index(inplace=True, drop=True)

        return sub_daily_data, sub_monthly_data, sub_spy_data

if __name__ == "__main__":
    mode='train'
    pipeline=DataPipeline(daily_mainwindow=DAILY_MAINWINDOW,monthly_mainwindow=MONTHLY_MAINWINDOW,spy_mainwindow=SPY_MAINWINDOW,volume_spikewindow=VOLUME_SPIKEWINDOW)
    Path(os.path.join(current_dir, "..","..","data",mode,"chart_1d")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(current_dir, "..","..","data",mode,"chart_1mo")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(current_dir, "..","..","data",mode,"spy_seq")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(current_dir, "..","..","data",mode,"target")).mkdir(parents=True, exist_ok=True)
    df_tickers=pd.read_csv(f"s&p_tickers_{mode}.csv")
    spy_data=asyncio.run(get_ohlc( {
            "ticker": "SPY",
            "start_date": datetime.strptime("2005-01-01", "%Y-%m-%d"),
            "end_date": datetime.now(),
            "barSizeSetting": "1 day"
        }))
    spy_data=pipeline.format_price_data(ohlc_data=spy_data,atr_return_horizon=ATR_RETURN_HORIZON_SPY)
    import json

    index_path = os.path.join(current_dir, "..", "..", "data", mode, "index.json")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            try:
                entries = json.load(f)
                print('here')
            except Exception:
                entries = []
    else:
        entries = []
    for i in range(len(df_tickers)):
        ticker=df_tickers["ticker"].iloc[i]
        start_date=df_tickers["entry"].iloc[i]
        end_date=df_tickers["exit"].iloc[i]
        if start_date=='infinity':
            start_date=DEFAULT_START_DATE
        if end_date=='infinity':
            end_date=DEFAULT_END_DATE
        if np.busday_count(datetime.strptime(start_date, "%Y-%m-%d").date(), datetime.strptime(end_date, "%Y-%m-%d").date())>pipeline.daily_mainwindow:
            daily_data=asyncio.run(get_ohlc( {
                    "ticker": ticker,
                    "start_date": datetime.strptime(start_date, "%Y-%m-%d"),
                    "end_date": datetime.strptime(end_date, "%Y-%m-%d"),
                    "barSizeSetting": "1 day"
                }))
            print(f"\033[92m{ticker}\033[0m \033[93m{len(daily_data)}\033[0m")
            if len(daily_data)>pipeline.daily_mainwindow:
                monthly_data=asyncio.run(get_ohlc( {
                    "ticker": ticker,
                    "start_date": datetime.strptime("2005-01-01", "%Y-%m-%d"),
                    "end_date": datetime.strptime(end_date, "%Y-%m-%d"),
                    "barSizeSetting": "1 month"
                }))
                daily_data=pipeline.format_price_data(ohlc_data=daily_data,atr_return_horizon=ATR_RETURN_HORIZON_DAILY)
                monthly_data=pipeline.format_price_data(ohlc_data=monthly_data)
                daily_data=pipeline.get_volumespike_signals(ohlc_data=daily_data)
        
                print(ticker)
                print(start_date)
                print(end_date)
                for i in range(len(daily_data)-pipeline.daily_mainwindow):
                    sub_daily_data, sub_monthly_data, sub_spy_data=pipeline.generate_temporal_subsets(daily_data=daily_data, monthly_data=monthly_data, spy_data=spy_data,i=i)
                    if sub_daily_data['volume_spike'].iloc[len(sub_daily_data)-1]==1 and len(sub_spy_data)==pipeline.spy_mainwindow:
                        chart_1d=pipeline.img_preprocess(pipeline.create_chart(sub_daily_data))
                        chart_1mo=pipeline.img_preprocess(pipeline.create_chart(sub_monthly_data))
                        spy_sequential=sub_spy_data['chg_perc_ATR'].values
                        spy_sequential=torch.from_numpy(spy_sequential).to(torch.float32)

                        target=torch.tensor([sub_daily_data['chg_perc_ATR'].iloc[len(sub_daily_data)-1]]).float()
                        print(f"\033[92m{sub_daily_data['chg_perc_ATR'].iloc[-1]}\033[0m")
                        print(f"\032[92m{sub_daily_data['volume_spike'].iloc[len(sub_daily_data)-1]}\032[0m")
                        sample_id = f"{ticker}_{start_date}_{end_date}_{i}"
                        torch.save(chart_1d, os.path.join(current_dir, "..","..","data",mode,"chart_1d",f"{sample_id}.pt"))
                        torch.save(chart_1mo, os.path.join(current_dir, "..","..","data",mode,"chart_1mo",f"{sample_id}.pt"))
                        torch.save(spy_sequential, os.path.join(current_dir, "..","..","data",mode,"spy_seq",f"{sample_id}.pt"))
                        torch.save(target, os.path.join(current_dir, "..","..","data",mode,"target",f"{sample_id}.pt"))

                        entries.append({
                            "id": sample_id,
                            "chart_1d": os.path.join(current_dir, "..","..","data",mode,"chart_1d",f"{sample_id}.pt"),
                            "chart_1mo": os.path.join(current_dir, "..","..","data",mode,"chart_1mo",f"{sample_id}.pt"),
                            "spy_seq": os.path.join(current_dir, "..","..","data",mode,"spy_seq",f"{sample_id}.pt"),
                            "target": os.path.join(current_dir, "..","..","data",mode,"target",f"{sample_id}.pt"),
                        })
                        with open(os.path.join(current_dir, "..","..","data",mode,"index.json"), "w") as f:
                            json.dump(entries, f, indent=2)