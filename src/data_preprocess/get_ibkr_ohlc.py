import asyncio
from ib_insync import *
import random
import pandas as pd
from datetime import datetime, timedelta

async def fetch_ohlc(ticker, start_date, end_date, barSizeSetting,ib,
                         whatToShow='TRADES', useRTH=True,
                         formatDate=1, keepUpToDate=False):
    contract = Stock(ticker, 'SMART', 'USD')
    contract.primaryExchange = 'NASDAQ'
    await ib.qualifyContractsAsync(contract)

    all_bars = []
    dt_end = end_date.strftime("%Y%m%d %H:%M:%S")
    batch_length = "365 D"
    
    while True:
        bars = await ib.reqHistoricalDataAsync(
            contract,
            endDateTime=dt_end,
            durationStr=batch_length,
            barSizeSetting=barSizeSetting,
            whatToShow=whatToShow,
            useRTH=useRTH,
            formatDate=formatDate,
            keepUpToDate=keepUpToDate
        )
        if not bars:
            break
        # print(bars)
        all_bars += bars

        earliest = min(b.date for b in bars)
        earliest = datetime.combine(earliest, datetime.min.time())
        if earliest < start_date or len(bars) == 0:
            break

        dt_end = (earliest - timedelta(seconds=1)).strftime("%Y%m%d %H:%M:%S")
    data = pd.DataFrame([
        {'date': b.date, 'open': b.open, 'high': b.high,
         'low': b.low, 'close': b.close, 'volume': b.volume}
        for b in sorted(all_bars, key=lambda b: b.date)
        if datetime.combine(b.date, datetime.min.time())>= start_date
    ])
    return data
async def connect_ibkr():
    try:
        ib = IB()
        client_id = random.randint(1, 10000)
        print(f"🔌 Connecting with clientId = {client_id}")
        
        await ib.connectAsync('127.0.0.1', 4002, clientId=client_id)

        accounts = ib.managedAccounts()
        print(f"✅ Connected account(s): {accounts}")
        return ib
    except:
        print('Error connecting to IBKR')


async def disconnect_ibkr(ib):
    ib.disconnect()
    print('ib disconnected successfully')


async def get_ohlc(payload:dict):
    ib=await connect_ibkr()
    payload['ib']=ib
    data= await fetch_ohlc(**payload)
    disconnect_ibkr(ib)
    return data

if __name__ == "__main__":
    payload = {
        "ticker": "TSLA",
        "start_date": datetime.now() - timedelta(days=9000),
        "end_date": datetime.now(),
        "barSizeSetting": "1 month"
    }
    data = asyncio.run(get_ohlc(payload))
    print(data)
