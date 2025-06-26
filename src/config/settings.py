
import os
from datetime import datetime

# Data Pipeline Configuration
DAILY_MAINWINDOW = 200
MONTHLY_MAINWINDOW = 30
SPY_MAINWINDOW = 30
VOLUME_SPIKEWINDOW = 10
ATR_RETURN_HORIZON_DAILY = 20
ATR_RETURN_HORIZON_SPY = 1

# Data Paths
BASE_DATA_DIR = os.path.join("..", "..", "data") # Relative to src/data_preprocess

# IBKR API Configuration (if applicable, though currently using get_ibkr_ohlc.py)
# IBKR_HOST = "127.0.0.1"
# IBKR_PORT = 7497
# IBKR_CLIENT_ID = 1

# Default Dates for Data Fetching
DEFAULT_START_DATE = "2005-01-01"
DEFAULT_END_DATE = datetime.now().strftime("%Y-%m-%d")

# Charting Configuration
CHART_WIDTH = 448
CHART_HEIGHT = 448
CHART_SCALE = 1
CHART_CROP_COORDS = (90, 375, 50, 400) # top, bottom, left, right
CHART_RESIZE_DIMS = (224, 224)

# Data Types
TORCH_FLOAT16 = "torch.float16"
TORCH_FLOAT32 = "torch.float32"