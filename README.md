# LongVPA-net

Supervised long-only trading model for US stocks powered by Volume Price Analysis (VPA). This repository contains the complete pipeline from data preprocessing to model training for a sophisticated trading bot.

## Features

*   **Automated Data Ingestion:** Scripts to fetch historical OHLC (Open, High, Low, Close) data from various sources like Yahoo Finance and Interactive Brokers.
*   **Robust Data Preprocessing:** A comprehensive pipeline to transform raw financial data into features suitable for machine learning, incorporating Volume Price Analysis principles.
*   **Deep Learning Model Architecture:** Implementation of a neural network designed for time-series forecasting in financial markets.
*   **Supervised Training Framework:** Tools and scripts for training the VPA-powered model using historical stock data.
*   **Modular Design:** Clearly separated modules for data handling, model definition, and training logic, promoting maintainability and extensibility.

## Installation

To set up the project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/mahdijaouadi/LongVPA-net.git
    cd LongVPA-net
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    While a `requirements.txt` is not provided, based on the project's nature, you will likely need the following libraries. Install them manually:
    ```bash
    pip install pandas numpy scikit-learn tensorflow  # Or pytorch, depending on model_architecture.py
    pip install yfinance  # For Yahoo Finance data
    pip install ib_insync  # For Interactive Brokers data (if used)
    ```
    *Note: Please check the specific imports within the `src` directory files to ensure all necessary libraries are installed.*

## Usage

The project is structured into three main phases: data preprocessing, model definition, and training.

### 1. Data Preprocessing

This phase involves fetching and preparing the financial data.

*   **Fetch Data:**
    *   For Yahoo Finance data:
        ```bash
        python src/data_preprocess/get_yfinance_ohlc.py
        ```
    *   For Interactive Brokers data (requires IBKR TWS/Gateway running and configured):
        ```bash
        python src/data_preprocess/get_ibkr_ohlc.py
        ```
*   **Run Data Pipeline:**
    The `pipeline.py` script is responsible for applying VPA principles and preparing the dataset for model training.
    ```bash
    python src/data_preprocess/pipeline.py
    ```
    *Ensure `s&p_tickers.csv`, `s&p_tickers_train.csv`, `s&p_tickers_validation.csv`, and `s&p_tickers_test.csv` are correctly populated with stock tickers.*

### 2. Model Architecture

The `model_architecture.py` file defines the neural network model. You can inspect and modify the model's layers and configuration here.

*   **View Model Definition:**
    ```bash
    # Open src/model/model_architecture.py in your editor
    ```

### 3. Training

The `training` directory contains scripts to train the defined model using the preprocessed data.

*   **Start Training:**
    ```bash
    python src/training/train_model.py  # Assuming a train_model.py exists or similar
    ```
    *Note: The exact training script name might vary. Please check the `src/training` directory for the main training entry point.*

## Project Structure

```
LongVPA-net/
├── src/
│   ├── data_preprocess/
│   │   ├── get_ibkr_ohlc.py       # Script to fetch OHLC data from Interactive Brokers
│   │   ├── get_yfinance_ohlc.py   # Script to fetch OHLC data from Yahoo Finance
│   │   ├── pipeline.py            # Main data preprocessing pipeline (VPA logic)
│   │   ├── s&p_tickers.csv        # List of S&P 500 tickers
│   │   ├── s&p_tickers_test.csv
│   │   ├── s&p_tickers_train.csv
│   │   └── s&p_tickers_validation.csv
│   ├── model/
│   │   └── model_architecture.py  # Defines the neural network model
│   └── training/
│       └── (e.g., train_model.py) # Scripts for model training
├── .gitignore                     # Specifies intentionally untracked files to ignore
├── LICENSE                        # Project license
└── README.md                      # This documentation file
```

## Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/YourFeature`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add some feature'`).
5.  Push to the branch (`git push origin feature/YourFeature`).
6.  Open a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
