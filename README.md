Automated Trading Bot Based on Congressional Trades & Public Sentiment
This project is a complete, end-to-end automated trading pipeline that leverages public data to generate and execute stock trades. The core hypothesis is that by analyzing the stock trades of U.S. politicians and combining this information with public sentiment from social media, we can identify potentially profitable trading opportunities.

The system is fully automated, from data scraping to model training and live paper trading via the Alpaca API, all controlled by a user-friendly Streamlit dashboard.

Project Structure
The project is organized into a modular pipeline, with each stage handled by a dedicated set of scripts:

/scrapers: Contains scripts for gathering raw data from public sources (Capitol Trades, Yahoo Finance, Reddit).

/processing: Includes scripts for cleaning, enriching, and engineering features from the raw data.

/modeling: Houses the machine learning pipeline, including signal generation, model training, and backtesting.

/testing: Contains the script for live paper trading execution via the Alpaca API.

/streamlit_app: The user interface to control and monitor the entire pipeline.

/data: The central location where all data files, from raw scrapes to final model predictions, are stored.

Setup and Installation
1. Clone the Repository
Clone this repository to your local machine.

2. Create a Python Virtual Environment
It is highly recommended to use a virtual environment to manage project dependencies.

python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`

3. Install Dependencies
Install all the required Python libraries using the requirements.txt file.

pip install -r requirements.txt

4. Configure API Keys
The project requires API keys for Reddit and Alpaca.

Create a file named .env in the root directory of the project and add your keys as follows:

# .env file

# Reddit API Credentials (for scraping public sentiment)
CLIENT_ID="YOUR_REDDIT_CLIENT_ID"
CLIENT_SECRET="YOUR_REDDIT_CLIENT_SECRET"
USER_AGENT="YourAppName/1.0"

# Alpaca API Credentials (for paper trading)
ALPACA_API_KEY="YOUR_ALPACA_PAPER_API_KEY"
ALPACA_SECRET_KEY="YOUR_ALPACA_PAPER_SECRET_KEY"

How to Run the Trading Bot
The entire pipeline is managed through the Streamlit dashboard.

Navigate to the project's root directory in your terminal.

Run the following command:

streamlit run streamlit_app/streamlit_dashboard.py

This will launch the dashboard in your web browser.

Using the Dashboard
Run Full Data Pipeline: Click this button in the sidebar to execute the entire workflow, from scraping the latest data to generating new trade suggestions. The logs will show the progress of each script in real time.

View Results: Once the pipeline is complete, the main dashboard will display the latest trade suggestions and the performance metrics from the most recent model training run.

Paper Trading on Alpaca: The final script in the pipeline (daily_trade_runner.py) will automatically connect to your Alpaca account and submit the high-confidence trades to your paper trading portfolio. You can log in to your Alpaca account to see the open positions and trade history.

Future Work
This project provides a strong foundation that can be extended in several ways:

Integrate Sell Signals: The current model is primarily trained on "buy" signals. The next major step is to enhance the ML model to also recognize and act on politician "sell" signals.

Advanced Risk Management: Implement more dynamic risk management strategies, such as portfolio-level risk limits or sector exposure constraints.

Alternative Data Sources: Incorporate other alternative data sources, such as economic indicators or corporate filings, to further enrich the feature set for the model.