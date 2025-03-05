# Deriv Trading Bot

A Python-based trading bot that connects to the Deriv.com API for automated trading.

## Setup Instructions

1. **Install Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API Token**
   - Create a `.env` file in the root directory
   - Add your Deriv API token:
     ```
     DERIV_API_TOKEN=your_api_token_here
     ```
   - To get your API token:
     1. Log in to your Deriv account
     2. Go to Settings > API Token
     3. Create a new token with trading permissions
     4. Copy the token and paste it in the `.env` file

3. **Run the Bot**
   ```bash
   python main.py
   ```

## Features
- WebSocket connection to Deriv API
- Secure authentication using API tokens
- Automatic reconnection on connection loss
- Real-time account balance display

## Security Notes
- Never share your API token
- Keep your `.env` file secure and never commit it to version control
- Use a demo account for testing

## Disclaimer
This is a basic trading bot template. Make sure to understand the risks involved in automated trading before using real funds. 