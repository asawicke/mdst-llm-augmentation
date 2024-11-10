# **Stock Market Data Summarizer**
This project provides a command-line interface (CLI) tool for fetching, storing, and summarizing recent stock market data using Alpha Vantage, MongoDB, and OpenAI APIs. It leverages text embeddings and cosine similarity to produce financial summaries based on user queries.

## **Features**
- **Fetch Stock Data:** Retrieves daily stock prices (open, high, low, close, volume) for a specified stock symbol using the Alpha Vantage API. 
- **Store and Query Data:** Stores and queries stock data from a MongoDB database.
- **Text Embeddings:** Embeds stock summaries using OpenAI's embedding model to enhance query relevance. 
- **Similarity Scoring:** Computes cosine similarity between user queries and stock data summaries. 
- **Financial Summaries:** Generates financial data summaries based on user prompts and relevant data.
