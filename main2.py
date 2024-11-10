import openai
import os
import requests
from pymongo import MongoClient
from datetime import datetime, timedelta
from dotenv import load_dotenv
import numpy as np
import certifi


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
alpha_vantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
mongo_uri = os.getenv("MONGO_URI")


client = MongoClient(
    mongo_uri,
    tlsCAFile=certifi.where()
)
db = client["financial_data"]
stocks_collection = db["stocks"]
news_collection = db["news"]

def fetch_stock_data(symbol: str) -> dict:
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={alpha_vantage_api_key}"
    response = requests.get(url)
    data = response.json()
    if "Error Message" in data:
        raise Exception(f"Error fetching data for {symbol}")
    return data

def save_stock_data(symbol: str, data: dict):
    for date, daily_data in data["Time Series (Daily)"].items():
        record = {
            "symbol": symbol,
            "date": datetime.strptime(date, "%Y-%m-%d"),
            "open": float(daily_data["1. open"]),
            "high": float(daily_data["2. high"]),
            "low": float(daily_data["3. low"]),
            "close": float(daily_data["4. close"]),
            "volume": int(daily_data["5. volume"])
        }
        stocks_collection.update_one(
            {"symbol": symbol, "date": record["date"]}, {"$set": record}, upsert=True
        )

# Query MongoDB and Embed Data
def query_stock_data(symbol: str, start_date: datetime, end_date: datetime):
    return list(stocks_collection.find({
        "symbol": symbol,
        "date": {"$gte": start_date, "$lte": end_date}
    }))

def create_embeddings(texts):
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=texts
    )
    return [embedding['embedding'] for embedding in response['data']]

def compute_similarity(embedding1, embedding2):
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

#OpenAI Function Calling
def get_summary(prompt, context_data):
    messages = [
        {"role": "system", "content": "You are an assistant that provides financial data summaries."},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": f"Relevant data: {context_data}"}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0
    )
    return response['choices'][0]['message']['content']

#CLI
def main():
    while True:
        print("\n--- Stock Market Data Summarizer ---")
        query = input("Enter your query (or 'exit' to quit): ")
        if query.lower() == "exit":
            break

        # Parsing the query
        symbol = input("Enter the stock ticker symbol (e.g., AAPL): ").upper()
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()

        #Fetch, save, and query data
        try:
            stock_data = fetch_stock_data(symbol)
            save_stock_data(symbol, stock_data)
            records = query_stock_data(symbol, start_date, end_date)

            # Prep data for embedding and querying
            summaries = [f"On {r['date'].strftime('%Y-%m-%d')}, {symbol} closed at {r['close']}." for r in records]
            embeddings = create_embeddings(summaries)

            #pick most relevant data based on embeddings
            question_embedding = create_embeddings([query])[0]
            relevant_data = [
                summaries[i] for i in range(len(embeddings))
                if compute_similarity(embeddings[i], question_embedding) > 0.8
            ]
            response = get_summary(query, relevant_data)
            print(f"\nSummary:\n{response}")

        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
