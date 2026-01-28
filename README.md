# MARKET_PRED

📈AI Stock Analyst: The Evolution of Prediction & Analysis
A Project by Chetan Yogbharti (IIT Indore)

This repository documents my journey in building a sophisticated financial dashboard. It transitions from traditional statistical forecasting to Deep Learning and ends with a Generative AI-powered analyst.

🔬 Task 1: The Predictive Engine
Objective: Forecast the next 30 days of stock prices using historical data.

The Evolution: ARIMA to LSTM
My approach evolved as I sought higher accuracy in the volatile stock market:

The Starting Point (ARIMA): I initially began with the ARIMA (AutoRegressive Integrated Moving Average) model. While great for stable trends, it struggled with the non-linear "noise" and sudden shifts of the modern market.

The Final Solution (LSTM): To capture long-term dependencies and complex patterns, I pivoted to Long Short-Term Memory (LSTM) networks. Using a window of the past 60 days, the model learns the "memory" of price movements to project future trends.

🤖 Task 2: The RAG-Based Analyst
Objective: Create an AI that doesn't just "guess," but "reads" the news before answering.

Beyond Simple LLMs
Instead of relying on a standard AI's outdated training data, I built a Retrieval-Augmented Generation (RAG) pipeline:

Live Context: The system fetches the latest headlines for any ticker (e.g., TSLA, NVDA) using NewsAPI.

Semantic Memory: These headlines are converted into numerical vectors using Google's text-embedding-004 and stored in a ChromaDB vector database.

Grounded Answers: When you ask a question, the system retrieves the most relevant news snippets and feeds them to Gemini 1.5 Flash, ensuring the answer is based on what is happening today.

🛠 Tech Stack
Forecasting: Python, yfinance, PyTorch/Keras (LSTM).

Generative AI: LangChain, Google Gemini 1.5 Flash.

Database: ChromaDB (Vector Store).

Dashboard: Streamlit.

🎭 About the Developer
I am a student at IIT Indore where I balance the logical world of AI with the expressive world of performance arts.

Technical: Deep Learning, RAG Architectures, Financial Modeling.

Creative: Mime, Stand-up Comedy, and Organizing TEDxIITIndore.

I believe the best AI systems aren't just powerful; they are communicative and secure.
