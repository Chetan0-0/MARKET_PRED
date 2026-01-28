# MARKET_PRED

📈 AI Stock Analyst: From ARIMA to RAG
Developed by Chetan Yogbharti (IIT Indore)

This project isn't just a dashboard; it’s the result of a "brain transplant" between Google Colab and VS Code. It tracks the evolution of a stock analysis system that combines deep learning predictions with real-time AI reasoning.

🔬 Task 1: The Predictive Logic (Price Forecasting)
The Evolution: I started with ARIMA, but realized it was too rigid for the chaotic swings of the modern market. I upgraded to LSTM to give the model a "memory."

The Logic Behind the Code:
Data Windowing: The code takes a 30-day sliding window of historical prices. It doesn't just look at yesterday; it looks at the sequence of the last one months.

LSTM Layers: Unlike standard neurons, LSTM cells have "gates" that decide which past information is important to keep and which is "noise" to forget.

Min-Max Scaling: Since stock prices can range from $10 to $4000, we squeeze them between 0 and 1 so the model's weights don't explode during training.

🤖 Task 2: The RAG Logic (AI Analyst)
The Problem: LLMs like Gemini have a "knowledge cutoff." They don't know what happened in the market this morning. The Solution: Retrieval-Augmented Generation (RAG).

The Logic Behind the Code:
Vector Embeddings: We use text-embedding-004 to turn news headlines into 768-dimensional mathematical vectors. This allows the AI to understand that a headline about "Skyrocketing Revenue" is semantically similar to a user asking about "Growth."

Similarity Search: When you ask a question, ChromaDB does a "distance calculation" to find the news articles closest to your query.

Contextual Prompting: We don't just ask Gemini a question. We wrap it in a prompt that says: "Read these 5 articles first, then answer based ONLY on this facts." This prevents the AI from "hallucinating."

🌋 The "VS Code vs. Colab" Struggle
Let’s be real—merging these two was harder than the actual math.

Trying to get a Google Colab project (Task 1) and a VS Code/Streamlit project (Task 2) to live together in the same GitHub repo felt like trying to perform a mime act in the middle of a stand-up set. Between the remote origin already exists errors and the "unrelated histories" drama, there were moments where I almost threw my laptop out of the window.

But, after some "Git-Jiu-Jitsu" 😭 and a lot of debugging, we managed the "Brain Transplant":

Exported the LSTM model weights (.pth) and Scaler (.pkl) from Colab.

Imported them into the VS Code local environment.

Deployed it all as a single, unified dashboard.

🛠 Tech Stack
Deep Learning: PyTorch/Keras (LSTM).

GenAI: Google Gemini 1.5 Flash & LangChain.

Database: ChromaDB.

Frontend: Streamlit.

🎭 About Me
I'm an Engineering Student at IIT Indore. I spend half my time coding AI agents and the other half performing Mime and Stand-up Comedy. I guess you could say I’m an expert at making things (and not the code part) perform under pressure.😏

I believe the best AI systems aren't just powerful; they are communicative and secure.

👋👋
👋👋
👋👋
