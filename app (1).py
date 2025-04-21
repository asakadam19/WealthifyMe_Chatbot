import streamlit as st
import os
import faiss
from openai import OpenAI
import numpy as np
import pandas as pd
import spacy
from dotenv import load_dotenv
import yfinance as yf
from enum import Enum

# ---------------------------
# Environment & Clients Setup
# ---------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load FAISS index and metadata
index = faiss.read_index("faiss_index.idx")
metadata = pd.read_pickle("faiss_metadata.pkl")

# Initialize spaCy model for entity recognition
nlp = spacy.load("en_core_web_sm")

# ---------------------------
# Helper: Company-to-Ticker Mapping
# ---------------------------
def convert_company_to_ticker(company):
    mapping = {
        "APPLE": "AAPL",
        "GOOGLE": "GOOGL",
        "ALPHABET": "GOOGL",
        "TESLA": "TSLA",
    }
    return mapping.get(company.upper(), company.upper())

# ---------------------------
# Intent & Market Data Enums/Functions
# ---------------------------
class IntentType(Enum):
    COMPARISON = "comparison"
    CONCEPT_EXPLANATION = "concept_explanation"
    MARKET_DATA = "market_data"
    FINANCIAL_ADVICE = "financial_advice"
    HISTORICAL_ANALYSIS = "historical_analysis"

def analyze_intent(query):
    messages = [
        {"role": "system", "content": (
            "You are a financial intent analyzer. Classify the user's query into one of these categories:\n"
            "- COMPARISON: Comparing two or more financial entities (e.g., companies or stocks)\n"
            "- CONCEPT_EXPLANATION: Explaining financial terms or concepts\n"
            "- MARKET_DATA: Current market information or stock prices\n"
            "- FINANCIAL_ADVICE: Seeking investment or financial advice\n"
            "- HISTORICAL_ANALYSIS: Analysis of historical performance\n"
            "Return ONLY the category name, nothing else."
        )},
        {"role": "user", "content": query}
    ]
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=messages,
        temperature=0
    )
    intent_str = response.choices[0].message.content.strip().lower()
    return IntentType(intent_str)

def extract_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in ("ORG", "PRODUCT", "MONEY", "GPE")]
    existing = {ent[0] for ent in entities}
    for token in doc:
        if token.pos_ == "PROPN" and token.text not in existing:
            if token.text.lower() not in ["compare", "and", "over", "stock"]:
                entities.append((token.text, "ORG"))
    st.write("DEBUG: All entities detected by spaCy:", entities)
    return entities

def extract_stock_symbol(query):
    entities = extract_entities(query)
    for entity, label in entities:
        if label == "ORG":
            return convert_company_to_ticker(entity)
    return None

def format_number(number_str):
    try:
        number = float(number_str)
        if number >= 1:
            return f"${number:,.2f}"
        else:
            return f"${number:.4f}"
    except (ValueError, TypeError):
        return "N/A"

def get_market_data(symbol):
    ticker = yf.Ticker(symbol)
    info = ticker.info
    historical = ticker.history(period="7d")

    historical_data_str = ""
    for date, row in historical.iterrows():
        date_str = date.strftime("%Y-%m-%d")
        close_price = f"${row['Close']:.2f}"
        historical_data_str += f"- {date_str}: {close_price}\n"

    return {
        "symbol": symbol,
        "current_price": info.get('currentPrice', 'N/A'),
        "change_percent": f"{info.get('regularMarketChangePercent', 'N/A')}%",
        "volume": info.get('volume', 'N/A'),
        "historical_data_str": historical_data_str
    }

# ---------------------------
# FAISS + GPT Definition Retrieval
# ---------------------------
def search_faiss_or_gpt(query, top_k=5):
    response = client.embeddings.create(
        input=[query],
        model="text-embedding-ada-002"
    )
    query_embedding = np.array(response.data[0].embedding).astype("float32").reshape(1, -1)
    D, I = index.search(query_embedding, top_k)

    st.write("DEBUG: Distances returned by FAISS:", D[0])
    st.write("DEBUG: Indices returned by FAISS:", I[0])

    valid_matches = []
    for idx, distance in zip(I[0], D[0]):
        if idx == -1 or idx >= len(metadata):
            continue
        row = metadata.iloc[idx]
        term = row["term"]
        definition = row["definition"]
        if distance > 0.5:
            continue
        valid_matches.append((term, definition))

    if valid_matches:
        best_match_term, best_match_definition = valid_matches[0]
        return best_match_term, best_match_definition, False

    term = query
    definition = generate_definition_openai(query)
    return term, definition, True

def generate_definition_openai(query):
    messages = [
        {"role": "system", "content": "You are a financial expert. Provide clear and accurate definitions."},
        {"role": "user", "content": f"Define '{query}' as a financial term concisely."}
    ]
    ai_response = client.chat.completions.create(model="gpt-4-turbo", messages=messages)
    return ai_response.choices[0].message.content.strip()

def concise_definition_agent(term, definition):
    messages = [
        {"role": "system", "content": "You are a financial assistant."},
        {"role": "user", "content": f"Provide a concise definition for '{term}':\n{definition}"}
    ]
    ai_response = client.chat.completions.create(model="gpt-4-turbo", messages=messages)
    return ai_response.choices[0].message.content.strip()

def simplified_explanation_agent(term, definition):
    messages = [
        {"role": "system", "content": "You are a friendly financial educator."},
        {"role": "user", "content": f"Explain '{term}' in simple terms, using an analogy if possible."}
    ]
    ai_response = client.chat.completions.create(model="gpt-4-turbo", messages=messages)
    return ai_response.choices[0].message.content.strip()

def source_recommendation_agent(term, definition):
    messages = [
        {"role": "system", "content": "You recommend reputable financial sources."},
        {"role": "user", "content": f"Recommend reliable sources for learning about '{term}'."}
    ]
    ai_response = client.chat.completions.create(model="gpt-4-turbo", messages=messages)
    return ai_response.choices[0].message.content.strip()

def follow_up_question_agent(term, definition):
    messages = [
        {"role": "system", "content": "You are a financial education assistant."},
        {"role": "user", "content": f"Given the term '{term}' and its definition: {definition}, what is a good follow-up question?"}
    ]
    ai_response = client.chat.completions.create(model="gpt-4-turbo", messages=messages)
    return ai_response.choices[0].message.content.strip()
def compare_investments(inv1, inv2):
    ticker1 = convert_company_to_ticker(inv1)
    ticker2 = convert_company_to_ticker(inv2)

    data1 = get_market_data(ticker1)
    data2 = get_market_data(ticker2)

    data_str1 = f"""
📊 **Market Data for {ticker1}**
- Current Price: {format_number(data1['current_price'])}
- Change: {data1['change_percent']}
- Volume: {data1['volume']}

📈 **7-Day Historical Data:**
{data1['historical_data_str']}
    """ if data1 else "❌ Market data not available for " + ticker1

    data_str2 = f"""
📊 **Market Data for {ticker2}**
- Current Price: {format_number(data2['current_price'])}
- Change: {data2['change_percent']}
- Volume: {data2['volume']}

📈 **7-Day Historical Data:**
{data2['historical_data_str']}
    """ if data2 else "❌ Market data not available for " + ticker2

    # Request GPT-4 to create a comparison considering the above data
    messages = [
        {"role": "system", "content": "You are an expert financial advisor."},
        {"role": "user", "content": (
            f"Compare these two companies:\n\n"
            f"{inv1} (Ticker: {ticker1}) Market Data:\n{data_str1}\n\n"
            f"{inv2} (Ticker: {ticker2}) Market Data:\n{data_str2}\n\n"
            "Compare them in terms of risk, returns, fees, and target investor."
        )}
    ]

    comparison = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=messages
    ).choices[0].message.content.strip()

    # Return market data first, then the GPT analysis
    return f"""
{data_str1}

{data_str2}

📝 **GPT-4 Comparison Analysis**  
{comparison}
    """


# ---------------------------
# Streamlit Frontend
# ---------------------------
st.set_page_config(page_title="💬 Money Mentor", layout="centered")

st.title("💬 Money Mentor")
st.markdown("Your AI-powered financial education companion with real market data and definitions!")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "👋 Hi! I'm your Money Mentor. Ask me about any financial term, stock, or concept!"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about a financial term, stock, or concept..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            intent = analyze_intent(prompt)
            response = ""
            if intent == IntentType.MARKET_DATA:
                symbol = extract_stock_symbol(prompt)
                if symbol:
                    market_data = get_market_data(symbol)
                    response = f"""
📊 **Market Data for {market_data['symbol']}**

Current Price: {format_number(market_data['current_price'])}
Change: {market_data['change_percent']}
Volume: {market_data['volume']}

📈 **7-Day Historical Data:**
{market_data['historical_data_str']}
                    """
                else:
                    response = "❌ Could not detect a valid stock symbol."
            elif intent == IntentType.CONCEPT_EXPLANATION:
                term, definition, _ = search_faiss_or_gpt(prompt)
                concise = concise_definition_agent(term, definition)
                simple = simplified_explanation_agent(term, definition)
                sources = source_recommendation_agent(term, definition)
                follow_up = follow_up_question_agent(term, definition)
                response = f"""
💡 **{term}**

📘 **Definition:** {concise}

🔍 **Explanation:** {simple}

📚 **Learn More:** {sources}

🤔 **Follow-Up Question:** {follow_up}
                """
            elif intent == IntentType.COMPARISON:
                entities = extract_entities(prompt)
                if len(entities) >= 2:
                    inv1 = entities[0][0]
                    inv2 = entities[1][0]
                    response = compare_investments(inv1, inv2)
                else:
                    response = "❌ Please mention two entities to compare."
            else:
                response = "❌ Sorry, I couldn't understand your request. Please rephrase."

        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

if len(st.session_state.messages) > 1:
    if st.button("Clear Chat"):
        st.session_state.messages = [{"role": "assistant", "content": "👋 Hi! I'm your Money Mentor. Ask me about any financial term, stock, or concept!"}]
        st.experimental_rerun()