# app.py
import streamlit as st
import requests

API_URL = "http://localhost:8000/ask"

st.set_page_config(page_title="Pharmabot", layout="wide", initial_sidebar_state="collapsed")
st.title("Pharmabot: Medication Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant", 
        "content": "Hi! How can I help you with your medication questions today?",
        "sources": None
    }]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("View Retrieved Sources"):
                for i, source in enumerate(message["sources"]):
                    med_name = source.get('drug_name', 'Unknown Medication')
                    # Corrected to use 'topic'
                    source_topic = source.get('topic', 'Unknown Topic')
                    st.write(f"**Source {i+1}:** {med_name} - *{source_topic}*")
                    st.caption(source.get('text'))

if prompt := st.chat_input("Ask about side effects, usage, etc."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching and thinking..."):
            try:
                response = requests.post(API_URL, json={"question": prompt})
                response.raise_for_status() # Will raise an error for 4xx/5xx responses

                api_response = response.json()
                full_response = api_response.get("answer", "Sorry, I couldn't find an answer.")
                sources = api_response.get("source_documents", [])

                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_response, 
                    "sources": sources
                })
                st.rerun()

            except requests.exceptions.HTTPError as e:
                # Specifically handle HTTP errors like 404, 500, etc.
                error_detail = e.response.json().get("detail", e.response.text)
                error_message = f"Error from backend API: {e.response.status_code}. Detail: {error_detail}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message, "sources": None})

            except requests.exceptions.RequestException as e:
                # Handle connection errors
                error_message = f"Error: Could not connect to the backend API. Is it running? Details: {e}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message, "sources": None})