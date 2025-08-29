import streamlit as st
import requests

# --- 1. CONFIGURATION ---
API_URL = "http://localhost:8000/ask"
st.set_page_config(page_title="Pharmabot", layout="wide", initial_sidebar_state="collapsed")
st.title("Pharmabot: Medication Assistant")

# --- 2. STREAMLIT APP LOGIC ---

# Initialize chat history
if "messages" not in st.session_state:
    # Each message is a dictionary that can also hold sources
    st.session_state.messages = [{
        "role": "assistant", 
        "content": "Hi! How can I help you with your medication questions today?",
        "sources": None
    }]

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Display the main content of the message
        st.markdown(message["content"])

        # If the message is from the assistant and has sources, display them
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("View Retrieved Sources"):
                for i, source in enumerate(message["sources"]):
                    med_name = source.get('drug_name', 'Unknown Medication')
                    source_col = source.get('section_title', 'Unknown Source')
                    st.write(f"**Source {i+1}:** {med_name} - *{source_col}*")
                    st.caption(source.get('text'))

# Accept and process user input
if prompt := st.chat_input("Ask about side effects, usage, etc."):
    # Add user message to the session state history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # --- THIS IS THE FIX ---
    # Display the user's message immediately in the chat
    with st.chat_message("user"):
        st.markdown(prompt)
    # --- END OF FIX ---

    # Now, get the assistant's response
    with st.chat_message("assistant"):
        with st.spinner("Searching and thinking..."):
            try:
                response = requests.post(API_URL, json={"question": prompt})
                response.raise_for_status()
                
                api_response = response.json()
                full_response = api_response.get("answer", "Sorry, I couldn't find an answer.")
                sources = api_response.get("source_documents", [])
                
                # Add the bot's full response and sources to the history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_response, 
                    "sources": sources
                })
                
                # Rerun the script to display the latest messages from history
                st.rerun()

            except requests.exceptions.RequestException as e:
                error_message = f"Error: Could not connect to the backend API. Details: {e}"
                st.error(error_message)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": error_message,
                    "sources": None
                })
                st.rerun()