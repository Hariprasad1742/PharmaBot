# app.py
import streamlit as st
import requests

API_URL = "http://localhost:8000/ask"

st.set_page_config(page_title="Pharmabot", layout="wide", initial_sidebar_state="collapsed")
st.title("Pharmabot: Medication Assistant 💊")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Hi! How can I help you with your medication questions today?",
        "sources": None,
        "processing_time": None
    }]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display sources and processing time only for assistant messages
        if message["role"] == "assistant":
            if message.get("sources"):
                with st.expander("View Retrieved Sources"):
                    for i, source in enumerate(message["sources"]):
                        med_name = source.get('drug_name', 'Unknown Medication')
                        source_topic = source.get('section_title', 'Unknown Topic')
                        st.write(f"**Source {i+1}:** {med_name} - *{source_topic}*")
                        st.caption(source.get('text'))
            
            if message.get("processing_time") is not None:
                st.caption(f"Generated in {message['processing_time']:.2f} seconds")

# Accept user input
if prompt := st.chat_input("Ask about side effects, usage, etc."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Searching and thinking..."):
            try:
                # Call the backend API
                response = requests.post(API_URL, json={"question": prompt})
                response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)

                api_response = response.json()
                full_response = api_response.get("answer", "Sorry, I couldn't find an answer.")
                sources = api_response.get("source_documents", [])
                processing_time = api_response.get("processing_time")

                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "sources": sources,
                    "processing_time": processing_time
                })
                # Rerun the app to display the new message and time
                st.rerun()

            except requests.exceptions.HTTPError as e:
                # Handle specific HTTP errors from the backend
                error_detail = "No additional details provided."
                try:
                    error_detail = e.response.json().get("detail", e.response.text)
                except Exception:
                    pass
                error_message = f"Error from backend API: {e.response.status_code}. Detail: {error_detail}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message, "sources": None, "processing_time": None})

            except requests.exceptions.RequestException as e:
                # Handle network/connection errors
                error_message = f"Error: Could not connect to the backend API. Please ensure it is running. Details: {e}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message, "sources": None, "processing_time": None})

