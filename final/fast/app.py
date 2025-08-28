import streamlit as st
import requests

# --- CONFIGURATION ---
API_URL = "http://localhost:8000/ask"
st.set_page_config(page_title="Pharmabot", layout="wide")
st.title(" Pharmabot:   Medication Assistant")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! How can I help you with your medication questions today?"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask about side effects, usage, etc."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get and display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        try:
            with st.spinner("Searching the database and generating an answer..."):
                response = requests.post(API_URL, json={"question": prompt})
                response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
                
                api_response = response.json()
                full_response = api_response.get("answer", "Sorry, I encountered an error.")
                
                # Display the sources as an expander
                sources = api_response.get("source_documents", [])
                with st.expander("View Retrieved Sources"):
                    for i, source in enumerate(sources):
                        st.write(f"**Source {i+1}:** {source.get('medication_name')} - *{source.get('source_column')}*")
                        st.caption(source.get('text'))

            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except requests.exceptions.RequestException as e:
            error_message = f"Error: Could not connect to the backend API. Please make sure it's running. Details: {e}"
            st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})
