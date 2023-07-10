# Infinite AI

## Streamlit app for Financial Chatbot

WIP

### Getting Started

This app uses Google Vertex AI using LangChain. The UI is built with Streamlit. 

1. Setup GCP credentials
2. Run the encoder.py
3. Start the app.py using Streamlit

Create or Copy your service account JSON file into the project directory.
Store the path to the JSON file as the  **GOOGLE_APPLICATION_CREDENTIALS** environment variable before starting this application.

In Linux terminal you can do so with the following command. *(Assuming your file is named service_account.json)*
```bash
export GOOGLE_APPLICATION_CREDENTIALS=./service_account.json
```

### Don't share or commit your service account JSON file to git to avoid potential misuse.

Just install streamlit and then you can run locally using the follow command
```bash
streamlit run app.py
```
