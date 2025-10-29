# AI Feedback Categorizer

A Streamlit web app that uses OpenAI embeddings to categorize customer feedback based on your defined categories.

## Deployment

1. Push this folder to a GitHub repository.
2. Go to [https://share.streamlit.io](https://share.streamlit.io) and connect your GitHub account.
3. Set:
   - **Main file:** `ai_feedback_categorizer_app.py`
   - **Dependencies:** from `requirements.txt`
4. Add your OpenAI API key under **App Settings → Secrets**:
   ```ini
   OPENAI_API_KEY = "sk-your-key-here"
   ```
5. Deploy and enjoy!
