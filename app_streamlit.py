import streamlit as st
from openai import OpenAI
import os

# ✅ Set your API key (either here or via environment variable)
OPENAI_API_KEY ="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"

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# ------------------- Streamlit UI -------------------
st.set_page_config(page_title="ChatGPT Academic Assistant", page_icon="📚")
st.title("📚 Academic Research Assistant")

st.markdown("Ask any research question and get an AI-generated answer instantly!")

# Input box
user_question = st.text_input("Enter your research question:")

if st.button("Get Answer") and user_question:
    try:
        # Call ChatGPT model
        response = client.chat.completions.create(
            model="gpt-4o-mini",   # you can use gpt-4o, gpt-4.1, or gpt-3.5-turbo too
            messages=[{"role": "user", "content": user_question}],
            temperature=0.5
        )

        # Extract answer
        answer = response.choices[0].message.content

        # Display result
        st.subheader("✅ Answer")
        st.write(answer)

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
