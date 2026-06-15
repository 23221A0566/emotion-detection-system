import streamlit as st
import torch
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ==========================
# Load Tokenizer and Model
# ==========================

tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased"
)

model = AutoModelForSequenceClassification.from_pretrained(
    "."
)

# ==========================
# Emotion Labels
# ==========================

emotion_cols = [
    'admiration', 'amusement', 'anger', 'annoyance',
    'approval', 'caring', 'confusion', 'curiosity',
    'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement',
    'fear', 'gratitude', 'grief', 'joy', 'love',
    'nervousness', 'optimism', 'pride',
    'realization', 'relief', 'remorse',
    'sadness', 'surprise', 'neutral'
]

# ==========================
# Page Config
# ==========================

st.set_page_config(
    page_title="Emotion Detection System",
    page_icon="😊",
    layout="wide"
)

# ==========================
# Sidebar
# ==========================

st.sidebar.header("📌 Model Information")

st.sidebar.write("""
**Model:** DistilBERT

**Task:** Multi-label Emotion Detection

**Framework:** Hugging Face Transformers

**Developer:** k saleem
""")

# ==========================
# History Storage
# ==========================

if "history" not in st.session_state:
    st.session_state.history = []

# ==========================
# Main Title
# ==========================

st.title("😊 Emotion Detection System")

st.write(
    "Analyze emotions from text using DistilBERT"
)

# ==========================
# Input Text
# ==========================

text = st.text_area(
    "Enter Text",
    height=150,
    placeholder="Type your text here..."
)

# ==========================
# Analyze Button
# ==========================

if st.button("🔍 Analyze Emotion"):

    if text.strip() == "":
        st.warning(
            "Please enter some text."
        )

    else:

        # ==========================
        # Tokenization
        # ==========================

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        # ==========================
        # Prediction
        # ==========================

        outputs = model(**inputs)

        probs = torch.sigmoid(
            outputs.logits
        )[0]

        scores = probs.detach().numpy()

        # ==========================
        # DataFrame
        # ==========================

        results = pd.DataFrame({
            "Emotion": emotion_cols,
            "Score": scores
        })

        results = results.sort_values(
            by="Score",
            ascending=False
        )

        # ==========================
        # Main Emotion
        # ==========================

        top_emotion = results.iloc[0]["Emotion"]
        top_score = results.iloc[0]["Score"] * 100

        st.success(
            f"🎯 Predicted Emotion: {top_emotion.upper()} ({top_score:.2f}%)"
        )

        # ==========================
        # Top 5 Emotions
        # ==========================

        st.subheader("🏆 Top 5 Emotions")

        for _, row in results.head(5).iterrows():

            st.write(
                f"✅ {row['Emotion']} : {row['Score']*100:.2f}%"
            )

        # ==========================
        # Bar Chart
        # ==========================

        st.subheader(
            "📊 Emotion Confidence Scores"
        )

        chart_df = results.set_index(
            "Emotion"
        )

        st.bar_chart(chart_df)

        # ==========================
        # Pie Chart
        # ==========================

        st.subheader(
            "🥧 Emotion Distribution"
        )

        fig, ax = plt.subplots(
            figsize=(8, 8)
        )

        ax.pie(
            results.head(10)["Score"],
            labels=results.head(10)["Emotion"],
            autopct="%1.1f%%"
        )

        st.pyplot(fig)

        # ==========================
        # Save History
        # ==========================

        st.session_state.history.append({
            "Text": text[:50],
            "Emotion": top_emotion,
            "Confidence": round(
                top_score, 2
            )
        })

        # ==========================
        # History Table
        # ==========================

        st.subheader(
            "🕒 Analysis History"
        )

        history_df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            history_df,
            use_container_width=True
        )

        # ==========================
        # Full Results Table
        # ==========================

        st.subheader(
            "📋 All Emotion Scores"
        )

        results["Score"] = (
            results["Score"] * 100
        ).round(2)

        st.dataframe(
            results,
            use_container_width=True
        )

        # ==========================
        # Download CSV
        # ==========================

        csv = results.to_csv(
            index=False
        )

        st.download_button(
            label="📥 Download Report",
            data=csv,
            file_name="emotion_report.csv",
            mime="text/csv"
        )

        # ==========================
        # Statistics
        # ==========================

        st.subheader(
            "📈 Statistics"
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Top Emotion",
                top_emotion
            )

        with col2:
            st.metric(
                "Confidence",
                f"{top_score:.2f}%"
            )