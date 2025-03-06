import re
import os
import pickle
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Import custom functions from your module
from utils_me.data_processing import (
    pre_process_text, perform_wordcloud, perform_and_show_NER, 
    top_persons_and_org, most_positive_n_negative, remove_specific_divs, 
    process_sentiment_analysis_sentence, perform_and_show_NER_with_sentiment, 
    visualize_sentiment_distribution_for_top_entities
)

# Page configuration
st.set_page_config(layout="wide")
st.title("Speech Analysis App")

# -------------------------------------------------
# Optional: Preprocessing for topic modeling
# (this function is separate from pre_process_text)
def preprocess_text(text, debug=False):
    import spacy
    from gensim.utils import simple_preprocess
    from gensim.parsing.preprocessing import STOPWORDS
    from nltk.stem import WordNetLemmatizer

    nlp = spacy.load('en_core_web_sm')
    lemmatizer = WordNetLemmatizer()

    custom_stopwords = set(STOPWORDS).union({
        'look', 'lot', 'going', 'said', 'know', 'thing', 'well', 
        'actually', 'thank', 'us', 'think', 'just', 'people', 'country',
        've', 'don', 'll'
    })

    if isinstance(text, list):
        text = ' '.join(text)
    
    # Remove time stamps like (00:00)
    text = re.sub(r'\(\d{2}:\d{2}\)', '', text)
    text = text.lower()

    doc = nlp(text)
    # Remove tokens with certain entity types if needed (e.g. PERSON, GPE)
    text = ' '.join([token.text for token in doc if token.ent_type_ not in ['PERSON', 'GPE']])
    
    tokens = simple_preprocess(text, deacc=True)  # Remove punctuation
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in custom_stopwords]
    
    processed_text = ' '.join(tokens)
    return processed_text

# -------------------------------------------------
# Topic Modeling Functions
def train_topic_model(df, n_topics=25, debug=False):
    if debug:
        st.write(f"Number of documents: {len(df)}")
        st.write(f"Sample raw text (first 100 chars): {df['speech_in_text'].iloc[0][:100]}")
    
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english', ngram_range=(1, 2))
    preprocessed_docs = df['speech_in_text'].apply(lambda x: preprocess_text(x, debug=debug))
    
    if debug:
        st.write(f"Sample preprocessed text (first 100 chars): {preprocessed_docs.iloc[0][:100]}")
    
    try:
        doc_term_matrix = vectorizer.fit_transform(preprocessed_docs)
    except ValueError as e:
        st.error(f"Error in vectorization: {str(e)}")
        return None, None
    
    if len(vectorizer.vocabulary_) == 0:
        st.error("Error: Empty vocabulary. Please check your preprocessing steps.")
        return None, None
    
    if debug:
        st.write(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
        st.write(f"Top 10 words in vocabulary: {list(vectorizer.vocabulary_.keys())[:10]}")
    
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=10)
    lda.fit(doc_term_matrix)
    
    return vectorizer, lda

def get_topic_words(model, feature_names, n_top_words=10):
    topic_words = []
    for topic_idx, topic in enumerate(model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        topic_words.append(top_words)
    return topic_words

def save_model(vectorizer, lda, topic_words):
    with open('topic_model.pkl', 'wb') as f:
        pickle.dump((vectorizer, lda, topic_words), f)

@st.cache_resource
def load_model():
    with open('topic_model.pkl', 'rb') as f:
        return pickle.load(f)

# -------------------------------------------------
# File Upload and Data Loading
st.subheader("Upload Your Dataset")
uploaded_file = st.file_uploader("Choose a dataset file (CSV, Excel, or Pickle)", type=["csv", "xlsx", "pkl", "pickle"])

if uploaded_file is not None:
    file_extension = os.path.splitext(uploaded_file.name)[1]
    try:
        if file_extension == ".csv":
            df = pd.read_csv(uploaded_file)
        elif file_extension in [".xlsx"]:
            df = pd.read_excel(uploaded_file)
        elif file_extension in [".pkl", ".pickle"]:
            df = pd.read_pickle(uploaded_file)
        else:
            st.error("Unsupported file type.")
            df = None
    except Exception as e:
        st.error(f"Error loading file: {e}")
        df = None
else:
    df = None

if df is not None:
    if "speech_in_text" not in df.columns:
        st.error("The uploaded file does not contain a 'speech_in_text' column.")
    else:
        st.success("Dataset loaded successfully.")
        # Create a short preview text for each row (first 50 characters)
        df["display"] = df["speech_in_text"].apply(lambda x: x[:50] + "..." if len(x) > 50 else x)
        
        st.subheader("Select a Transcript")
        selected_index = st.selectbox("Choose a transcript for analysis", df.index, 
                                      format_func=lambda idx: df.loc[idx, "display"])
        text = df.loc[selected_index, "speech_in_text"]
        
        if text:
            # ---------------------------
            # Word Cloud
            st.header("Word Cloud")
            processed_text = pre_process_text(text)
            img = perform_wordcloud(text)
            st.image(img, caption='Word Cloud', use_column_width=True)

            # ---------------------------
            # Named Entity Recognition
            st.header("Named Entities: A Closer Look")
            html, fig = perform_and_show_NER(text)
            html = remove_specific_divs(html)
            scrollable_html = f"""
            <div style="height: 400px; overflow-y: scroll; border: 1px solid #ccc; padding: 10px;">
                {html}
            </div>
            """
            st.markdown(scrollable_html, unsafe_allow_html=True)
            st.plotly_chart(fig)

            # ---------------------------
            # Top Persons and Organizations
            st.subheader("Top Mentioned Persons and Organizations")
            fig = top_persons_and_org(text, top_n=5)
            st.plotly_chart(fig)

            # ---------------------------
            # Topic Modeling
            st.subheader("Topic Identification")
            if not os.path.exists('topic_model.pkl'):
                st.write("Training topic model on the full dataset...")
                vectorizer, lda = train_topic_model(df, debug=True)
                if vectorizer is not None and lda is not None:
                    feature_names = vectorizer.get_feature_names_out()
                    topic_words = get_topic_words(lda, feature_names)
                    save_model(vectorizer, lda, topic_words)
                else:
                    st.error("Failed to train topic model. Please check the debug information above.")
            else:
                st.write("Loading pre-trained topic model...")
                vectorizer, lda, topic_words = load_model()

            if vectorizer is not None and lda is not None:
                st.write("Key Topics for the Selected Transcript:")
                preprocessed_text_topic = preprocess_text(text)
                doc_term_matrix = vectorizer.transform([preprocessed_text_topic])
                doc_topics = lda.transform(doc_term_matrix)[0]
                
                relevance_threshold = 0.1
                relevant_topics = [(i, strength) for i, strength in enumerate(doc_topics) if strength > relevance_threshold]
                relevant_topics.sort(key=lambda x: x[1], reverse=True)
                
                if relevant_topics:
                    for topic_idx, strength in relevant_topics[:3]:
                        st.write(f"• {', '.join(topic_words[topic_idx][:5])} (Strength: {strength:.2f})")
                else:
                    st.write("No strong topics identified in the selected transcript.")

            # ---------------------------
            # Sentiment Analysis
            st.header("Sentiment Analysis")
            most_positive_n_negative(text)
            result_df = process_sentiment_analysis_sentence(text)

            st.subheader("Sentiment Distribution (Pie Chart)")
            sentiment_counts = result_df['overall'].value_counts()
            sentiment_data = pd.DataFrame({
                'Sentiment': sentiment_counts.index,
                'Count': sentiment_counts.values
            })
            custom_colors = ['#636EFA', '#EF553B', '#00CC96']
            fig_pie = px.pie(sentiment_data, values='Count', names='Sentiment',
                             title='Sentiment Distribution', color_discrete_sequence=custom_colors)
            st.plotly_chart(fig_pie)

            st.subheader("Named Entities with Sentiment Analysis")
            entity_sentiment_df = perform_and_show_NER_with_sentiment(text)

            st.subheader("Sentiment Distribution for Top Entities")
            visualize_sentiment_distribution_for_top_entities(entity_sentiment_df, top_n=2)
        else:
            st.warning("The selected transcript is empty.")
else:
    st.info("Please upload a dataset file to begin analysis.")
