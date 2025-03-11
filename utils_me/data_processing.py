import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import streamlit as st

import nltk
import spacy
import os

def download_nltk_data():
    nltk_data = ['punkt', 'vader_lexicon', 'wordnet']
    for data in nltk_data:
        try:
            nltk.data.find(f'tokenizers/{data}')
        except LookupError:
            nltk.download(data, quiet=True)

def download_spacy_model():
    model_name = 'en_core_web_sm'
    if not spacy.util.is_package(model_name):
        spacy.cli.download(model_name, False, '--quiet')

download_nltk_data()
download_spacy_model()

#python -m spacy download en
from bs4 import BeautifulSoup

def remove_specific_divs(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    
    for div in soup.find_all("div", class_="stCodeBlock"):
        div.decompose()  

    return str(soup)

def pre_process_text(text):
    nltk.download('stopwords')
    nltk.download('wordnet')
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    text = text.lower()

    text = re.sub(r'\(\d{2}:\d{2}\)', ' ', text)

    text = ' '.join([word for word in text.split() if word.lower() not in stop_words])

    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

    text = ' '.join([word for word in text.split() if len(word) > 2])

    text = re.sub(r'\s+', ' ', text).strip()

    return text


def perform_wordcloud(transcript):
    from PIL import Image
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud
    from io import BytesIO
    """This function performs the following analysis on the transcript:
        i. Word count (top words; two and three word phrases)
        ii. Word cloud
        iii. Entity identification and sentiment analysis
        iv. Topic identification and sentiment analysis
        v. Sentiment analysis for sentences
    """
    text = pre_process_text(transcript)
    
    wordcloud = WordCloud(width=800, height=800, background_color='white', max_words=150).generate(text)
    
    buffer = BytesIO()
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    
    plt.savefig(buffer, format='png')
    plt.close()
    
    buffer.seek(0)
    
    img = Image.open(buffer)
    
    return img


# @st.cache_data
def perform_and_show_NER(transcript):
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
    import pandas as pd
    import spacy
    from spacy import displacy

    tokenizer = AutoTokenizer.from_pretrained("dslim/distilbert-NER")
    model = AutoModelForTokenClassification.from_pretrained("dslim/distilbert-NER")

    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

    nlp_spacy = spacy.load("en_core_web_sm")
    doc = nlp_spacy(transcript)

    entity_counts = {}
    for ent in doc.ents:
        if ent.label_ in entity_counts:
            entity_counts[ent.label_] += 1
        else:
            entity_counts[ent.label_] = 1

    html = displacy.render(doc, style="ent", page=True)

    fig = top_entities_ner(entity_counts, top_n=10)

    return html, fig



def top_entities_ner(entity_counts, top_n=10, output_filename='top_entities_graph_plotly.png'):
    import plotly.express as px
    import pandas as pd
    """
    Plots the top N named entities from a given entity count dictionary using Plotly.

    Args:
    - entity_counts (dict): A dictionary with entity types as keys and their counts as values.
    - top_n (int): Number of top entities to display in the plot. Default is 10.

    Returns:
    - fig (plotly.graph_objs._figure.Figure): The Plotly figure object.    
    """
    
    df = pd.DataFrame(list(entity_counts.items()), columns=['Entity', 'Count'])

    df_top = df.nlargest(top_n, 'Count')

    fig = px.bar(
        df_top,
        x='Entity',
        y='Count',
        text='Count',
        color='Entity',  
        color_discrete_sequence=px.colors.qualitative.Pastel,  
        title=f'Top {top_n} Named Entities mentioned in the Transcript'
    )

    fig.update_layout(
        title_font_size=16,
        xaxis_title='Entity Types',
        yaxis_title='Count',
        xaxis_tickangle=-45,
        xaxis_tickfont_size=12,
        yaxis_tickfont_size=12,
        title_pad=dict(t=20, b=20),
        margin=dict(t=50, b=50),
    )


    fig.update_traces(textposition='outside', textfont=dict(size=12))

    return fig





def top_persons_and_org(transcript, top_n=5):
    import spacy
    from collections import Counter
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    nlp = spacy.load("en_core_web_sm")

    entity_mapping = {
        "clinton": "Hillary Clinton",
        "hillary clinton": "Hillary Clinton",
        "hillary": "Hillary Clinton",
        "trump": "Donald Trump",
        "donald trump": "Donald Trump",
        "donald": "Donald Trump",
        "americans": "American",
        "american": "American",
        "republicans": "Republicans",
        "african-americans": "African-Americans",
    }

    doc = nlp(transcript)

    persons_entity = Counter()
    combined_org_norp_entity = Counter()

    for ent in doc.ents:
        normalized_text = entity_mapping.get(ent.text.lower(), ent.text)
        if ent.label_ == "PERSON":
            persons_entity[normalized_text] += 1
        elif ent.label_ in ["ORG", "NORP"]:
            combined_org_norp_entity[normalized_text] += 1


    top_n_persons = persons_entity.most_common(top_n)
    top_n_combined_org_norp = combined_org_norp_entity.most_common(top_n+2)

    entities = [person[0] for person in top_n_persons] + [org[0] for org in top_n_combined_org_norp]
    person_counts = [person[1] for person in top_n_persons] + [0] * top_n
    org_norp_counts = [0] * top_n + [org[1] for org in top_n_combined_org_norp]

    fig = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.02)

    fig.add_trace(
        go.Bar(x=entities, y=person_counts, name="PERSON", marker_color='skyblue'),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(x=entities, y=org_norp_counts, name="ORG/NORP", marker_color='lightgreen'),
        row=1, col=1
    )

    # Update layout
    fig.update_layout(
        title_text=f"Top {top_n} Entities Comparison",
        barmode='group',
        yaxis_title="Counts",
        xaxis_title="Entities",
        legend_title="Entity Type",
        font=dict(size=12),
        height=600,
        width=1000
    )

    fig.update_xaxes(tickangle=45)

    st.subheader(f"Top {top_n} PERSON entities with counts:")
    for person, count in top_n_persons:
        st.write(f"{person}: {count}")

    st.subheader(f"\nTop {top_n} combined ORG and NORP entities with counts:")
    for entity, count in top_n_combined_org_norp:
        st.write(f"{entity}: {count}")

    return fig


import re
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('vader_lexicon')

def process_sentiment_analysis_sentence(transcript):
    transcript = transcript.lower()
    transcript = re.sub(r'\(\d{2}:\d{2}\)', '', transcript)

    sia = SentimentIntensityAnalyzer()

    sentences = sent_tokenize(transcript)

    grouped_sentences = [" ".join(sentences[i:i+3]) for i in range(0, len(sentences), 3)]

    sentiments = []
    for group in grouped_sentences:
        sentiment = sia.polarity_scores(group)
        sentiments.append({
            'sentence_group': group,
            'sentiment': sentiment,
            'overall': 'positive' if sentiment['compound'] > 0 else 'negative' if sentiment['compound'] < 0 else 'neutral'
        })

    results_df = pd.DataFrame(sentiments)

    return results_df

def most_positive_n_negative(text):
    results_df = process_sentiment_analysis_sentence(text)


    st.write("## Sentiment Summary")
    st.write(results_df['overall'].value_counts(normalize=True))

    most_positive = results_df.loc[results_df['sentiment'].apply(lambda x: x['compound']).idxmax()]
    most_negative = results_df.loc[results_df['sentiment'].apply(lambda x: x['compound']).idxmin()]

    st.write("### Most Positive Sentence Group")
    compound = most_positive['sentiment']["compound"]
    if compound > 0:
        st.write(most_positive['sentence_group'])

        st.write("Positive with compound score: ", compound)

    st.write("### Most Negative Sentence Group")
    compound = most_negative['sentiment']["compound"]
    if compound < 0:
        st.write(most_negative['sentence_group'])
        st.write("Negative with compound score: ", compound)

    st.write("### Detailed Sentiment Analysis Results")
    st.dataframe(results_df.drop(columns='sentiment'))
import spacy

nlp = spacy.load("en_core_web_sm")

sia = SentimentIntensityAnalyzer()


# @st.cache_data
def perform_and_show_NER_with_sentiment(transcript):
    entity_mapping = {
        "joe": "Joe Biden",
        "biden": "Joe Biden",
        "joe biden": "Joe Biden",
        "trump": "Donald Trump",
        "donald trump": "Donald Trump",
        "donald": "Donald Trump",
        "americans": "American",
        "american": "American",
        "republicans": "Republicans",
        "democrats": "Democrats",
        "african-americans": "African-Americans",
        "clinton": "Hillary Clinton",
        "hillary clinton": "Hillary Clinton",
        "hillary": "Hillary Clinton",
        "united states": "USA",
        "america": "USA",
    }

    contractions = {
        "n't": " not",
        "'s": " is",
        "'m": " am",
        "'re": " are",
        "'ll": " will",
        "'ve": " have",
        "'d": " would"
    }
    for contraction, expansion in contractions.items():
        transcript = re.sub(r'\b' + re.escape(contraction) + r'\b', expansion, transcript, flags=re.IGNORECASE)

    transcript = re.sub(r'\(\d{2}:\d{2}\)', '', transcript)
    transcript = re.sub(r'\b(uh|um|eh|ah|oh)\b', '', transcript, flags=re.IGNORECASE)

    doc = nlp(transcript)
    sentences = sent_tokenize(transcript)
    entity_sentiment_data = []

    expanded_contractions = set(contractions.values())

    for ent in doc.ents:
        if (ent.label_ in ["PERSON", "ORG", "NORP", "GPE"] and 
            len(ent.text) > 2 and 
            ent.text.lower().strip() not in expanded_contractions):
            
            normalized_entity = entity_mapping.get(ent.text.lower(), ent.text)
            entity_type = ent.label_

            entity_sentences = [sentence for sentence in sentences if ent.text.lower() in sentence.lower()]

            for sentence in entity_sentences:
                sentiment = sia.polarity_scores(sentence)
                overall_sentiment = 'positive' if sentiment['compound'] > 0 else 'negative' if sentiment['compound'] < 0 else 'neutral'
                entity_sentiment_data.append({
                    'entity': normalized_entity,
                    'entity_type': entity_type,
                    'sentiment': overall_sentiment
                })

    entity_sentiment_df = pd.DataFrame(entity_sentiment_data)
    return entity_sentiment_df

import plotly.express as px



@st.cache_data
def visualize_sentiment_distribution_for_top_entities(entity_sentiment_df, top_n=2):
    entity_mapping = {
        "joe": "Joe Biden",
        "biden": "Joe Biden",
        "joe biden": "Joe Biden",
        "trump": "Donald Trump",
        "donald trump": "Donald Trump",
        "donald": "Donald Trump",
        "americans": "American",
        "american": "American",
        "republicans": "Republicans",
        "democrats": "Democrats",
        "african-americans": "African-Americans",
        "clinton": "Hillary Clinton",
        "hillary clinton": "Hillary Clinton",
        "hillary": "Hillary Clinton",
        "united states": "USA",
        "america": "USA",
        "n't": "",  # Map "n't" to an empty string
    }
    
    entity_sentiment_df['entity'] = entity_sentiment_df['entity'].apply(
        lambda entity_text: entity_mapping.get(entity_text.lower(), entity_text)
    )

    # Filter out empty entities and "n't"
    entity_sentiment_df = entity_sentiment_df[
        (entity_sentiment_df['entity'] != "") & 
        (entity_sentiment_df['entity'].str.lower() != "n't")
    ]

    entity_sentiment_counts = entity_sentiment_df.groupby(['entity', 'entity_type', 'sentiment']).size().reset_index(name='count')
    
    # Separate dataframes for each entity type
    entity_types = ['PERSON', 'ORG', 'NORP', 'GPE']
    
    for entity_type in entity_types:
        type_df = entity_sentiment_df[entity_sentiment_df['entity_type'] == entity_type]
        
        # Get top 2 entities for this type
        top_entities = type_df['entity'].value_counts().head(top_n).index.tolist()
        
        st.subheader(f"Top {top_n} {entity_type} Entities")
        
        for entity in top_entities:
            entity_data = entity_sentiment_counts[
                (entity_sentiment_counts['entity'] == entity) & 
                (entity_sentiment_counts['entity_type'] == entity_type)
            ]
            
            if not entity_data.empty:
                fig = px.pie(entity_data, values='count', names='sentiment', 
                             title=f'Sentiment Distribution for {entity} ({entity_type})',
                             color_discrete_sequence=['#636EFA', '#EF553B', '#00CC96'])
                
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(
                    height=500,
                    width=700,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )

                st.plotly_chart(fig)

                # Display entity counts
                total_mentions = entity_sentiment_df[
                    (entity_sentiment_df['entity'] == entity) & 
                    (entity_sentiment_df['entity_type'] == entity_type)
                ].shape[0]
                st.write(f"Total mentions of {entity}: {total_mentions}")
            else:
                st.write(f"No data available for {entity}")
        
        st.write("---")  # Add a separator between entity types

    # Display summary of analyzed entities
    st.subheader("Summary of Analyzed Entities")
    for entity_type in entity_types:
        type_df = entity_sentiment_df[entity_sentiment_df['entity_type'] == entity_type]
        top_entities = type_df['entity'].value_counts().head(top_n).index.tolist()
        st.write(f"Top {top_n} {entity_type} entities: {', '.join(top_entities)}")