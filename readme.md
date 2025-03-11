
# Speech Analysis Application

This repository contains an interactive web application designed for comprehensive analysis of speeches. The application allows users to explore various insights such as sentiment analysis, named entity recognition (NER), word clouds, and more. By selecting a speech transcript, users can delve into the prominent words, entities, and sentiments presented in the speech.

## Key Features
- **Interactive Word Cloud**: Provides a visual representation of the most frequently used words in the speech, allowing for quick identification of key themes.
  
- **Named Entity Recognition (NER)**: Automatically detects and categorizes key entities such as people, organizations, places, dates, and more from the speech. Users can see which entities are most frequently mentioned and explore detailed information about them.
  
- **Sentiment Analysis**: Analyzes the emotional tone of the speech, breaking it down into positive, negative, or neutral sentiments. Detailed insights are provided for each segment of the speech.

- **Entity-Specific Sentiment Distribution**: Displays the sentiment distribution specific to key figures or organizations mentioned in the speech, offering a nuanced understanding of how each entity is discussed.

- **Named Entity Statistics**: Users can explore the top mentioned entities (e.g., persons, organizations) in the speech, with the option to compare their occurrence.

## Installation and Setup
### Prerequisites:
Ensure that Python is installed on your system. 

### Step 1: Install Dependencies
To install the required libraries, run the following command:
```bash
pip install -r requirements.txt
```

### Step 2: Run the Application
To launch the application, execute:
```bash
streamlit run main.py
```

## How to Use the Application
1. **Select a Transcript**: Begin by selecting a transcript from the provided dropdown menu. The application will process the selected speech and display the corresponding insights.
  
2. **View Word Cloud**: The word cloud provides a visual summary of the most frequently occurring words in the selected speech. Larger words represent terms that are used more frequently.

3. **Analyze Named Entities**: Explore a detailed breakdown of the named entities (people, locations, organizations) mentioned in the speech. The application displays a closer look at these entities along with their sentiment context.

4. **Explore Sentiment Analysis**: Gain insights into the overall sentiment distribution across the speech. The app provides a breakdown of positive, negative, and neutral sentiments, allowing you to understand the emotional tone of the content.

5. **Entity Comparison**: The application offers a comparison of the most mentioned entities (such as individuals or organizations), helping users understand their prominence and context within the speech.

## Visual Insights 
- **Word Cloud**: Visualizes the most frequently used words in the speech, allowing users to grasp the main topics or themes at a glance.

- **Entity Mentions and Sentiment**: Understand the context in which different entities (like "Hillary Clinton," "Donald Trump," or "Obamacare") are discussed, including sentiment distribution for each.

- **Detailed Sentiment Distribution**: Users can see how sentiments vary across different sections of the speech and compare sentiment differences for key figures or topics.

## Insights to be Gained
- **Key Topics**: Easily identify the main topics and issues addressed in a speech through the word cloud and named entity recognition.
  
- **Speaker's Tone**: Understand whether the speech leans towards a positive, negative, or neutral tone using the sentiment analysis features.
  
- **Focus on Entities**: Gain an understanding of how much focus is given to specific entities (e.g., people, organizations) and how they are discussed emotionally.

## Example Use Cases
- **Political Speech Analysis**: Analyze debates or speeches for insights into the rhetoric and focus areas of political figures.
  
- **Public Figure Sentiment**: Explore how different public figures or entities are portrayed in speeches.
  
- **Thematic Insights**: Understand the key themes and emotional undertones of any given speech, helping in content evaluation or media analysis.
