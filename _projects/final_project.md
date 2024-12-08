import os

os.system("pip install matplotlib")
os.system("pip install nltk")
os.system("pip install wordcloud")
os.system("pip install collections")
os.system("pip install numpy")
os.system("pip install praw")
os.system("pip install json")
os.system("pip install altair")
os.system("pip install panel")

import streamlit as st
from streamlit.runtime.state import SessionState

st.set_page_config(layout="wide", page_title="AskReddit Data Explorer")

import os
import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import nltk
import altair as alt
from datetime import datetime
import praw

st.markdown("""
<style>
    .reportview-container {
        background: #EFF7FF;  /* light bg */
    }
    .main {
        background: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    h1 {
        color: #FF4500;  /* orangered */
        text-align: center;
        padding-bottom: 20px;
        border-bottom: 3px solid #FF4500;
        margin-bottom: 30px;
    }
    h2 {
        color: #336699;  /* ui text */
        border-bottom: 1px solid #CEE3F8;  /* header */
        padding-bottom: 10px;
    }
    .stButton>button {
        background-color: #FF8b60;  /* lighter reddit orange */
        color: white;
        border: none;
        border-radius: 4px;
        padding: 8px 16px;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #FFA07A;  /* even lighter shade */
        box-shadow: 0 2px 4px rgba(255, 139, 96, 0.2);
    }
    .stButton>button:active {
        background-color: #FF8b60;  /* return to original shade when clicked */
    }
</style>
""", unsafe_allow_html=True)

# Streamlit App Title
st.title("AskReddit Dashboard")

# Download NLTK resources
nltk.download('vader_lexicon', quiet=True)
nltk.download('stopwords', quiet=True)

# Initialize the Reddit API client
reddit = praw.Reddit(
    client_id="dFhdA1_6NWavjmAEt83r2A",
    client_secret="l4UsnweG9HdHLffJ9ZbAow0-HRocWA",
    user_agent="DataExplorer"
)

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_askreddit_posts(limit=100):
    subreddit = reddit.subreddit("AskReddit")
    posts = []
    for post in subreddit.hot(limit=limit):
        posts.append({
            'title': post.title,
            'score': post.score,
            'id': post.id,
            'url': post.url,
            'num_comments': post.num_comments,
            'num_crossposts': post.num_crossposts,
            'created_utc': datetime.fromtimestamp(post.created_utc),
            'author': str(post.author),
            'upvote_ratio': post.upvote_ratio,
            'day_of_week': datetime.fromtimestamp(post.created_utc).strftime('%A'),
            'hour': datetime.fromtimestamp(post.created_utc).hour
        })
    return pd.DataFrame(posts)

def refresh_app():
    for key in list(st.session_state.keys()):
        if key != 'df': 
            del st.session_state[key]
    st.rerun()

if st.button("Refresh App"):
    refresh_app()

df = fetch_askreddit_posts()

row1_col1, row1_col2 = st.columns(2)
row2_col1, row2_col2 = st.columns(2)

# Plot 1: Number of Posts vs Time of Day
with row1_col1:
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    st.subheader("Number of Posts vs Time of Day")
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    if 'selected_day' not in st.session_state:
        st.session_state.selected_day = 'Monday'
    selected_day = st.selectbox('Select Day of the Week:', days_order, key='selected_day')
    plot_data = df[df['day_of_week'] == selected_day].groupby('hour').size().reset_index(name='count')
    
    chart = alt.Chart(plot_data).mark_line(point=True, color='#FF4500').encode(
        x=alt.X('hour:Q', title='Hour of Day'),
        y=alt.Y('count:Q', title='Number of Posts'),
        tooltip=['hour', 'count']
    ).properties(title=f'Posts on {selected_day}', height=300)
    
    st.altair_chart(chart, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Plot 2: Scatter Plot: Comments vs Score
with row1_col2:
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    st.subheader("Scatter Plot of Comments or Crossposts vs Score")
    
    if 'scatter_plot_type' not in st.session_state:
        st.session_state.scatter_plot_type = 'num_comments'
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Number of Comments"):
            st.session_state.scatter_plot_type = 'num_comments'
    with col2:
        if st.button("Number of Crossposts"):
            st.session_state.scatter_plot_type = 'num_crossposts'
    
    x_axis = st.session_state.scatter_plot_type
    scatter_chart = alt.Chart(df).mark_circle(color='#FF4500', opacity=0.7).encode(
    x=alt.X(f'{x_axis}:Q', title=x_axis.replace('_', ' ').title()),
    y=alt.Y('score:Q', title='Score'),
    tooltip=['title', x_axis, 'score']
    ).properties(
    height=300,
    title=f'Correlation between {x_axis.replace("_", " ").title()} and Score'
    )

    
    st.altair_chart(scatter_chart, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Plot 3: Word Cloud by Sentiment
with row2_col1:
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    st.subheader("Word Cloud by Sentiment")
    analyzer = SentimentIntensityAnalyzer()
    df['sentiment'] = df['title'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
    df['sentiment_category'] = df['sentiment'].apply(
        lambda score: 'positive' if score > 0.05 else 'negative' if score < -0.05 else 'neutral'
    )
    if 'selected_sentiment' not in st.session_state:
        st.session_state.selected_sentiment = 'positive'
    selected_sentiment = st.selectbox("Select Sentiment", ['positive', 'negative', 'neutral'], key='selected_sentiment')
    filtered_titles = " ".join(df[df['sentiment_category'] == selected_sentiment]['title'])
    
    wordcloud = WordCloud(
        width=400,
        height=150,
        background_color='white',
        colormap='viridis',
    ).generate(filtered_titles)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)


# Plot 4: Author Performance Bubble Chart
with row2_col2:
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    st.subheader("Author Performance Bubble Chart")
    bubble_chart = alt.Chart(df).mark_circle().encode(
    x=alt.X('num_comments:Q', title='Number of Comments'),
    y=alt.Y('score:Q', title='Score'),
    size=alt.Size('upvote_ratio:Q', title='Upvote Ratio', scale=alt.Scale(range=[20, 200])),
    color=alt.Color('upvote_ratio:Q', scale=alt.Scale(scheme='reds'), legend=None),
    tooltip=['author', 'num_comments', 'score', 'upvote_ratio']
    ).properties(
        width=600,
        height=350
    )

    st.altair_chart(bubble_chart, use_container_width=False)
    st.markdown('</div>', unsafe_allow_html=True)

# Display raw data
if st.checkbox("Show Raw Data"):
    st.write(df)

# Summary
st.markdown('<div class="plot-container">', unsafe_allow_html=True)
st.markdown("### Project Summary")
st.markdown("This project used dataset from the r/AskReddit subreddit. It includes various fields such as post scores, the number of comments, upvote ratios, and timestamps, providing a comprehensive view of post performance and user engagement. The dataset was explored through several interactive visualizations designed to reveal key insights effectively. A line graph was used to display the number of posts by the hour for each day of the week, with a dropdown menu allowing users to select specific days, making it easy to identify and compare posting patterns such as peak activity hours. A scatter plot illustrated the correlation between post scores and other engagement metrics like the number of comments and crossposts, with interactive radio buttons enabling users to switch the x-axis variable and examine relationships influencing post success. A word cloud incorporating sentiment analysis enriched the exploration by categorizing post titles as positive, negative, or neutral using VADER sentiment analysis. Words were color-coded based on their sentiment, and users could toggle between categories and explore different color palettes, making the visualization both informative and visually appealing. Lastly, a bubble chart provided a unique perspective on individual author performance, with bubbles representing authors, their size reflecting the upvote ratio, and their position showing the number of comments and post scores, offering an engaging way to analyze user contributions.")
st.markdown('</div>', unsafe_allow_html=True)

# Group Members:
st.markdown("### Group Members")
st.markdown("1. Cindy Chung")
st.markdown("2. Irith Chaturvedi")
st.markdown("3. Ning Gao")
st.markdown("4. Pia Schwarzinger")
st.markdown("5. Tanvi Lakhani")
