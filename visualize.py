# !pip install plotly
# !pip install gensim
# !pip install pyLDAvis
# !pip3 install tensorflow
# import visulizatoin using Plotly library
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
from plotly.subplots import make_subplots

# For data visualization
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this

# To view the most frequent words
from wordcloud import WordCloud
import seaborn

# Timer to track the run time or completeion of execution
from tqdm import tqdm

# Data Manipulation
import pandas as pd
import numpy as np

# To use regular expression
import re

# To find frequency of a token occuring in corpus
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Used different type of containers to store collection
from collections import Counter, defaultdict
import itertools
import collections

# use algo related to NLP
import nltk
from nltk import bigrams
from nltk import ngrams
from nltk.corpus import stopwords

# Lematization
import spacy

# To visualize data in graph format
import networkx as nx

# Decode emoji's into text
# !pip install demoji
import demoji

# Gensim
# import gensim
# import gensim.corpora as corpora
# from gensim.utils import simple_preprocess
# from gensim.models import CoherenceModel
df = pd.read_excel(r"data.xlsx")

print(df.head(10))
import plotly.graph_objs as go
import plotly.offline as py

# Calculate word frequencies
all_words = df['text'].str.split(expand=True).unstack().value_counts()

# # Create a bar chart
# data = [
#     go.Bar(
#         x=all_words.index[:50],
#         y=all_words.values[:50],
#         marker=dict(colorscale='Jet', color=all_words.values[:50]),
#         text='Word counts'
#     )
# ]

# layout = go.Layout(
#     title='Top 50 (Uncleaned) Word Frequencies in the Training Dataset',
#     xaxis=dict(title='Words'),
#     yaxis=dict(title='Frequency')
# )

# fig = go.Figure(data=data, layout=layout)

# # Plot the chart
# py.plot(fig, filename='basic-bar.html')
def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color='black',
        max_words=200,
        max_font_size=40,
        scale=1,
        random_state=1
).generate(" ".join(data))

    fig = plt.figure(1, figsize=(15, 15))
    plt.axis('off')
    if title:
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()
# Convert float values to strings
df['text'] = df['text'].astype(str)

# Show word cloud
show_wordcloud(df["text"].values)
plt.style.use('seaborn')
plt.figure(figsize=(10, 6))

# Bar plot
ax = sns.countplot(data=df, x='rating')

# Annotate count values on the bar plot
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.0f'),
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha = 'center', va = 'center',
                   xytext = (0, 9),
                   textcoords = 'offset points')

plt.title('Rating Distribution')
plt.xlabel('Rating')
plt.ylabel('Count')

plt.show()

# Define your functions
def word_tokenizer(text):
    text = text.lower()
    text = re.sub(r'[^\w\sÀ-ÿ]', ' ', text)
    text = re.sub("\'", "", text)
    text = text.split()
    return text
# label encode
def label_encode(x):
    if x == 1 or x == 2:
        return 0
    if x == 3:
        return 1
    if x == 5 or x == 4:
        return 2

# label to name
def label2name(x):
    if x == 0:
        return "Negative"
    if x == 1:
        return "Neutral"
    if x == 2:
        return "Positive"
# encode label and mapping label name
df["label"] = df["rating"].apply(lambda x: label_encode(x))
df["label_name"] = df["label"].apply(lambda x: label2name(x))
print(df.head(10))
import re

# calculate char count for each review
df['char_count'] = df['text'].apply(lambda x: len(str(x)))


def plot_dist3(df, feature, title):
    fig = plt.figure(constrained_layout=True, figsize=(18, 8))
    grid = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)
    ax1 = fig.add_subplot(grid[0, :2])
    ax1.set_title('Histogram')
    sns.distplot(df.loc[:, feature],
                 hist=True,
                 kde=True,
                 ax=ax1,
                 )
    ax1.set(ylabel='Frequency')
    ax1.xaxis.set_major_locator(MaxNLocator(nbins=20))


    plt.suptitle(f'{title}', fontsize=24)
plot_dist3(df, 'char_count', 'Characters Count in Data')
# Creating a new feature for the visualization.
df['Character Count'] = df['text'].apply(lambda x: len(str(x)))


def plot_dist3(df, feature, title):
    fig = plt.figure(constrained_layout=True, figsize=(24, 12))
    grid = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)

    # Customizing the histogram grid.
    ax1 = fig.add_subplot(grid[0, :2])
    # Set the title.
    ax1.set_title('Histogram')
    # plot the histogram.
    sns.distplot(df.loc[:, feature],
                 hist=True,
                 kde=True,
                 ax=ax1,
                 color='#e74c3c')
    ax1.set(ylabel='Frequency')
    
    ax1.xaxis.set_major_locator(MaxNLocator(nbins=20))

    # Customizing the ecdf_plot.
    ax2 = fig.add_subplot(grid[1, :2])
    # Set the title.
    ax2.set_title('Empirical CDF')
    # Plotting the ecdf_Plot.
    sns.distplot(df.loc[:, feature],
                 ax=ax2,
                 kde_kws={'cumulative': True},
                 hist_kws={'cumulative': True},
                 color='#e74c3c')
    ax2.xaxis.set_major_locator(MaxNLocator(nbins=20))
    ax2.set(ylabel='Cumulative Probability')

    plt.suptitle(f'{title}', fontsize=24)
plot_dist3(df[df['label'] == 0], 'Character Count', 'Characters Count "Negative" Content')
plot_dist3(df[df['label'] == 2], 'Character Count', 'Characters Count "Positive" Content')
plot_dist3(df[df['label'] == 1], 'Character Count', 'Characters Count "Neutral" Content')