# =============================================================================
# INFORMATION RETRIEVAL USING MACHINE LEARNING
# Root Cause Analysis of Public Grievances
# =============================================================================
# Results (For one Department):
#   LDA Mallet  : Coherence score = 0.62 | Optimal topics k = 8
#   VADER       : 47% negative (true grievances) | 53% positive/neutral
#   Top 2 causes: Account for 60.38% of all grievances
#
# Usage:
#   pip install -r requirements.txt
#   python information_retrieval_nlp.py
#
# DATA NOTE:
#   The original dataset is confidential data (NIC / CPGRAMS).
#   It is NOT included in this repository.
#   To run this pipeline, provide your own Excel file with a
#   'GrievanceDescription' column. See README.md for format details.
# =============================================================================

# -----------------------------------------------------------------------------
# 1. Install & Import Libraries
# -----------------------------------------------------------------------------
import re
import os
import numpy as np
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt
import warnings
import logging

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spaCy for lemmatisation
import spacy

# NLTK
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# pyLDAvis for visualisation
import pyLDAvis
import pyLDAvis.gensim_models

# VADER for sentiment analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.ERROR
)
warnings.filterwarnings("ignore", category=DeprecationWarning)

os.makedirs('outputs', exist_ok=True)

# -----------------------------------------------------------------------------
# 2. Configuration
# -----------------------------------------------------------------------------

# Path to your grievance data file (Excel format)
DATA_FILE = 'data/grievance_data.xlsx'

# Path to LDA Mallet binary
# Download from: http://mallet.cs.umass.edu/
# Windows: 'C:/mallet-2.0.8/bin/mallet.bat'
# Linux/Mac: '/home/user/mallet-2.0.8/bin/mallet'
MALLET_PATH = 'path/to/mallet-2.0.8/bin/mallet'

# Number of topics to test for optimal k selection
K_MIN = 2
K_MAX = 25

# Output file name for dominant topic results
OUTPUT_FILE = 'outputs/topic_output.xlsx'

# -----------------------------------------------------------------------------
# 3. Load Data
# -----------------------------------------------------------------------------
print("Loading data...")
data_raw = pd.read_excel(DATA_FILE)

print(f"Raw dataset shape : {data_raw.shape}")
print(f"Columns           : {list(data_raw.columns)}")

# Keep only required columns
# Adjust column names to match your dataset
data = data_raw.filter(['RegistrationNo', 'GrievanceDescription', 'State'])
print(f"\nFiltered shape    : {data.shape}")

# -----------------------------------------------------------------------------
# 4. Data Cleaning
# -----------------------------------------------------------------------------
print("\nCleaning data...")

# 4a. Drop missing values
data.dropna(subset=['GrievanceDescription'], inplace=True)
print(f"After dropping NA : {data.shape}")

# 4b. Drop duplicates
data.drop_duplicates(subset=['GrievanceDescription'], inplace=True)
print(f"After dedup       : {data.shape}")

# 4c. Remove regional language rows — keep only ASCII-detectable English text
def is_english(text):
    """Return True if text contains sufficient ASCII characters to be English."""
    try:
        text.encode('ascii')
        return True
    except UnicodeEncodeError:
        ascii_ratio = sum(1 for c in text if ord(c) < 128) / len(text)
        return ascii_ratio > 0.85

data = data[data['GrievanceDescription'].apply(is_english)]
print(f"After language filter : {data.shape}")

# 4d. Clean text — remove emails, newlines, extra spaces
def clean_text(text):
    """Remove emails, newlines, URLs and extra whitespace."""
    text = str(text)
    text = re.sub(r'\S+@\S+', '', text)          # Remove emails
    text = re.sub(r'http\S+|www\S+', '', text)    # Remove URLs
    text = re.sub(r'\n', ' ', text)               # Remove newlines
    text = re.sub(r'\s+', ' ', text)              # Remove extra spaces
    return text.strip()

data['GrievanceDescription'] = data['GrievanceDescription'].apply(clean_text)
data1 = data['GrievanceDescription'].tolist()
print(f"\nFinal clean dataset : {len(data1)} grievances")

# =============================================================================
# SECTION A: VADER SENTIMENT ANALYSIS
# Run first to filter genuine negative grievances before topic modelling
# =============================================================================
print("\n" + "=" * 55)
print("SECTION A: VADER SENTIMENT ANALYSIS")
print("=" * 55)

analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    """Classify text as Positive, Negative, or Neutral using VADER."""
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.05:
        return 'Positive'
    elif compound <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

data['Sentiment'] = data['GrievanceDescription'].apply(get_sentiment)

# Sentiment distribution
sentiment_counts = data['Sentiment'].value_counts(normalize=True) * 100
print("\nSentiment Distribution:")
print(sentiment_counts.round(2))

# Visualise sentiment distribution
plt.figure(figsize=(8, 5))
sentiment_counts.plot(kind='bar',
                      color=['steelblue', 'coral', 'seagreen'],
                      edgecolor='black')
plt.title('Sentiment Distribution of Grievances (VADER)')
plt.xlabel('Sentiment')
plt.ylabel('Percentage (%)')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('outputs/sentiment_distribution.png', dpi=150)
plt.show()

# Filter to negative grievances only for topic modelling
# Negative = true grievances; positive often contains suggestions
data_negative = data[data['Sentiment'] == 'Negative']
data1_negative = data_negative['GrievanceDescription'].tolist()

print(f"\nNegative grievances : {len(data1_negative)}")
print(f"Percentage negative : {len(data1_negative)/len(data1)*100:.1f}%")
print("\nProceeding with negative grievances for LDA Mallet topic modelling...")

# =============================================================================
# SECTION B: NLP PRE-PROCESSING
# =============================================================================
print("\n" + "=" * 55)
print("SECTION B: NLP PRE-PROCESSING")
print("=" * 55)

# Extended stop words
stop_words = stopwords.words('english')
stop_words.extend([
    'from', 'subject', 're', 'edu', 'use', 'sir', 'madam',
    'dear', 'kindly', 'please', 'would', 'could', 'also',
    'regards', 'thank', 'respectfully', 'request'
])

def sent_to_words(sentences):
    """Tokenise sentences into word lists."""
    for sentence in sentences:
        yield gensim.utils.simple_preprocess(str(sentence), deacc=True)

data_words = list(sent_to_words(data1_negative))

# Build bigrams and trigrams
bigram  = gensim.models.Phrases(data_words, min_count=5, threshold=100)
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

bigram_mod  = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def remove_stopwords(texts):
    return [[w for w in simple_preprocess(str(doc)) if w not in stop_words]
            for doc in texts]

def lemmatisation(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """Lemmatise words keeping only specified POS tags."""
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([
            token.lemma_ for token in doc
            if token.pos_ in allowed_postags
        ])
    return texts_out

# Apply pipeline
data_words_nostops   = remove_stopwords(data_words)
data_words_bigrams   = make_bigrams(data_words_nostops)
data_words_lemmatised = lemmatisation(data_words_bigrams,
                                      allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

print(f"Sample processed tokens (first doc):\n{data_words_lemmatised[0][:15]}")

# Build dictionary and corpus
id2word = corpora.Dictionary(data_words_lemmatised)
corpus  = [id2word.doc2bow(text) for text in data_words_lemmatised]

print(f"\nDictionary size : {len(id2word)} unique tokens")
print(f"Corpus size     : {len(corpus)} documents")

# =============================================================================
# SECTION C: OPTIMAL K SELECTION (COHERENCE SCORE)
# =============================================================================
print("\n" + "=" * 55)
print("SECTION C: OPTIMAL K SELECTION")
print("=" * 55)

def compute_coherence_values(corpus, dictionary, texts,
                              mallet_path, k_min, k_max, step=1):
    """
    Compute coherence scores for LDA Mallet models across a range of k values.
    Returns list of models and coherence values.
    """
    coherence_values = []
    model_list = []

    for num_topics in range(k_min, k_max + 1, step):
        model = gensim.models.wrappers.LdaMallet(
            mallet_path,
            corpus          = corpus,
            num_topics      = num_topics,
            id2word         = id2word,
            iterations      = 1000,
            random_seed     = 42
        )
        model_list.append(model)

        coherence_model = CoherenceModel(
            model   = model,
            texts   = texts,
            dictionary = dictionary,
            coherence  = 'c_v'
        )
        coherence_values.append(coherence_model.get_coherence())
        print(f"  k={num_topics:2d} | coherence={coherence_values[-1]:.4f}")

    return model_list, coherence_values

print(f"\nTesting k from {K_MIN} to {K_MAX}...")
print("(This may take several minutes depending on dataset size)")

model_list, coherence_values = compute_coherence_values(
    corpus       = corpus,
    dictionary   = id2word,
    texts        = data_words_lemmatised,
    mallet_path  = MALLET_PATH,
    k_min        = K_MIN,
    k_max        = K_MAX
)

# Plot coherence scores
x = range(K_MIN, K_MAX + 1)
plt.figure(figsize=(10, 5))
plt.plot(x, coherence_values, marker='o', color='steelblue', lw=2)
plt.xlabel('Number of Topics (k)')
plt.ylabel('Coherence Score (c_v)')
plt.title('LDA Mallet — Coherence Score vs Number of Topics')
plt.xticks(x)
plt.tight_layout()
plt.savefig('outputs/coherence_score_plot.png', dpi=150)
plt.show()

# Select optimal k
optimal_k   = x[coherence_values.index(max(coherence_values))]
optimal_model = model_list[coherence_values.index(max(coherence_values))]
best_coherence = max(coherence_values)

print(f"\nOptimal k           : {optimal_k}")
print(f"Best coherence score: {best_coherence:.4f}")

# =============================================================================
# SECTION D: FINAL LDA MALLET MODEL
# =============================================================================
print("\n" + "=" * 55)
print("SECTION D: FINAL LDA MALLET MODEL")
print("=" * 55)

# Print topics and keywords
print(f"\nTopics extracted (k={optimal_k}):\n")
topics = optimal_model.show_topics(num_topics=optimal_k,
                                    num_words=10,
                                    formatted=True)
for topic in topics:
    print(f"  Topic {topic[0]}: {topic[1]}")

# pyLDAvis visualisation
print("\nGenerating pyLDAvis interactive visualisation...")
vis = pyLDAvis.gensim_models.prepare(
    gensim.models.wrappers.ldamallet.malletmodel2ldamodel(optimal_model),
    corpus,
    id2word,
    mds='mmds'
)
pyLDAvis.save_html(vis, 'outputs/lda_visualisation.html')
print("Saved: outputs/lda_visualisation.html")

# =============================================================================
# SECTION E: DOMINANT TOPIC PER GRIEVANCE
# =============================================================================
print("\n" + "=" * 55)
print("SECTION E: DOMINANT TOPIC ASSIGNMENT")
print("=" * 55)

def format_topics_sentences(ldamodel, corpus, texts):
    """Assign dominant topic and contribution score to each document."""
    sent_topics_df = pd.DataFrame()

    for i, row in enumerate(ldamodel[corpus]):
        row_sorted = sorted(row, key=lambda x: x[1], reverse=True)
        for j, (topic_num, prop_topic) in enumerate(row_sorted):
            if j == 0:
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(
                    pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]),
                    ignore_index=True
                )
            else:
                break

    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return sent_topics_df

df_topic_sents_keywords = format_topics_sentences(
    ldamodel = optimal_model,
    corpus   = corpus,
    texts    = data1_negative
)

df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = [
    'Document_No', 'Dominant_Topic',
    'Topic_Perc_Contrib', 'Keywords', 'Text'
]

print(f"\nDominant topic assigned to {len(df_dominant_topic)} grievances")
print(df_dominant_topic[['Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords']].head())

# Topic frequency distribution
topic_counts = df_dominant_topic['Dominant_Topic'].value_counts()
topic_pct    = (topic_counts / len(df_dominant_topic) * 100).round(2)

print("\nTopic Distribution (%):")
print(topic_pct)

# Pareto chart of root cause frequency
plt.figure(figsize=(10, 6))
cumulative = topic_pct.sort_values(ascending=False).cumsum()
ax1 = topic_pct.sort_values(ascending=False).plot(
    kind='bar', color='steelblue', edgecolor='black')
ax2 = ax1.twinx()
cumulative.sort_values().sort_index().reindex(
    topic_pct.sort_values(ascending=False).index
).plot(ax=ax2, color='red', marker='o', lw=2)
ax1.set_xlabel('Topic (Root Cause)')
ax1.set_ylabel('Count')
ax2.set_ylabel('Cumulative %')
plt.title('Pareto Chart — Root Cause Topics')
plt.tight_layout()
plt.savefig('outputs/pareto_chart.png', dpi=150)
plt.show()

# =============================================================================
# SECTION F: EXPORT RESULTS
# =============================================================================
print("\n" + "=" * 55)
print("SECTION F: EXPORT")
print("=" * 55)

# NOTE: The export strips actual grievance text for privacy
# Only topic assignments and keywords are exported
export_df = df_dominant_topic[['Document_No', 'Dominant_Topic',
                                 'Topic_Perc_Contrib', 'Keywords']]
export_df.to_excel(OUTPUT_FILE, index=False)
print(f"Topic assignments exported: {OUTPUT_FILE}")
print("Note: Grievance text not exported — data is confidential.")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 55)
print("PIPELINE COMPLETE — SUMMARY")
print("=" * 55)
print(f"Input grievances    : {len(data1)}")
print(f"After cleaning      : {len(data1)}")
print(f"Negative (true)     : {len(data1_negative)} ({len(data1_negative)/len(data1)*100:.1f}%)")
print(f"Optimal topics (k)  : {optimal_k}")
print(f"Best coherence score: {best_coherence:.4f}")
print(f"\nOutputs saved to: outputs/")
print("  ├── sentiment_distribution.png")
print("  ├── coherence_score_plot.png")
print("  ├── lda_visualisation.html  (interactive)")
print("  ├── pareto_chart.png")
print("  └── topic_output.xlsx       (topic assignments only)")
print("=" * 55)
print("\nNext step: Open topic_output.xlsx, read keywords per topic,")
print("and assign human-readable root cause names to each topic.")
print("Then build Pareto + bubble charts in Tableau for reporting.")
print("=" * 55)
