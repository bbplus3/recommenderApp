import streamlit as st
import numpy as np
import pandas as pd
import warnings
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import faiss
import torch
torch.classes.__path__ = []  # Neutralizes the path inspection
from transformers import AutoTokenizer, AutoModel
import gdown
import os
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"  # Disables problematic inspection
from diffusers import StableDiffusionPipeline
from huggingface_hub import hf_hub_download

# Suppress warnings
warnings.filterwarnings('ignore')

# Download necessary NLTK resources
nltk.download('stopwords')

# Streamlit Page Configuration
st.set_page_config(page_title="Bible Verse Recommender and Visualization", layout="wide")
##################################################################################################################
# Cache Data Loading
@st.cache_data
def load_data():
    data = pd.read_csv("t_bbe.csv").dropna()
    
    # Map book numbers to book names
    book_names = {
        1: 'Genesis', 2: 'Exodus', 3: 'Leviticus', 4: 'Numbers', 5: 'Deuteronomy',
        6: 'Joshua', 7: 'Judges', 8: 'Ruth', 9: '1 Samuel', 10: '2 Samuel',
        11: '1 Kings', 12: '2 Kings', 13: '1 Chronicles', 14: '2 Chronicles',
        15: 'Ezra', 16: 'Nehemiah', 17: 'Esther', 18: 'Job', 19: 'Psalms',
        20: 'Proverbs', 21: 'Ecclesiastes', 22: 'Song of Solomon', 23: 'Isaiah',
        24: 'Jeremiah', 25: 'Lamentations', 26: 'Ezekiel', 27: 'Daniel',
        28: 'Hosea', 29: 'Joel', 30: 'Amos', 31: 'Obadiah', 32: 'Jonah',
        33: 'Micah', 34: 'Nahum', 35: 'Habakkuk', 36: 'Zephaniah', 37: 'Haggai',
        38: 'Zechariah', 39: 'Malachi', 40: 'Matthew', 41: 'Mark', 42: 'Luke',
        43: 'John', 44: 'Acts', 45: 'Romans', 46: '1 Corinthians',
        47: '2 Corinthians', 48: 'Galatians', 49: 'Ephesians', 50: 'Philippians',
        51: 'Colossians', 52: '1 Thessalonians', 53: '2 Thessalonians',
        54: '1 Timothy', 55: '2 Timothy', 56: 'Titus', 57: 'Philemon',
        58: 'Hebrews', 59: 'James', 60: '1 Peter', 61: '2 Peter',
        62: '1 John', 63: '2 John', 64: '3 John', 65: 'Jude', 66: 'Revelation'
    }

    # Map book numbers to names
    data['Book Name'] = data['b'].map(book_names)

    # Process text: Remove stopwords
    stop_words = set(stopwords.words('english'))
    data['corpus'] = data['t'].astype(str).str.lower().apply(
        lambda x: ' '.join([word for word in x.split() if word not in stop_words])
    )

    return data, book_names

# Load data
data, book_names = load_data()

##################################################################################################################

# Compute TF-IDF & Cosine Similarity
@st.cache_resource
def compute_similarity():
    vectorizer = TfidfVectorizer()
    tf_idf_matrix = vectorizer.fit_transform(data['corpus'])
    return cosine_similarity(tf_idf_matrix)

similarity_matrix = compute_similarity()

# Reverse book name lookup
book_numbers = {v: k for k, v in book_names.items()}

# **Find Similar Verses Function**
def top_verse(input_book, input_chapter, input_verse, top_n=10):
    try:
        book_num = str(book_numbers.get(input_book, ""))
        locator = data.loc[
            (data['b'].astype(str) == book_num) &
            (data['c'].astype(str) == str(input_chapter)) &
            (data['v'].astype(str) == str(input_verse))
        ]
        if locator.empty:
            return pd.DataFrame(columns=["Book", "Chapter", "Verse", "Text", "Similarity Score"])
        idx = locator.index[0]
        similarity_scores = list(enumerate(similarity_matrix[idx]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        sim_indices = [i[0] for i in similarity_scores[1:top_n + 1]]
        sim_values = [i[1] for i in similarity_scores[1:top_n + 1]]
        recommended = data.iloc[sim_indices].copy()
        recommended['Similarity Score'] = sim_values
        recommended = recommended[['Book Name', 'c', 'v', 't', 'Similarity Score']]
        recommended.columns = ["Book", "Chapter", "Verse", "Text", "Similarity Score"]
        return recommended
    except Exception as e:
        st.error(f"Error in recommendation: {e}")
        return pd.DataFrame(columns=["Book", "Chapter", "Verse", "Text", "Similarity Score"])

##################################################################################################################

# **Embeddings & FAISS Index Loading**
@st.cache_resource
def load_embeddings_and_index():
    """Download and load embeddings and FAISS index."""
    
    # Define file paths
    emb_file = "bible_embeddings.npy"
    index_file = "bible_faiss.index"
    
    # Download embeddings if not present
    if not os.path.exists(emb_file):
        gdown.download("https://drive.google.com/uc?id=1-z5RDrWKn13t65PmsWb4FhOGyRcJbOpB", emb_file, quiet=False)
    
    # Download FAISS index if not present
    if not os.path.exists(index_file):
        gdown.download("https://drive.google.com/uc?id=1I7sqgWmMjFcjqDVic73IMPXK8tehcX-A", index_file, quiet=False)

    # Load files
    embeddings = np.load(emb_file, allow_pickle=True)
    index = faiss.read_index(index_file)

    return embeddings, index

embeddings, index = load_embeddings_and_index()

##################################################################################################################
# Load Sentence-BERT model
@st.cache_resource
def load_model():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model
tokenizer, model = load_model()

# Function to get embedding for a new query
def get_embedding(text):
    """Generate embedding for the given text using Sentence-BERT."""
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        output = model(**tokens)
    return output.last_hidden_state[:, 0, :].numpy().astype(np.float32)

# Find similar verses
def find_similar_verses(query, top_n=5):
    """Find similar Bible verses based on FAISS search."""
    query_embedding = get_embedding(query).reshape(1, -1)
    query_embedding = np.array(query_embedding, dtype=np.float32)  # Ensure NumPy array
    query_embedding = query_embedding.reshape(1, -1)  # Reshape for FAISS search
    
    distances, indices = index.search(query_embedding, top_n)

    if indices is None or len(indices[0]) == 0:
        st.error("No similar verses found.")
        return pd.DataFrame(columns=["Book Name", "Chapter", "Verse", "Text", "Similarity"])

    # Extract results
    results = data.iloc[indices[0]][["Book Name", "c", "v", "t"]].copy()
    
    # Ensure distances is not empty and matches the indices size
    if len(distances) > 0 and len(distances[0]) == len(results):
        results["Similarity"] = 1 - distances[0]  # Convert distance to similarity
    else:
        results["Similarity"] = np.nan  # Fallback if distances are empty
    
    return results

##################################################################################################################

# Load Stable Diffusion pipeline
@st.cache_resource
def load_pipeline():
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    return pipe

pipe = load_pipeline()
##################################################################################################################
# Cache Data Loading
@st.cache_data
def load_prophets():
    prophets = pd.read_csv("t_bbe.csv").dropna()
    
    # Map book numbers to book names
    prophets_names = {
        1: 'Genesis', 2: 'Exodus', 3: 'Leviticus', 4: 'Numbers', 5: 'Deuteronomy',
        6: 'Joshua', 7: 'Judges', 8: 'Ruth', 9: '1 Samuel', 10: '2 Samuel',
        11: '1 Kings', 12: '2 Kings', 13: '1 Chronicles', 14: '2 Chronicles',
        15: 'Ezra', 16: 'Nehemiah', 17: 'Esther', 18: 'Job', 19: 'Psalms',
        20: 'Proverbs', 21: 'Ecclesiastes', 22: 'Song of Solomon', 23: 'Isaiah',
        24: 'Jeremiah', 25: 'Lamentations', 26: 'Ezekiel', 27: 'Daniel',
        28: 'Hosea', 29: 'Joel', 30: 'Amos', 31: 'Obadiah', 32: 'Jonah',
        33: 'Micah', 34: 'Nahum', 35: 'Habakkuk', 36: 'Zephaniah', 37: 'Haggai',
        38: 'Zechariah', 39: 'Malachi'
    }
    # Map book numbers to names
    prophets['Book Name'] = prophets['b'].map(prophets_names)

    # Process text: Remove stopwords
    stop_words = set(stopwords.words('english'))
    prophets['corpus'] = prophets['t'].astype(str).str.lower().apply(
        lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

    return prophets, prophets_names

# Load data
prophets, prophets_names = load_prophets()
##################################################################################################################
# Cache Data Loading
@st.cache_data
def load_fulfilled():
    fulfilled = pd.read_csv("t_bbe.csv").dropna()
    
    # Map book numbers to book names
    fulfilled_names = {
        40: 'Matthew', 41: 'Mark', 42: 'Luke',
        43: 'John', 44: 'Acts', 45: 'Romans', 46: '1 Corinthians',
        47: '2 Corinthians', 48: 'Galatians', 49: 'Ephesians', 50: 'Philippians',
        51: 'Colossians', 52: '1 Thessalonians', 53: '2 Thessalonians',
        54: '1 Timothy', 55: '2 Timothy', 56: 'Titus', 57: 'Philemon',
        58: 'Hebrews', 59: 'James', 60: '1 Peter', 61: '2 Peter',
        62: '1 John', 63: '2 John', 64: '3 John', 65: 'Jude', 66: 'Revelation'
    }
    # Map book numbers to names
    fulfilled['Book Name'] = fulfilled['b'].map(fulfilled_names)

    # Process text: Remove stopwords
    stop_words = set(stopwords.words('english'))
    fulfilled['corpus'] = fulfilled['t'].astype(str).str.lower().apply(
        lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

    return fulfilled, fulfilled_names

# Load data
fulfilled, fulfilled_names = load_fulfilled()

##################################################################################################################

# Compute TF-IDF & Cosine Similarity for prophecy
@st.cache_resource
def compute_similarity_prophecy():
    vectorizer = TfidfVectorizer()
    tf_idf_matrix = vectorizer.fit_transform(fulfilled['corpus'])
    return cosine_similarity(tf_idf_matrix)

similarity_matrix_prophecy = compute_similarity_prophecy()

# Reverse book name lookup
prophets_book_numbers = {v: k for k, v in prophets_names.items()}
fulfilled_book_numbers = {v: k for k, v in fulfilled_names.items()}

# **Find Similar Verses Function**
def top_verse_prophecy(input_book, input_chapter, input_verse, top_n=10):
    try:
        book_num = str(prophets_book_numbers.get(input_book, ""))
        locator = prophets.loc[
            (prophets['b'].astype(str) == book_num) &
            (prophets['c'].astype(str) == str(input_chapter)) &
            (prophets['v'].astype(str) == str(input_verse))
        ]
        if locator.empty:
            return pd.DataFrame(columns=["Book", "Chapter", "Verse", "Text", "Similarity Score"])
        idx = locator.index[0]
        similarity_scores = list(enumerate(similarity_matrix_prophecy[idx]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        sim_indices = [i[0] for i in similarity_scores[1:top_n + 1]]
        sim_values = [i[1] for i in similarity_scores[1:top_n + 1]]
        recommended = fulfilled.iloc[sim_indices].copy()
        recommended['Similarity Score'] = sim_values
        recommended = recommended[['Book Name', 'c', 'v', 't', 'Similarity Score']]
        recommended.columns = ["Book", "Chapter", "Verse", "Text", "Similarity Score"]
        recommended = recommended[recommended['Book'].notna()]
        return recommended
    except Exception as e:
        st.error(f"Error in recommendation: {e}")
        return pd.DataFrame(columns=["Book", "Chapter", "Verse", "Text", "Similarity Score"])

##################################################################################################################

# **Streamlit UI**
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Table of Contents", "Verse Recommender", "Semantic Search Recommender", "Create Image", "Prophecy"])

with tab1:
    st.title("📖 Bible Application")
    st.write("Tab2: Verse Recommender ➡️ Find other verses similar to a verse selection from the Old and New Testament")
    st.write("Tab3: Semantic Search Recommender ➡️ Find Bible Verses most similar to user-input words or phrases")
    st.write("Tab4: Create Image ➡️ Select a verse or passage to create an image")
    st.write("Tab5: Prophecy ➡️ Select an Old Testament Prophecy to see where it was fulfilled in the New Testament")

with tab2:
    st.title("📖 Bible Verse Recommender")
    st.write("Find verses similar to your selection from the Old and New Testament.")

    with st.form("user_input"):
        col1, col2, col3 = st.columns(3)
        with col1:
            input_book = st.selectbox("Select Book", book_names.values())
        with col2:
            input_chapter = st.number_input("Chapter", min_value=1, max_value=150, value=1, step=1)
        with col3:
            input_verse = st.number_input("Verse", min_value=1, max_value=176, value=1, step=1)

        top_n = st.slider("Number of Similar Verses", min_value=1, max_value=50, value=10, step=5)

        submitted = st.form_submit_button("Find Similar Verses")

    if submitted:
        results = top_verse(input_book, input_chapter, input_verse, top_n)
        searched_verse = data.loc[
            (data['Book Name'] == input_book) &
            (data['c'].astype(str) == str(input_chapter)) &
            (data['v'].astype(str) == str(input_verse))
        ]

        if not searched_verse.empty:
            st.write(f"**Input Verse:** {searched_verse.iloc[0]['t']}")
            st.write("### 🔍 Similar Verses:")
        else:
            st.write("Verse not found.")

        st.table(results)

##################################################################################################################
with tab3:

    # Streamlit UI
    st.title("📖 Bible Verse Similarity Finder")
    query = st.text_input("Enter a phrase or verse:", "Love your neighbor as yourself")
    top_n = st.slider("Number of similar verses:", min_value=1, max_value=50, value=10, step=5)

    if st.button("Find Similar Verses"):
        results = find_similar_verses(query, top_n)
        st.write("### 🔍 Similar Verses:")
        for i, row in results.iterrows():
            st.write(f"**Book:** {row['Book Name']} | **Chapter:** {row['c']} | **Verse:** {row['v']}")
            st.write(f"**Text:** {row['t']} (Similarity: {row['Similarity']:.2f})")

##################################################################################################################    

with tab4:
    st.title("🖼️ Bible Passage Text-to-Image Generator")
    st.write("Select a verse or passage to create an image.")
    with st.form("input"):
        col1, col2 = st.columns(2)
        with col1:
            in_book = st.selectbox("Select Book", book_names.values())
            in_chapter = st.number_input("Chapter", min_value=1, max_value=150, value=1, step=1)
        col3a, col3b = st.columns(2)
        with col3a:
            start_verse = st.number_input("Start Verse", min_value=1, max_value=176, value=1, step=1)
        with col3b:
            end_verse = st.number_input("End Verse", min_value=1, max_value=176, value=1, step=1)

        col4a, col4b = st.columns(2)
        with col4a:
            style = st.selectbox("Select Style", ["realistic", "oil painting", "digital art", "sketch", "fantasy art"])
        with col4b:
            resolution = st.selectbox("Image Resolution", ["512x512", "768x768"])  # Safe resolutions

        submitted = st.form_submit_button("Generate Image")

    if submitted:
        selected_verses = data.loc[
            (data['Book Name'] == in_book) &
            (data['c'].astype(str) == str(in_chapter)) &
            (data['v'].astype(int) >= start_verse) &
            (data['v'].astype(int) <= end_verse)
        ]

        if not selected_verses.empty:
            passage = ' '.join(selected_verses['t'].tolist())
            truncated_passage = passage[:300]  # Limit length for safety
            prompt = prompt = f"{truncated_passage}, style: {style}"

            width, height = map(int, resolution.split("x"))

            st.write(f"**Input Passage:** {passage}")
            st.write("### 🖼️ Generated Image:")
            image = pipe(prompt, height=height, width=width).images[0]
            st.image(image, width=600)
        else:
            st.write("Passage not found.")

################################################################################################################## 

with tab5:
    st.title("📖 Prophecy Verse Search")
    st.write("Select an Old Testament Prophecy to see where it was fulfilled in the New Testament.")
    st.info("Enter a Book, Chapter and Verse ➡️ click 'Find Prophecy Fulfillment Verses' to find New Testament Bible verses where Old Testament Prophecies were fulfilled.")
    st.info("This application works best when a verse containing a prophecy is selected.")
    st.info("Check out the links immediately below to find prophetic Old Testament verses:")
    st.markdown("[Review Prophecies and Corresponding Fullfillment Verses](https://www.jesusfilm.org/blog/old-testament-prophecies/)")
    st.markdown("[Example List of Prophecies](https://www.newtestamentchristians.com/bible-study-resources/351-old-testament-prophecies-fulfilled-in-jesus-christ/)")

    with st.form("u_input"):
        col1, col2, col3 = st.columns(3)
        with col1:
            input_book = st.selectbox("Select Book", prophets_names.values())
        with col2:
            input_chapter = st.number_input("Chapter", min_value=1, max_value=150, value=1, step=1)
        with col3:
            input_verse = st.number_input("Verse", min_value=1, max_value=176, value=1, step=1)

        submitted = st.form_submit_button("➡️Find Prophecy Fulfillment Verses")

    if submitted:
        results = top_verse_prophecy(input_book, input_chapter, input_verse, top_n=10)
        searched_verse = prophets.loc[
            (prophets['Book Name'] == input_book) &
            (prophets['c'].astype(str) == str(input_chapter)) &
            (prophets['v'].astype(str) == str(input_verse))
        ]

        if not searched_verse.empty:
            st.write(f"**Input Verse:** {searched_verse.iloc[0]['t']}")
            st.write("### 🔍 Corresponding Verses:")
            for i, row in results.iterrows():
                st.write(f"**Book:** {row['Book']} | **Chapter:** {row['Chapter']} | **Verse:** {row['Verse']}")
                st.write(f"**Text:** {row['Text']} (Similarity: {row['Similarity Score']:.2f})")
        else:
            st.write("Verse not found.")
