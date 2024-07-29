import openai
import streamlit as st
import pandas as pd
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI API using environment variable
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Load and clean CSV data with specified encoding
@st.cache_data
def load_and_clean_data(file_path, encoding='utf-8'):
    data = pd.read_csv(file_path, encoding=encoding)
    data.columns = data.columns.str.replace('ï»¿', '').str.replace('Ã', '').str.strip()  # Clean unusual characters and whitespace
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]  # Remove unnamed columns
    return data

# Create vector database for a given dataframe and columns
@st.cache(allow_output_mutation=True)
def create_vector_db(data, columns):
    combined_text = data[columns].fillna('').apply(lambda x: ' '.join(x.astype(str)), axis=1)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(combined_text)
    X = normalize(X)
    index = faiss.IndexFlatL2(X.shape[1])
    index.add(X.toarray())
    return index, vectorizer

# Function to query GPT with context from vector DB
def query_gpt_with_data(question, matters_data, matters_index, matters_vectorizer):
    try:
        practice_area = question.split("for")[-1].strip()
        practice_area_vec = matters_vectorizer.transform([practice_area])
        
        D, I = matters_index.search(normalize(practice_area_vec).toarray(), k=5)
        
        relevant_data = matters_data.iloc[I[0]]

        # Filter relevant columns for output
        filtered_data = relevant_data[['Attorney', 'Practice Area', 'Matter Description', 'Work Email', 'Work Phone']]

        # Debugging: Display the filtered data
        st.write("Filtered Data for Debugging:")
        st.write(filtered_data)

        # Find the top 3 lawyers with the best vector match
        top_matches = relevant_data.head(3)
        top_matches = top_matches.dropna(subset=['Attorney', 'Work Email', 'Work Phone'])

        if top_matches.empty:
            st.write("No recommended lawyers found with complete information.")
        else:
            top_matches = top_matches.rename(columns={'Attorney': 'Attorney Name'})
            st.write("Top 3 Recommended Lawyer(s) (Best Vector Match):")
            st.write(top_matches[['Attorney Name', 'Work Email', 'Work Phone']])

    except Exception as e:
        st.error(f"Error querying GPT: {e}")

# Streamlit app layout
st.title("Rolodex AI: Find Your Ideal Lawyer 👨‍⚖️ Utilizing Open AI GPT 4 LLM's V2 Playground Version")
st.write("Ask questions about the top lawyers in a specific practice area at Scale LLP:")
user_input = st.text_input("Your question:", placeholder="e.g., 'Who are the top lawyers for corporate law?'")

if user_input:
    # Clear cache before each search
    st.cache_data.clear()
    
    # Load CSV data on the backend
    matters_data = load_and_clean_data('Matters.csv', encoding='latin1')  # Ensure correct file path and encoding
    
    if not matters_data.empty:
        # Ensure the correct column names are used
        matters_index, matters_vectorizer = create_vector_db(matters_data, ['Practice Area', 'Matter Description'])  # Adjusted columns
        
        if matters_index is not None:
            query_gpt_with_data(user_input, matters_data, matters_index, matters_vectorizer)
    else:
        st.error("Failed to load data.")
