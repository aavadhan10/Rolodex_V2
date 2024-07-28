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

# Load CSV data
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

# Create vector database
@st.cache(allow_output_mutation=True)
def create_vector_db(data):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data['Matter Number'].astype(str))  # Use relevant data column
    X = normalize(X)
    index = faiss.IndexFlatL2(X.shape[1])
    index.add(X.toarray())
    return index, vectorizer

# Function to query GPT with context from vector DB
def query_gpt_with_data(question, data, index, vectorizer, message):
    try:
        practice_area = question.split("for")[-1].strip()
        practice_area_vec = vectorizer.transform([practice_area])
        D, I = index.search(normalize(practice_area_vec).toarray(), k=5)
        relevant_data = data.iloc[I[0]]

        if "contact information" in question.lower():
            return relevant_data.to_dict(orient='records')
        else:
            prompt = f"Given the following data on top lawyers:\n{relevant_data.to_string()}\n{message} {practice_area}?"
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an AI assistant helping to identify top lawyers for specific cases at a law firm based on data in a csv file given to you (Matters_Bio). Recommend a top lawyer based on all the information in the file. Dont lie. Return the information in a table format (lawyer name practice group, area of expertise, related case and contact. The lawyers contact info is in the column "contact", please provide their work email and phone number "},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150
            )
            return response.choices[0]['message']['content'].strip()
    except Exception as e:
        st.error(f"Error querying GPT: {e}")
        return None

# Function to display the response in a table format
def display_response_in_table(response):
    if isinstance(response, list):
        st.table(response)
    else:
        st.write(response)

# Streamlit app layout
st.title("Rolodex AI: Find Your Ideal Lawyer 👨‍⚖️ Utilizing Open AI GPT 4 LLM's V2 Play Version")
st.write("Ask questions about the top lawyers in a specific practice area at Scale LLP:")
user_input = st.text_input("Your question:", placeholder="e.g., 'Who are the top lawyers for corporate law?'")

if user_input:
    data = load_data('Matter_Bio.csv')
    if not data.empty:
        index, vectorizer = create_vector_db(data)
        if index is not None and vectorizer is not None:
            message = "Please provide the top lawyers for the practice area of"
            answer = query_gpt_with_data(user_input, data, index, vectorizer, message)
            if isinstance(answer, list):
                st.table(answer)
            elif answer:
                st.write("Answer:", answer)
            else:
                st.error("Failed to retrieve an answer. Please check your input.")
    else:
        st.error("Failed to load data.")
