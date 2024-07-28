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
    data = pd.read_csv(file_path)
    # Trim spaces from column names
    data.columns = data.columns.str.strip()
    return data

# Create vector representation of the data
@st.cache(allow_output_mutation=True)
def vectorize_data(data):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data['Matter Number'].astype(str))
    X = normalize(X)
    return X, vectorizer

# Function to query GPT with context from vectorized data
def query_gpt_with_data(question, data, X, vectorizer, message):
    try:
        practice_area = question.split("for")[-1].strip()
        practice_area_vec = vectorizer.transform([practice_area])
        D = (X @ practice_area_vec.T).toarray().ravel()
        indices = D.argsort()[-5:][::-1]
        relevant_data = data.iloc[indices]

        if relevant_data.empty:
            return "No relevant data found."

        if "contact" in question.lower():
            return relevant_data.to_dict(orient='records')
        else:
            prompt = f"Given the following data on top lawyers:\n{relevant_data.to_string(index=False)}\n{message} {practice_area}?"
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an AI assistant helping to identify top lawyers for specific cases at a law firm based on data in a csv file given to you (Matters_Bio). Recommend a top lawyer based on all the information in the file. Don't lie. Return the information in a table format (lawyer name, practice group, area of expertise, related case, and contact). The lawyer's contact info is in the column 'contact', please provide their work email and phone number."},
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
    data = load_data('/mnt/data/Matter_Bio.csv')
    
    # Validate data
    required_columns = ['Attorney Name', 'Practice Group', 'Area of Expertise', 'Matter Description', 'Contact']
    if not all(column in data.columns for column in required_columns):
        st.error("The CSV file does not contain all the required columns.")
    elif data[required_columns].isnull().any().any():
        st.error("The CSV file contains missing values in the required columns.")
    else:
        X, vectorizer = vectorize_data(data)
        message = "Please provide the top lawyers for the practice area of"
        answer = query_gpt_with_data(user_input, data, X, vectorizer, message)
        if isinstance(answer, list):
            st.table(answer)
        elif answer:
            st.write("Answer:", answer)
        else:
            st.error("Failed to retrieve an answer. Please check your input.")
