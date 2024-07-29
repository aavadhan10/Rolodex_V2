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

# Load CSV data with specified encoding
@st.cache_data
def load_data(file_path, encoding='utf-8'):
    return pd.read_csv(file_path, encoding=encoding)

# Create vector database for a given dataframe and column
@st.cache(allow_output_mutation=True)
def create_vector_db(data, column):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data[column].astype(str))
    X = normalize(X)
    index = faiss.IndexFlatL2(X.shape[1])
    index.add(X.toarray())
    return index, vectorizer

# Function to query GPT with context from vector DB
def query_gpt_with_data(question, matters_data, users_data, matters_index, users_index, matters_vectorizer, users_vectorizer):
    try:
        practice_area = question.split("for")[-1].strip()
        practice_area_vec_matters = matters_vectorizer.transform([practice_area])
        practice_area_vec_users = users_vectorizer.transform([practice_area])
        
        D_matters, I_matters = matters_index.search(normalize(practice_area_vec_matters).toarray(), k=5)
        D_users, I_users = users_index.search(normalize(practice_area_vec_users).toarray(), k=5)
        
        relevant_matters_data = matters_data.iloc[I_matters[0]]
        relevant_users_data = users_data.iloc[I_users[0]]

        relevant_data_combined = pd.concat([relevant_matters_data, relevant_users_data], axis=1)

        if "contact information" in question.lower():
            return relevant_data_combined.to_dict(orient='records')
        else:
            prompt = f"Given the following data on top lawyers:\n{relevant_data_combined.to_string()}\nPlease provide the top lawyers for the practice area of {practice_area}."
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an assistant helping to identify top lawyers. Return recommendations in a table with lawyer name, contact, practice group and relevant cases."},
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
st.title("Rolodex AI: Find Your Ideal Lawyer 👨‍⚖️ Utilizing Open AI GPT 4 LLM's V2 Playground Version")
st.write("Ask questions about the top lawyers in a specific practice area at Scale LLP:")
user_input = st.text_input("Your question:", placeholder="e.g., 'Who are the top lawyers for corporate law?'")

if user_input:
    matters_data = load_data('Matters.csv', encoding='latin1')  # Try 'latin1' encoding if 'utf-8' fails
    users_data = load_data('Users.csv', encoding='latin1')      # Try 'latin1' encoding if 'utf-8' fails
    
    st.write("Matters Data Columns:", matters_data.columns.tolist())
    st.write("Users Data Columns:", users_data.columns.tolist())
    
    if not matters_data.empty and not users_data.empty:
        # Ensure the correct column names are used
        matters_index, matters_vectorizer = create_vector_db(matters_data, 'Responsible Attorney')  # Adjusted column name
        users_index, users_vectorizer = create_vector_db(users_data, 'Attorney Name')              # Adjusted column name
        
        if matters_index is not None and users_index is not None:
            answer = query_gpt_with_data(user_input, matters_data, users_data, matters_index, users_index, matters_vectorizer, users_vectorizer)
            if answer:
                display_response_in_table(answer)
            else:
                st.error("Failed to retrieve an answer. Please check your input.")
    else:
        st.error("Failed to load data.")
