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

# Create vector database for a given dataframe and column
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
def query_gpt_with_data(question, matters_data, users_data, matters_index, users_index, matters_vectorizer, users_vectorizer):
    try:
        practice_area = question.split("for")[-1].strip()
        practice_area_vec_matters = matters_vectorizer.transform([practice_area])
        practice_area_vec_users = users_vectorizer.transform([practice_area])
        
        D_matters, I_matters = matters_index.search(normalize(practice_area_vec_matters).toarray(), k=5)
        D_users, I_users = users_index.search(normalize(practice_area_vec_users).toarray(), k=5)
        
        relevant_matters_data = matters_data.iloc[I_matters[0]]
        relevant_users_data = users_data.iloc[I_users[0]]

        # Merge data based on Attorney Name
        combined_data = relevant_matters_data.merge(relevant_users_data, left_on='Attorney', right_on='Attorney Name', how='left')

        # Debugging: Display the combined data
       # st.write("Combined Data for Debugging:")
       # st.write(combined_data)

        if "contact information" in question.lower():
            return combined_data[['Attorney', 'Work Email', 'Work Phone']].to_dict(orient='records')
        else:
            prompt = f"Given the following data on top lawyers:\n{combined_data.to_string()}\nPlease provide the top lawyers for the practice area of {practice_area}."
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an assistant helping to identify top lawyers. You work at Scale LLP and you're in charge of helping lawyers find other lawyers based on a skillset. We have given you two datasets to look through to makee your reccomendation, matters and user.csv. Make your best recommendation for a lawyer but don't lie.  Return recommendations in a table with lawyer name, work email and work phone, and relevant cases, this information is in Users CSV. Go through the matters.csv file first to make a recommendation, then find the corresponding lawyers information(work email and work phone) when you return it to the user. Your end user is a lawyer who doesn't know this is coming from a csv file, make sure you keep this a secret.You have all the data you need. If you really don't know the answer, just say Not Avalable for the data. Return only 2-3 lawyers. All their contact information should be available in Users.csv"},
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
    matters_data = load_and_clean_data('Matters.csv', encoding='latin1')  # Try 'latin1' encoding if 'utf-8' fails
    users_data = load_and_clean_data('Users.csv', encoding='latin1')      # Try 'latin1' encoding if 'utf-8' fails
    
    if not matters_data.empty and not users_data.empty:
        # Ensure the correct column names are used
        matters_index, matters_vectorizer = create_vector_db(matters_data, ['Attorney', 'Practice Area', 'Matter Description'])  # Adjusted columns
        users_index, users_vectorizer = create_vector_db(users_data, ['Role', 'Practice Group', 'Practice Description', 'Area of Expertise'])  # Adjusted columns
        
        if matters_index is not None and users_index is not None:
            answer = query_gpt_with_data(user_input, matters_data, users_data, matters_index, users_index, matters_vectorizer, users_vectorizer)
            if answer:
                display_response_in_table(answer)
            else:
                st.error("Failed to retrieve an answer. Please check your input.")
    else:
        st.error("Failed to load data.")
