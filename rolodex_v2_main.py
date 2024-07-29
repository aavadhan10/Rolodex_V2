import openai
import streamlit as st
import pandas as pd
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from dotenv import load_dotenv
# Load environment variables
load_dotenv()
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
def query_gpt_with_data(question, matters_data, matters_index, matters_vectorizer):
    try:
        practice_area = question.split("for")[-1].strip()
        practice_area_vec = matters_vectorizer.transform([practice_area])
        
        D, I = matters_index.search(normalize(practice_area_vec).toarray(), k=5)
        
        relevant_data = matters_data.iloc[I[0]]

        # Filter relevant columns for debugging
        filtered_data = relevant_data[['Attorney', 'Matter Description', 'Work Email', 'Work Phone']]

        # Debugging: Display the filtered combined data
        st.write("Combined Data for Debugging:")
        st.write(filtered_data)

        if "contact information" in question.lower():
            return filtered_data.to_dict(orient='records')
        else:
            prompt = f"Given the following data on top lawyers:\n{filtered_data.to_string()}\nPlease provide the top lawyers for the practice area of {practice_area}."
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an assistant helping to identify top lawyers. You work at Scale LLP and you're in charge of helping lawyers find other lawyers based on a skillset. You are not manipulating data, you are just looking through one csv file to make a decision. Return out your best recommendation for lawyers (2-3) with their Lawyer Name, Work Email, Work phone and Relevant Case, return this information in a table for the end user. If it's the same lawyer, don't repeat in the table (only one lawyer). If you don't have a recommendation just say data not available."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150
            )
            return response.choices[0]['message']['content'].strip()
    except Exception as e:
        st.error(f"Error querying GPT: {e}")
        return None

# Function to display the response in a table format
#def display_response_in_table(response):
  #  if isinstance(response, pd.DataFrame):
      #  st.table(response)
 #   else:
    #    st.write(response)


# Streamlit app layout
st.title("Rolodex AI: Find Your Ideal Lawyer 👨‍⚖️ Utilizing Open AI GPT 4 LLM's V2 Playground Version")
st.write("Ask questions about the top lawyers in a specific practice area at Scale LLP:")
user_input = st.text_input("Your question:", placeholder="e.g., 'Who are the top lawyers for corporate law?'")

if user_input:
    # Load CSV data on the backend
    matters_data = load_and_clean_data('Matters.csv', encoding='latin1')  # Ensure correct file path and encoding
    
    if not matters_data.empty:
        # Ensure the correct column names are used
        matters_index, matters_vectorizer = create_vector_db(matters_data, ['Attorney', 'Practice Area', 'Matter Description'])  # Adjusted columns
        
        if matters_index is not None:
            answer = query_gpt_with_data(user_input, matters_data, matters_index, matters_vectorizer)
            if answer is not None:
                display_response_in_table(answer)
            else:
                st.error("Failed to retrieve an answer. Please check your input.")
    else:
        st.error("Failed to load data.")
