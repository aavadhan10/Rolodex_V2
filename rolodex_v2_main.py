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
    X = vectorizer.fit_transform(data['Responsible Attorney'].astype(str))  # Use relevant data column
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
                    {"role": "system", "content": "You are an AI assistant helping to identify top lawyers for specific cases at a law firm based on data in a csv file given to you. You are playing the role of a lawyer matching lead at a law firm called Scale LLP. You specifically focus on reviewing a csv file matter_bio) that has data about attorneys, time spent on cases, their personal information, their practice area and types of clients they work for. This data shows you all this information and your main job is to review this data and use your extensive experience in the law firm space to make sure we are extracting the right information from the data to recommend the best lawyers for the users prompt input. Your goal is to extract out specific information from these. You have a JD (law degree) in the US which gives you sound expertise in law terminology. You believe it is your responsibility to be as accurate as possible in abstracting info from the prompt.You have worked as a top-performing Lawyer Lead for five years and have recently taken over the division to lead your team.\nYou are an AI assistant that helps people find information.\n\nAlways retrieve the following information from the csv file given to you.Do not make information up. I also want you to return lawyer  information only in the past year (based on the date) to make sure it is a lawyer that has not left the firm yet. Return only the lawyer name, practice area, relevant cases worked on and contact information if available. Return multiple lawyers with that information if they are applicable matches. Do not return where the lawyer went to school. If there is no data do not say that to the user, just say "No Lawyer match based on criterea submitted. "},
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
