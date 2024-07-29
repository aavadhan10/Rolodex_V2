import openai
import streamlit as st
import pandas as pd
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from dotenv import load_dotenv
import os 

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

# Function to call GPT-4
def call_gpt(prompt):
    response = openai.Completion.create(
        model="gpt-4",
        prompt=prompt,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.5,
    )
    return response.choices[0].text.strip()

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

        # Find the most relevant case
        most_relevant_case = relevant_data.iloc[D[0].argmin()]
        most_relevant_case_details = {
            'Attorney Name': most_relevant_case['Attorney'],
            'Work Email': most_relevant_case['Work Email'],
            'Work Phone': most_relevant_case['Work Phone'],
            'Relevant Matters': f"Practice Area: {most_relevant_case['Practice Area']}, Matter Description: {most_relevant_case['Matter Description']}"
        }

        # Find the top 1-3 lawyers with complete information and best vector match
        complete_lawyers = filtered_data.dropna(subset=['Attorney', 'Work Email', 'Work Phone'])
        top_complete_lawyers = complete_lawyers.head(3)

        # Remove duplicates
        top_complete_lawyers = top_complete_lawyers[top_complete_lawyers['Attorney'] != most_relevant_case['Attorney']]

        # Add the most relevant case to the top lawyers
        if not top_complete_lawyers.empty:
            top_complete_lawyers = top_complete_lawyers.rename(columns={'Attorney': 'Attorney Name'})
            top_complete_lawyers['Relevant Matters'] = top_complete_lawyers.apply(
                lambda row: f"Practice Area: {row['Practice Area']}, Matter Description: {row['Matter Description']}",
                axis=1
            )
            top_complete_lawyers = top_complete_lawyers[['Attorney Name', 'Work Email', 'Work Phone', 'Relevant Matters']]
            combined_results = pd.DataFrame([most_relevant_case_details]).append(top_complete_lawyers, ignore_index=True)
        else:
            combined_results = pd.DataFrame([most_relevant_case_details])

        # Prepare the prompt for GPT-4
        context = combined_results.to_string(index=False)
        prompt = f"Based on the following information, please make a recommendation:\n\n{context}\n\nRecommendation:"
        
        # Call GPT-4 for a recommendation
        gpt_response = call_gpt(prompt)
        
        st.write("Top 1-3 Recommended Lawyer(s) (Best Vector Match with Complete Information) and Most Relevant Case:")
        st.write(combined_results)
        st.write("GPT-4 Recommendation:")
        st.write(gpt_response)

    except Exception as e:
        st.error(f"Error querying GPT: {e}")

# Streamlit app layout
st.title("Rolodex AI: Find Your Ideal Lawyer 👨‍⚖️ Utilizing Open AI GPT-4")
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
