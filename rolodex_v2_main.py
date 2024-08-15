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
def load_and_clean_data(file_path, encoding='latin1'):
    data = pd.read_csv(file_path, encoding=encoding)
    data.columns = data.columns.str.replace('ï»¿', '').str.replace('Ã', '').str.strip()  # Clean unusual characters and whitespace
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]  # Remove unnamed columns
    return data

# Create vector database for a given dataframe and columns
@st.cache(allow_output_mutation=True)
def create_vector_db(data, columns):
    combined_text = data[columns].fillna('').apply(lambda x: ' '.join(x.astype(str)), axis=1)
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(combined_text)
    X = normalize(X)
    index = faiss.IndexFlatL2(X.shape[1])
    index.add(X.toarray())
    return index, vectorizer

# Function to call GPT-4
def call_gpt(messages):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.5,
    )
    return response.choices[0]['message']['content'].strip()

# Function to query GPT with context from vector DB
def query_gpt_with_data(question, matters_data, matters_index, matters_vectorizer):
    try:
        # Preprocess the input question to extract key terms
        question = ' '.join(question.split()[-3:])  # Consider the last three words in the query
        question_vec = matters_vectorizer.transform([question])
        
        # Search for the most relevant entries
        D, I = matters_index.search(normalize(question_vec).toarray(), k=10)  # Increase k to 10
        
        relevant_data = matters_data.iloc[I[0]] if I.size > 0 and not (I == -1).all() else matters_data.head(1)
        
        # Filter relevant columns for output
        filtered_data = relevant_data[['Attorney', 'Practice Area', 'Matter Description', 'Work Email', 'Work Phone']].drop_duplicates(subset=['Attorney'])
        
        # Ensure there is at least one lawyer to recommend
        if filtered_data.empty:
            filtered_data = matters_data[['Attorney', 'Practice Area', 'Matter Description', 'Work Email', 'Work Phone']].dropna(subset=['Attorney']).drop_duplicates(subset=['Attorney']).head(1)

        # Prepare the context for GPT-4
        context = filtered_data.to_string(index=False)
        messages = [
            {"role": "system", "content": "You are the CEO of Scale Law firm. I want you to go through the csv files and make a recommendation based on the type of case, matter, and the attorney background. Don't make any information up. Don't say things like 'As an AI, I'm unable to go through actual files or access real-time data.' We just want you to make a recommendation based on the given files. Assume you are the CEO of this law firm and you are confident in your recommendation even without certain information. For the recommendation you can even cite cases that the attorney has worked on for your recommendation. I want you to look through Matters_Updated.csv when making a recommendation so you can see the lawyers' case history. Utilize the users prompt when giving a recommendation. Don't tell the user that you're utilizing the matters file! Tell them based on the data available instead. Be really detailed in your recommendation and even cite some cases in it too. A user will often use the phrases recommend me a lawyer or tell me the best person or other introductory phases. Make sure you try your best to return a result."},
            {"role": "user", "content": f"Based on the following information, please make a recommendation:\n\n{context}\n\nRecommendation:"}
        ]
        
        # Call GPT-4 for a recommendation
        gpt_response = call_gpt(messages)

        # Process the GPT-4 response to extract recommendations
        recommendations = gpt_response.split('\n')
        recommendations = [rec for rec in recommendations if rec.strip()]

        # Ensure no duplicates
        recommendations = list(dict.fromkeys(recommendations))

        # Convert recommendations into a DataFrame
        recommendations_df = pd.DataFrame(recommendations, columns=['Recommendation Reasoning'])

        # Extract relevant details of the top recommended lawyers
        top_recommended_lawyers = filtered_data.drop_duplicates(subset=['Attorney'])

        # Display the main table without the index
        st.write("All Potential Lawyers with Recommended Skillset:")
        st.write(top_recommended_lawyers.to_html(index=False), unsafe_allow_html=True)
        st.write("Recommendation Reasoning:")
        st.write(recommendations_df.to_html(index=False), unsafe_allow_html=True)

        # Create and display another table listing all matters for each attorney
        for lawyer in top_recommended_lawyers['Attorney'].unique():
            st.write(f"**{lawyer}'s Matters:**")
            lawyer_matters = matters_data[matters_data['Attorney'] == lawyer][['Practice Area', 'Matter Description']]
            st.write(lawyer_matters.to_html(index=False), unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error querying GPT: {e}")

# Streamlit app layout
st.title("Rolodex AI: Find Your Ideal Lawyer 👨‍⚖️ Utilizing Open AI GPT-4 Version 2")
st.write("Ask questions about the top lawyers in a specific practice area at Scale LLP:")
st.write("Note this is a prototype and can make mistakes!")

# Default questions as buttons
default_questions = {
    "Who are the top lawyers for corporate law?": "corporate law",
    "Which attorneys have the most experience with intellectual property?": "intellectual property",
    "Can you recommend a lawyer specializing in employment law?": "employment law",
    "Who are the best litigators for complex cases?": "complex cases",
    "Which lawyer should I contact for real estate matters?": "real estate"
}

# Check if a default question button is clicked
user_input = ""
for question_text, question_value in default_questions.items():
    if st.button(question_text):
        user_input = question_text
        break

# Also allow users to input custom questions
if not user_input:
    user_input = st.text_input("Or type your own question:", placeholder="e.g., 'Who are the top lawyers for corporate law?'")

if user_input:
    # Clear cache before each search
    st.cache_data.clear()
    
    # Load CSV data on the backend
    matters_data = load_and_clean_data('Cleaned_Matters_Data.csv', encoding='latin1')  # Ensure correct file path and encoding
    
    if not matters_data.empty:
        # Ensure the correct column names are used
        matters_index, matters_vectorizer = create_vector_db(matters_data, ['Attorney', 'Matter Description'])  # Adjusted columns
        
        if matters_index is not None:
            query_gpt_with_data(user_input, matters_data, matters_index, matters_vectorizer)
    else:
        st.error("Failed to load data.")

    # Accuracy feedback section
    st.write("### How accurate was this result?")
    accuracy_options = ["Accurate", "Not Accurate", "Type your own feedback"]
    accuracy_choice = st.radio("Please select one:", accuracy_options)

    # If user chooses to type their own feedback, display a text input field
    if accuracy_choice == "Type your own feedback":
        custom_feedback = st.text_input("Please provide your feedback:")
    else:
        custom_feedback = accuracy_choice

    # Optionally, save or process this feedback
    if st.button("Submit Feedback"):
        if custom_feedback:
            st.write(f"Thank you for your feedback: '{custom_feedback}'")
        else:
            st.error("Please provide feedback before submitting.")
