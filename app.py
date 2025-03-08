# import streamlit as st
# import requests
# from bs4 import BeautifulSoup
# import numpy as np
# import time
# import faiss
# from mistralai import Mistral


# # Define API key for Mistral API
# # api_key = "NXyKdE5JFehmTjXn1RtYyVBOlMzPLGyB"  # Replace with your actual API key
# api_key = st.secrets["MISTRAL_API_KEY"]


# # Initialize Mistral client
# def initialize_mistral():
#     return Mistral(api_key=api_key)


# # Function to obtain text embeddings with retry logic
# def generate_embeddings(text_chunks, max_retries=3, delay=5):
#     client = initialize_mistral()
#     for attempt in range(max_retries):
#         try:
#             response = client.create_embeddings(model="mistral-embed", input=text_chunks)
#             return response.data
#         except Exception as e:
#             if "429" in str(e):
#                 time.sleep(delay * (2**attempt))  # Exponential backoff
#             else:
#                 raise e
#     raise Exception("Exceeded retry attempts for embeddings.")

# # Function to retrieve policy content
# def extract_policy_content(url):
#     response = requests.get(url)
#     soup = BeautifulSoup(response.text, "html.parser")
#     return soup.get_text() if soup else "Policy text unavailable."

# # Function to split text into chunks
# def split_text(text, size=512):
#     return [text[i : i + size] for i in range(0, len(text), size)]

# # Function to setup FAISS index
# def setup_faiss(embeddings):
#     dimension = len(embeddings[0].embedding)
#     index = faiss.IndexFlatL2(dimension)
#     index.add(np.array([e.embedding for e in embeddings]))
#     return index

# # Function to process user queries with retry logic
# def process_query(user_query, text_chunks, faiss_index, max_retries=3, delay=5):
#     query_embedding = np.array([generate_embeddings([user_query])[0].embedding])
#     distances, indices = faiss_index.search(query_embedding, k=2)
#     retrieved_chunks = " ".join([text_chunks[i] for i in indices[0]])
#     query_prompt = f"""
#     Reference:
#     ----------------
#     {retrieved_chunks}
#     ----------------
#     Based only on the above, respond to:
#     {user_query}
#     """
#     return generate_response(query_prompt, max_retries, delay)

# # Function to communicate with Mistral with retry logic
# def generate_response(input_text, max_retries=3, delay=5, model="mistral-large-latest"):
#     client = initialize_mistral()
#     for attempt in range(max_retries):
#         try:
#             response = client.chat_completion(
#                 model=model,
#                 messages=[{"role": "user", "content": input_text}],
#             )
#             return response.choices[0].message.content
#         except Exception as e:
#             if "429" in str(e):
#                 time.sleep(delay * (2**attempt))  # Exponential backoff
#             else:
#                 raise e
#     raise Exception("Exceeded retry attempts for chat generation.")

# # Streamlit UI configuration
# def main():
#     st.title("UDST Policy Assistant")

#     policies = {
#         "Sport and Wellness Policy": "https://www.udst.edu.qa/policy1",
#         "Student Conduct Policy": "https://www.udst.edu.qa/policy2",
#         "Student Attendance Policy": "https://www.udst.edu.qa/policy3",
#         "Academic Standing Policy": "https://www.udst.edu.qa/policy4",
#         "Final Grade Policy": "https://www.udst.edu.qa/policy5",
#         "Registration Policy": "https://www.udst.edu.qa/policy6",
#         "Student Engagement Policy": "https://www.udst.edu.qa/policy7",
#         "International Student Policy": "https://www.udst.edu.qa/policy8",
#         "Graduation Policy": "https://www.udst.edu.qa/policy9",
#         "Student Counselling Policy": "https://www.udst.edu.qa/policy10",
#     }

#     selected_policy = st.selectbox("Choose a Policy", list(policies.keys()))
#     url = policies[selected_policy]
#     policy_text = extract_policy_content(url)
#     text_chunks = split_text(policy_text)
#     embeddings = generate_embeddings(text_chunks)
#     faiss_index = setup_faiss(embeddings)

#     user_input = st.text_input("Ask a question about the policy:")
#     if user_input:
#         response = process_query(user_input, text_chunks, faiss_index)
#         st.text_area("Response:", value=response, height=250)

# if __name__ == "__main__":
#     main()

import streamlit as st
import requests
from bs4 import BeautifulSoup
import numpy as np
import faiss
import time
from mistralai import Mistral

# Define API key for Mistral API
api_key = "NXyKdE5JFehmTjXn1RtYyVBOlMzPLGyB"  # Replace with your actual API key

# Function to get text embeddings from Mistral with rate limit handling
def get_text_embedding(list_txt_chunks, max_retries=3, delay=5):
    client = Mistral(api_key=api_key)
    for attempt in range(max_retries):
        try:
            embeddings_batch_response = client.embeddings.create(model="mistral-embed", inputs=list_txt_chunks)
            return embeddings_batch_response.data
        except Exception as e:
            if "429" in str(e):
                time.sleep(delay * (2 ** attempt))  # Exponential backoff
            else:
                raise e
    raise Exception("Exceeded retry attempts for embeddings.")

# Fetching and parsing the UDST policies page based on the policy URL
def fetch_policies(url):
    response = requests.get(url)
    html_doc = response.text
    soup = BeautifulSoup(html_doc, "html.parser")
    policies_text = soup.get_text()  # Extract text from the webpage
    return policies_text

# Chunking the policies text into smaller chunks for processing
def chunk_text(text, chunk_size=512):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Initialize the FAISS index for similarity search
def initialize_faiss(embeddings):
    d = len(embeddings[0].embedding)
    index = faiss.IndexFlatL2(d)
    index.add(np.array([embedding.embedding for embedding in embeddings]))
    return index

# Function to handle user queries
def handle_query(query, chunks, index):
    question_embeddings = np.array([get_text_embedding([query])[0].embedding])
    D, I = index.search(question_embeddings, k=2)  # Searching for top 2 similar chunks
    retrieved_chunk = [chunks[i] for i in I[0]]
    prompt = f"""
    Context information is below.
    ---------------------
    {retrieved_chunk}
    ---------------------
    Given the context information and not prior knowledge, answer the query.
    Query: {query}
    Answer:
    """
    response = mistral(prompt)
    return response

# Function to interact with Mistral for generating answers with rate limit handling
def mistral(user_message, model="mistral-large-latest", max_retries=3, delay=5):
    client = Mistral(api_key=api_key)
    for attempt in range(max_retries):
        try:
            chat_response = client.chat.complete(
                model=model,
                messages=[
                    {"role": "user", "content": user_message},
                ]
            )
            return chat_response.choices[0].message.content
        except Exception as e:
            if "429" in str(e):
                time.sleep(delay * (2 ** attempt))  # Exponential backoff
            else:
                raise e
    raise Exception("Exceeded retry attempts for chat generation.")

# Streamlit UI Setup
def main():
    st.title("UDST Policies Chatbot")

    # Define the policy links
    policy_links = {
        "Sport and Wellness Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/sport-and-wellness-facilities-and",
        "Student Conduct Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/student-conduct-policy",
        "Student Attendance Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/student-attendance-policy",
        "Academic Standing Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-standing-policy",
        "Final Grade Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/final-grade-policy",
        "Registration Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/registration-policy",
        "Student Engagement Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/student-engagement-policy",
        "International Student Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/international-student-policy",
        "Graduation Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/graduation-policy",
        "Student Counselling Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/student-council-procedure",
    }

    # Listbox for policy selection
    selected_policy = st.selectbox("Select a Policy", list(policy_links.keys()))
    
    # Fetch and chunk the selected policy text
    url = policy_links[selected_policy]
    policies_text = fetch_policies(url)
    chunks = chunk_text(policies_text)
    text_embeddings = get_text_embedding(chunks)
    index = initialize_faiss(text_embeddings)

    # Text box for entering the query
    user_query = st.text_input("Enter your query:")

    # Text area to display the response
    if user_query:
        answer = handle_query(user_query, chunks, index)
        st.text_area("Answer", value=answer, height=300)

if __name__ == "__main__":
    main()
