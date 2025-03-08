import streamlit as st
import requests
from bs4 import BeautifulSoup
import numpy as np
import time
import faiss
from mistralai import Mistral


# Define API key for Mistral API
# api_key = "NXyKdE5JFehmTjXn1RtYyVBOlMzPLGyB"  # Replace with your actual API key
api_key = st.secrets["MISTRAL_API_KEY"]


# Initialize Mistral client
def initialize_mistral():
    return Mistral(api_key=api_key)


# Function to obtain text embeddings with retry logic
def generate_embeddings(text_chunks, max_retries=3, delay=5):
    client = initialize_mistral()
    for attempt in range(max_retries):
        try:
            response = client.create_embeddings(model="mistral-embed", input=text_chunks)
            return response.data
        except Exception as e:
            if "429" in str(e):
                time.sleep(delay * (2**attempt))  # Exponential backoff
            else:
                raise e
    raise Exception("Exceeded retry attempts for embeddings.")

# Function to retrieve policy content
def extract_policy_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    return soup.get_text() if soup else "Policy text unavailable."

# Function to split text into chunks
def split_text(text, size=512):
    return [text[i : i + size] for i in range(0, len(text), size)]

# Function to setup FAISS index
def setup_faiss(embeddings):
    dimension = len(embeddings[0].embedding)
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array([e.embedding for e in embeddings]))
    return index

# Function to process user queries with retry logic
def process_query(user_query, text_chunks, faiss_index, max_retries=3, delay=5):
    query_embedding = np.array([generate_embeddings([user_query])[0].embedding])
    distances, indices = faiss_index.search(query_embedding, k=2)
    retrieved_chunks = " ".join([text_chunks[i] for i in indices[0]])
    query_prompt = f"""
    Reference:
    ----------------
    {retrieved_chunks}
    ----------------
    Based only on the above, respond to:
    {user_query}
    """
    return generate_response(query_prompt, max_retries, delay)

# Function to communicate with Mistral with retry logic
def generate_response(input_text, max_retries=3, delay=5, model="mistral-large-latest"):
    client = initialize_mistral()
    for attempt in range(max_retries):
        try:
            response = client.chat_completion(
                model=model,
                messages=[{"role": "user", "content": input_text}],
            )
            return response.choices[0].message.content
        except Exception as e:
            if "429" in str(e):
                time.sleep(delay * (2**attempt))  # Exponential backoff
            else:
                raise e
    raise Exception("Exceeded retry attempts for chat generation.")

# Streamlit UI configuration
def main():
    st.title("UDST Policy Assistant")

    policies = {
        "Sport and Wellness Policy": "https://www.udst.edu.qa/policy1",
        "Student Conduct Policy": "https://www.udst.edu.qa/policy2",
        "Student Attendance Policy": "https://www.udst.edu.qa/policy3",
        "Academic Standing Policy": "https://www.udst.edu.qa/policy4",
        "Final Grade Policy": "https://www.udst.edu.qa/policy5",
        "Registration Policy": "https://www.udst.edu.qa/policy6",
        "Student Engagement Policy": "https://www.udst.edu.qa/policy7",
        "International Student Policy": "https://www.udst.edu.qa/policy8",
        "Graduation Policy": "https://www.udst.edu.qa/policy9",
        "Student Counselling Policy": "https://www.udst.edu.qa/policy10",
    }

    selected_policy = st.selectbox("Choose a Policy", list(policies.keys()))
    url = policies[selected_policy]
    policy_text = extract_policy_content(url)
    text_chunks = split_text(policy_text)
    embeddings = generate_embeddings(text_chunks)
    faiss_index = setup_faiss(embeddings)

    user_input = st.text_input("Ask a question about the policy:")
    if user_input:
        response = process_query(user_input, text_chunks, faiss_index)
        st.text_area("Response:", value=response, height=250)

if __name__ == "__main__":
    main()
