import streamlit as st
from pypdf import PdfReader
from openai import OpenAI
from dotenv import load_dotenv
import os
import math
import numpy as np

# here pypdf, openai, dotenv are toolboxes and PdfReader, OpenAi, load_dotenv are the actual tools we are gonna use

load_dotenv()

client = OpenAI(
    api_key = os.getenv("OPENAI_API_KEY")
)

st.write("Chat with pdf")

uploaded_files = st.file_uploader(
        "Upload a pdf",
        type=["pdf"],
        accept_multiple_files=True
)
all_text = ""

def create_chunks(text,chunk_size):
    i = 0
    chunks = []
    while(i<len(text)):
        chunks.append(text[i:i+chunk_size])
        i+=chunk_size
    return chunks  
  
def get_embedding(text):
    response = client.embeddings.create(
    model="text-embedding-3-small",
    input=text
    )
    return response.data[0].embedding  

def cosine_similarity(vec1,vec2):
    v1 = np.array(vec1)
    v2 = np.array(vec2)

    return np.dot(v1,v2)/(
        np.linalg.norm(v1) * np.linalg.norm(v2)
    )

if uploaded_files:

    group_text = ""

    for file in uploaded_files:
        reader = PdfReader(file)

        all_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                all_text += text

        group_text += all_text

    if (
        "data" not in st.session_state
        or "pdf_text" not in st.session_state
        or st.session_state.pdf_text != group_text
    ):

        pdf_library = []

        for file in uploaded_files:
            reader = PdfReader(file)

            all_text = ""
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    all_text += text

            chunks = create_chunks(all_text, 500)

            for chunk in chunks:
                embedding = get_embedding(chunk)
                pdf_library.append(
                    {
                        "pdf_name": file.name,
                        "text": chunk,
                        "embedding": embedding,
                    }
                )

        st.session_state.data = pdf_library
        st.session_state.pdf_text = group_text

    question = st.text_input(
        "Ask a question about the pdf"
    ) 

    if question.strip()!="" and group_text.strip()!="":
        question_embedding = get_embedding(question)

        best_chunk = ""
        storage = []
        data = st.session_state.data

        for chunk in data:
            score = cosine_similarity(chunk["embedding"],question_embedding)
            storage.append((score,chunk["text"],chunk["pdf_name"]))

        sources = set()
        storage.sort(reverse=True)
        top_3 = storage[:3]
        for x in top_3:
            best_chunk += x[1] +"\n\n"
            sources.add(x[2])        
        prompt =f"""
        you are an assistant whose purpose is to help user understand the given document 


        Relevant context:
        {best_chunk}

        Question:
        {question}

        Answer the given question using only the relevant context available here
        Be as concise and precise as u can be
        Also if u cant answer based on it just tell him so
        """
        response = client.responses.create(
            model="gpt-5.4",
            input=prompt
        )   
        st.write(response.output_text)
        st.write("Sources: ")
        for source in sources:
            st.write(source)