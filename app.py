import streamlit as st
from pinecone import Pinecone
import openai
from pdfminer.high_level import extract_text
import requests
from bs4 import BeautifulSoup
import tempfile
from dotenv import load_dotenv
import os
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT
from io import BytesIO

# Load environment variables
load_dotenv()

# Initialize OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Pinecone
pinecone_api_key = os.getenv("PINECONE_API_KEY")
index_name = "resumebuilder"
namespace = "ns1"

if not pinecone_api_key:
    st.error("Pinecone API key not found. Please check your .env file.")
else:
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(index_name)

def extract_text_from_pdf(pdf_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(pdf_file.read())
            temp_file_path = temp_file.name
        
        text = extract_text(temp_file_path)
        os.unlink(temp_file_path)
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

def embed_text(text):
    try:
        response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
        return np.array(response['data'][0]['embedding'], dtype=np.float32)
    except Exception as e:
        st.error(f"Error generating embedding: {e}")
        return np.array([])

def store_resume_in_pinecone(resume_id, resume_text):
    embedding = embed_text(resume_text)
    if embedding.size > 0:
        index.upsert(
            vectors=[
                {
                    "id": resume_id, 
                    "values": embedding.tolist(), 
                    "metadata": {"text": resume_text}
                }
            ],
            namespace=namespace
        )

def fetch_job_description(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        job_description = soup.get_text()
        return job_description
    except Exception as e:
        st.error(f"Error fetching job description: {e}")
        return ""

def match_resumes_to_job_description(job_description):
    job_embedding = embed_text(job_description)
    if job_embedding.size > 0:
        results = index.query(
            namespace=namespace,
            vector=job_embedding.tolist(),
            top_k=1,
            include_values=True,
            include_metadata=True
        )
        return results
    return []

def optimize_resume(resume_text, job_description):
    prompt = f"""Optimize the following resume for the job description. 
    The optimized resume should include the following sections in order: 
    Summary, Skills, Work Experience, Certificates, and Education. 
    Start with the person's name and the profession mentioned in the job description.
    Ensure all work experience is included, but prioritize relevant experience.
    The entire resume must fit on one page, so be concise while preserving key information.
    Use bullet points (•) for listing items, not dashes (-).
    Do not include any additional text or explanations outside of the resume content.

    Job Description: {job_description}

    Resume:
    {resume_text}"""
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a professional resume optimizer."},
                {"role": "user", "content": prompt}
            ]
        )
        optimized_resume = response.choices[0].message['content'].strip()
        return optimized_resume
    except Exception as e:
        st.error(f"Error optimizing resume: {e}")
        return ""

def create_pdf(text):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch, leftMargin=0.5*inch, rightMargin=0.5*inch)
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(name='Name', fontSize=14, leading=16, spaceAfter=4, alignment=1))
    styles.add(ParagraphStyle(name='Profession', fontSize=12, leading=14, spaceAfter=8, alignment=1))
    styles.add(ParagraphStyle(name='Section', fontSize=11, leading=13, spaceAfter=4, fontName='Helvetica-Bold'))
    styles.add(ParagraphStyle(name='Content', fontSize=9, leading=11, spaceAfter=2, bulletIndent=10, leftIndent=20))
    styles.add(ParagraphStyle(name='ContentNoBullet', fontSize=9, leading=11, spaceAfter=2, leftIndent=10))

    content = []
    lines = text.split('\n')
    
    content.append(Paragraph(lines[0], styles['Name']))
    content.append(Paragraph(lines[1], styles['Profession']))
    content.append(Spacer(1, 8))

    current_section = ''
    for line in lines[2:]:
        line = line.strip()
        if line in ['Summary:', 'Skills:', 'Work Experience:', 'Certificates:', 'Education:']:
            current_section = line
            content.append(Paragraph(f"<b>{current_section}</b>", styles['Section']))
        elif line.startswith('•'):
            content.append(Paragraph(line, styles['Content']))
        elif line:
            content.append(Paragraph(line, styles['ContentNoBullet']))

    doc.build(content)
    buffer.seek(0)
    return buffer

# Streamlit UI
st.title("Resume Optimizer")

uploaded_file = st.file_uploader("Upload Your Resume PDF", type="pdf")
job_description_url = st.text_input("Enter Job Description URL")

if st.button("Optimize Resume"):
    if uploaded_file and job_description_url:
        st.write("Processing resume and job description...")

        resume_text = extract_text_from_pdf(uploaded_file)
        store_resume_in_pinecone("current_resume", resume_text)

        job_description = fetch_job_description(job_description_url)
        matched_resumes = match_resumes_to_job_description(job_description)

        if matched_resumes and 'matches' in matched_resumes:
            best_resume_text = matched_resumes['matches'][0]['metadata']['text']
            optimized_resume = optimize_resume(best_resume_text, job_description)

            pdf_buffer = create_pdf(optimized_resume)

            st.download_button(
                label="Download Optimized Resume",
                data=pdf_buffer,
                file_name="optimized_resume.pdf",
                mime="application/pdf"
            )
        else:
            st.error("No matching resumes found.")
    else:
        st.error("Please upload your resume PDF and provide a job description URL.")