# Resume Optimizer

This Streamlit application optimizes resumes based on job descriptions using AI. It extracts text from a PDF resume, matches it with a job description, and generates an optimized one-page resume.

## Prerequisites

- Python 3.7 or higher
- Pinecone account
- OpenAI API key

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/resume-optimizer.git
   cd resume-optimizer
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root directory with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   PINECONE_API_KEY=your_pinecone_api_key_here
   ```

## Usage

1. Ensure you have set up a Pinecone index named "resumebuilder" with the appropriate dimension for the OpenAI embeddings (1536 for "text-embedding-ada-002" model).

2. Run the Streamlit app:
   ```
   streamlit run resume_optimizer.py
   ```

3. Open your web browser and go to the URL provided by Streamlit (usually `http://localhost:8501`).

4. Use the web interface to:
   - Upload your resume PDF
   - Enter a job description URL
   - Click "Optimize Resume" to generate an optimized version

5. Download the optimized resume PDF.

## Features

- Extracts text from PDF resumes
- Fetches job descriptions from URLs
- Uses AI to optimize resumes based on job descriptions
- Generates a one-page PDF with optimized content
- Includes all work experience while prioritizing relevant information
- Uses proper formatting with bold section headers and bullet points

## Troubleshooting

If you encounter any issues:

1. Ensure all required libraries are installed correctly.
2. Check that your API keys in the `.env` file are valid and correctly set.
3. Verify that your Pinecone index is set up correctly.
4. Make sure you have an active internet connection.

]