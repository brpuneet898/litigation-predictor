from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from dotenv import load_dotenv
import os
import json
from groq import Groq
import PyPDF2
from docx import Document
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from werkzeug.utils import secure_filename
import tempfile
import shutil

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "fallback-secret-key")

# Initialize Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Ensure data directory exists
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# System prompts
FIELD_EXTRACTION_PROMPT = """You are a legal AI assistant analyzing litigation documents. 
Carefully read the document and extract the following information:

1. **case_type**: The type/category of the case (e.g., "Commercial civil suit", "Criminal prosecution", "Civil contract dispute", "Arbitration", etc.). Be specific and accurate based on the document.

2. **jurisdiction**: The court name and city (e.g., "Supreme Court, Delhi")

3. **parties**: Names of plaintiff/petitioner and defendant/respondent

4. **short_facts**: A 2-3 sentence summary of what happened. ALWAYS provide this - summarize the key events, dispute, or allegations from the document.

5. **relief_sought**: What the plaintiff is asking for (damages, injunction, specific performance, etc.)

6. **stage_of_case**: Must be EXACTLY one of: "Pre-filing", "Trial", or "Appeal"
   - If document is a draft, consultation, or pre-litigation → "Pre-filing"
   - If case is ongoing in court, evidence stage → "Trial"
   - If challenging a lower court decision → "Appeal"

Return ONLY a valid JSON object with these exact keys:
{
  "case_type": "",
  "jurisdiction": "",
  "parties": "",
  "short_facts": "",
  "relief_sought": "",
  "stage_of_case": ""
}

IMPORTANT:
- Use EXACT capitalization for case_type and stage_of_case as shown above
- ALWAYS try to provide short_facts - even a brief summary is better than empty
- Only leave fields empty if truly no information is available
- Do not include any explanatory text, only the JSON object."""

ANALYSIS_PROMPT = """You are an expert legal AI assistant analyzing a litigation case.
Based on the provided case information and document content, provide a comprehensive analysis.

Analyze the following:
1. **issues**: List 3-7 key legal issues/questions that need to be resolved in this case
2. **claims**: List all legal claims being made (e.g., breach of contract, negligence, defamation)
3. **defenses**: List potential defenses the defendant might raise
4. **monetary_value**: Estimate the monetary value involved (damages claimed, contract value, etc.) - provide a specific amount or range if mentioned
5. **urgency_flags**: List any time-sensitive matters (statute of limitations, pending deadlines, urgent relief needed)

Return ONLY a valid JSON object with this exact structure:
{
  "issues": ["issue 1", "issue 2", "issue 3"],
  "claims": ["claim 1", "claim 2"],
  "defenses": ["defense 1", "defense 2"],
  "monetary_value": "amount or range",
  "urgency_flags": ["flag 1", "flag 2"]
}

IMPORTANT:
- Provide specific, detailed analysis based on the case facts
- Each array should have at least 2-3 items if information is available
- Use clear, professional legal terminology
- Do not include any explanatory text outside the JSON object
- If you cannot determine something, provide an empty array [] or empty string """""


def extract_text_from_pdf(file_path):
    """Extract text from PDF file"""
    text = ""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Error extracting PDF: {e}")
    return text


def extract_text_from_docx(file_path):
    """Extract text from DOCX file"""
    text = ""
    try:
        doc = Document(file_path)
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
    except Exception as e:
        print(f"Error extracting DOCX: {e}")
    return text


def create_vector_embeddings(text, session_id):
    """Create FAISS vector embeddings from text"""
    # Split text into chunks (simple chunking by paragraphs)
    chunks = [chunk.strip() for chunk in text.split('\n') if chunk.strip()]
    
    if not chunks:
        return None, []
    
    # Create embeddings
    embeddings = embedding_model.encode(chunks)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    
    # Save index for this session
    index_path = os.path.join(DATA_DIR, f"{session_id}_faiss.index")
    faiss.write_index(index, index_path)
    
    return index_path, chunks


def retrieve_context(session_id, chunks, top_k=5):
    """Retrieve relevant context from FAISS index"""
    if not chunks:
        return ""
    
    # For simplicity, return first few chunks as context
    return "\n".join(chunks[:top_k])


def call_llm(prompt, context=""):
    """Call Groq LLM API"""
    try:
        messages = [
            {"role": "system", "content": prompt}
        ]
        
        if context:
            messages.append({"role": "user", "content": f"Document content:\n{context}"})
        
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.1,
            max_tokens=2000
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_case():
    file = request.files.get("case_file")

    if not file:
        return redirect(url_for("index"))

    # Create unique session ID
    if 'session_id' not in session:
        session['session_id'] = os.urandom(16).hex()
    
    session_id = session['session_id']
    
    # Save file temporarily
    temp_dir = tempfile.mkdtemp()
    filename = secure_filename(file.filename)
    file_path = os.path.join(temp_dir, filename)
    file.save(file_path)
    
    # Extract text based on file type
    if filename.lower().endswith('.pdf'):
        text = extract_text_from_pdf(file_path)
    elif filename.lower().endswith('.docx'):
        text = extract_text_from_docx(file_path)
    else:
        shutil.rmtree(temp_dir)
        return "Unsupported file type", 400
    
    # Create vector embeddings
    index_path, chunks = create_vector_embeddings(text, session_id)
    
    # Store chunks and text in session
    session['document_chunks'] = chunks
    session['document_text'] = text[:5000]  # Store first 5000 chars for context
    
    # Get context for LLM
    context = retrieve_context(session_id, chunks, top_k=10)
    
    # Call LLM to extract fields
    llm_response = call_llm(FIELD_EXTRACTION_PROMPT, context)
    
    # Parse LLM response
    try:
        # Extract JSON from response (in case LLM adds extra text)
        response_text = llm_response.strip()
        if '```json' in response_text:
            response_text = response_text.split('```json')[1].split('```')[0]
        elif '```' in response_text:
            response_text = response_text.split('```')[1].split('```')[0]
        
        extracted_fields = json.loads(response_text.strip())
        
        # Normalize stage_of_case values to match dropdown
        stage_mapping = {
            'pre-filing': 'Pre-filing',
            'prefiling': 'Pre-filing',
            'trial': 'Trial',
            'appeal': 'Appeal'
        }
        if extracted_fields.get('stage_of_case'):
            normalized = extracted_fields['stage_of_case'].lower().strip().replace(' ', '-')
            extracted_fields['stage_of_case'] = stage_mapping.get(normalized, extracted_fields['stage_of_case'])
            
    except Exception as e:
        print(f"Error parsing LLM response: {e}")
        print(f"Raw response: {llm_response}")
        # If parsing fails, use empty fields
        extracted_fields = {
            "case_type": "",
            "jurisdiction": "",
            "parties": "",
            "short_facts": "",
            "relief_sought": "",
            "stage_of_case": ""
        }
    
    # Store extracted fields in session
    session['extracted_fields'] = extracted_fields
    
    # Clean up temp file
    shutil.rmtree(temp_dir)
    
    return redirect(url_for("case_form"))


@app.route("/case-form")
def case_form():
    if 'extracted_fields' not in session:
        return redirect(url_for("index"))
    
    fields = session['extracted_fields']
    return render_template("form.html", fields=fields)


@app.route("/analyze", methods=["POST"])
def analyze_case():
    if 'document_chunks' not in session:
        return redirect(url_for("index"))
    
    # Get form data (user can edit the fields)
    case_data = {
        "case_type": request.form.get("case_type", ""),
        "jurisdiction": request.form.get("jurisdiction", ""),
        "parties": request.form.get("parties", ""),
        "short_facts": request.form.get("short_facts", ""),
        "relief_sought": request.form.get("relief_sought", ""),
        "stage_of_case": request.form.get("stage_of_case", "")
    }
    
    # Prepare context for second LLM call
    session_id = session['session_id']
    chunks = session['document_chunks']
    context = retrieve_context(session_id, chunks, top_k=15)
    
    # Combine case data with context
    full_context = f"""Case Information:
Case Type: {case_data['case_type']}
Jurisdiction: {case_data['jurisdiction']}
Parties: {case_data['parties']}
Short Facts: {case_data['short_facts']}
Relief Sought: {case_data['relief_sought']}
Stage of Case: {case_data['stage_of_case']}

Document Content:
{context}
"""
    
    # Call LLM for analysis
    llm_response = call_llm(ANALYSIS_PROMPT, full_context)
    
    # Parse LLM response
    try:
        # Extract JSON from response (in case LLM adds extra text)
        response_text = llm_response.strip() if llm_response else "{}"
        if '```json' in response_text:
            response_text = response_text.split('```json')[1].split('```')[0]
        elif '```' in response_text:
            response_text = response_text.split('```')[1].split('```')[0]
        
        analysis = json.loads(response_text.strip())
        
        # Ensure all required keys exist
        if 'issues' not in analysis:
            analysis['issues'] = []
        if 'claims' not in analysis:
            analysis['claims'] = []
        if 'defenses' not in analysis:
            analysis['defenses'] = []
        if 'monetary_value' not in analysis:
            analysis['monetary_value'] = ""
        if 'urgency_flags' not in analysis:
            analysis['urgency_flags'] = []
            
    except Exception as e:
        print(f"Error parsing analysis response: {e}")
        print(f"Raw LLM response: {llm_response}")
        analysis = {
            "issues": [f"Error parsing analysis: {str(e)}"],
            "claims": [],
            "defenses": [],
            "monetary_value": "",
            "urgency_flags": []
        }
    
    # Store analysis in session
    session['case_data'] = case_data
    session['analysis'] = analysis
    
    return redirect(url_for("results"))


@app.route("/results")
def results():
    if 'analysis' not in session:
        return redirect(url_for("index"))
    
    case_data = session.get('case_data', {})
    analysis = session['analysis']
    
    return render_template("results.html", case_data=case_data, analysis=analysis)


if __name__ == "__main__":
    app.run(debug=True)
