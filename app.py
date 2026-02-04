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

1. **case_title**: Brief title/name of the case (e.g., "Ram Kumar vs State of Maharashtra")

2. **case_type**: MUST select ONLY ONE from this exact list:
["Abuses and threatening", "Adoption", "ADMIRALTY SUITS", "Anticipatory Bail", "Appeal against the Judgment and Order of conviction", "Application by a tenant for fixation of standard rent", "Application for damages before the Motor Accidents Claims Tribunal", "Application for temporary injunction", "Application U/S 340 Perjury", "Application u/s 156 (Direction to register FIR)", "Arbitration Application", "Arbitration Petition", "Bail before a magistrate", "Bigamous marriage", "Cancellation of Bail", "Caveat", "Cheating", "Civil revision Application", "Civil suit (compensation)", "Civil suit (injunction)", "Civil suit (posession)", "Civil Suit (recovery)", "Commercial Admiralty Suit", "Commercial Arbitration Petition", "Commercial Execution Application", "Commercial Intellectual property Rights suit", "Commercial Suit", "Commercial SUMMARY SUITS", "Complaint under Sec 138 of NI Act", "Conjugal Rights", "COUNTER CLAIM", "Criminal complaint", "Criminal Revision Application", "Criminal tresspass", "Damages against a Doctor for negligent act", "Damages for defamation", "Defamation", "Dissolution of partnership and rendition of accounts", "Divorce", "Divorce by mutual consent", "Domestic violence", "Execution Application", "Execution of maintenance order already passed in favour of wife", "Execution petition (Darkhast) on the basis of a decree of Civil Court", "First Appeal/ Civil Appeal", "For posession by a landlord against the tenant under the Rent Control Act", "Garnishee Notice", "Heirship Certificate", "Hurt", "Judicial separation", "Maintenance Application", "Mandatory Injunction", "Marriage petition", "Memo of revision application against the order of maintenance", "Mesne profits", "Motor Accident Claim petition", "PARSI SUITS", "Partition in a Hindu joint family", "Permanent Injunction", "Plain maintenance application", "Probate on the basis of Will", "Recovery of amount ordered", "Recovery of money for price of goods sold or work done", "Recovery of money on the basis of a promissory note", "Return of property Application", "Review Application", "Setting aside a decree obtained by fraud", "Specific performance of contract or damages", "Succession certificate", "Summary Suit", "Wrongful dismissal against the Government"]
If document doesn't match any option exactly, choose the closest match.

3. **jurisdiction**: MUST select ONLY ONE from this exact list:
["Ahmednagar", "Akola", "Amravati", "Aurangabad", "Beed", "Bhandara", "Buldhana", "Chandrapur", "Dhule", "Gadchiroli", "Gondia", "Jalgaon", "Jalna", "Jamkhed", "Karjat", "Kolhapur", "Kopergaon", "Latur", "Maharashtra Family Courts", "Maharashtra Industrial and Lab", "Maharashtra School Tribunals", "Mah State Cooperative Appellat", "Mumbai City Civil Court", "Mumbai CMM Courts", "Mumbai Motor Accident Claims T", "Mumbai Small Causes Court", "Nagpur", "Nanded", "Nandurbar", "Nashik", "Newasa", "Osmanabad", "Parbhani", "Parner", "Pathardi", "Pune", "Rahata", "Rahuri", "Raigad", "Ratnagiri", "Sangli", "Sangamner", "Satara", "Shevgaon", "Shrirampur", "Shrigonda", "Sindhudurg", "Solapur", "Thane", "Wardha", "Washim", "Yavatmal"]
Select the city/court location mentioned in the document.

4. **court_level**: MUST select ONLY ONE from this exact list:
["Supreme Court", "High Court", "District and Sessions Court", "Sessions Court", "Additional Sessions Court", "Chief Judicial Magistrate", "Metropolitan Magistrate Court", "Civil Court Senior Division", "Civil Court Junior Division", "Family Court", "Small Causes Court", "Consumer Court", "Labour Court", "Motor Accident Claims Tribunal", "Special Court", "Tribunal"]
Select based on the court hierarchy mentioned in the document.

5. **claim_amount**: Numeric value of claim/damages amount in currency. Extract ONLY the number (e.g., "500000" or "0" if not mentioned).

6. **opponent_counsel**: Name of the opposing party's lawyer/advocate. Look for:
   - "Advocate for respondent/defendant"
   - "Counsel appearing for the other side"
   - Lawyer names mentioned alongside the opposing party
   - Leave empty ONLY if no advocate/lawyer name is found

7. **opponent_profile**: Brief description of opponent party (company/individual, their role, background). Include:
   - Party name (respondent/defendant)
   - Whether individual, company, or government entity
   - Their position/status in the case

8. **key_legal_issues**: 2-3 sentence summary of the main legal questions/issues that need to be resolved

Return ONLY a valid JSON object with these exact keys:
{
  "case_title": "",
  "case_type": "",
  "jurisdiction": "",
  "court_level": "",
  "claim_amount": "0",
  "opponent_counsel": "",
  "opponent_profile": "",
  "key_legal_issues": ""
}

IMPORTANT:
- For case_type, jurisdiction, and court_level: MUST use EXACT text from the lists provided
- For claim_amount: provide number only, default to "0" if not found
- ALWAYS try to extract key_legal_issues - even a brief summary is better than empty
- Only leave fields empty if truly no information is available in the document
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
    
    # Save chunks to disk as JSON (don't store in session - too large for cookies)
    chunks_path = os.path.join(DATA_DIR, f"{session_id}_chunks.json")
    with open(chunks_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False)
    
    return index_path, chunks


def load_chunks_from_disk(session_id):
    """Load text chunks from disk"""
    chunks_path = os.path.join(DATA_DIR, f"{session_id}_chunks.json")
    if os.path.exists(chunks_path):
        with open(chunks_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


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
    
    # Get context for LLM (increased to 20 chunks for better coverage)
    context = retrieve_context(session_id, chunks, top_k=20)
    
    # Call LLM to extract fields
    llm_response = call_llm(FIELD_EXTRACTION_PROMPT, context)
    
    # Debug: Print LLM response
    print("=" * 80)
    print("LLM EXTRACTION RESPONSE:")
    print(llm_response)
    print("=" * 80)
    
    # Parse LLM response
    if llm_response:
        try:
            # Extract JSON from response (in case LLM adds extra text)
            response_text = llm_response.strip()
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0]
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0]
            
            extracted_fields = json.loads(response_text.strip())
                
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            print(f"Raw response: {llm_response}")
            # If parsing fails, use empty fields
            extracted_fields = {
                "case_title": "",
                "case_type": "",
                "jurisdiction": "",
                "court_level": "",
                "claim_amount": "0",
                "opponent_counsel": "",
                "opponent_profile": "",
                "key_legal_issues": ""
            }
    else:
        print("LLM call failed - no response received")
        # If LLM call fails, use empty fields
        extracted_fields = {
            "case_title": "",
            "case_type": "",
            "jurisdiction": "",
            "court_level": "",
            "claim_amount": "0",
            "opponent_counsel": "",
            "opponent_profile": "",
            "key_legal_issues": ""
        }
    
    # Store extracted fields in session
    session['extracted_fields'] = extracted_fields
    
    # Clean up temp file
    shutil.rmtree(temp_dir)
    
    return redirect(url_for("case_form"))


@app.route("/form")
def new_case_form():
    """Route for creating a new case with optional file upload"""
    # Initialize empty fields for manual entry
    empty_fields = {
        "case_title": "",
        "case_type": "",
        "jurisdiction": "",
        "court_level": "",
        "claim_amount": "",
        "opponent_counsel": "",
        "opponent_profile": "",
        "key_legal_issues": ""
    }
    return render_template("form.html", fields=empty_fields)


@app.route("/case-form")
def case_form():
    if 'extracted_fields' not in session:
        return redirect(url_for("index"))
    
    fields = session['extracted_fields']
    return render_template("form.html", fields=fields)


@app.route("/analyze", methods=["POST"])
def analyze_case():
    # Get form data (user can edit the fields)
    case_data = {
        "case_title": request.form.get("case_title", ""),
        "case_type": request.form.get("case_type", ""),
        "jurisdiction": request.form.get("jurisdiction", ""),
        "court_level": request.form.get("court_level", ""),
        "claim_amount": request.form.get("claim_amount", ""),
        "opponent_counsel": request.form.get("opponent_counsel", ""),
        "opponent_profile": request.form.get("opponent_profile", ""),
        "key_legal_issues": request.form.get("key_legal_issues", "")
    }
    
    # Check if document was uploaded
    has_document = 'session_id' in session
    
    if has_document:
        # Prepare context for second LLM call with document
        session_id = session['session_id']
        chunks = load_chunks_from_disk(session_id)
        if chunks:
            context = retrieve_context(session_id, chunks, top_k=15)
        else:
            has_document = False
            context = ""
    else:
        # Manual entry without document - use case data as context
        context = f"""Case Title: {case_data['case_title']}
Case Type: {case_data['case_type']}
Jurisdiction: {case_data['jurisdiction']}
Court Level: {case_data['court_level']}
Claim Amount: {case_data['claim_amount']}
Opponent Counsel: {case_data['opponent_counsel']}
Opponent Profile: {case_data['opponent_profile']}
Key Legal Issues: {case_data['key_legal_issues']}"""
    
    # Combine case data with context
    full_context = f"""Case Information:
Case Title: {case_data['case_title']}
Case Type: {case_data['case_type']}
Jurisdiction: {case_data['jurisdiction']}
Court Level: {case_data['court_level']}
Claim Amount: {case_data['claim_amount']}
Opponent Counsel: {case_data['opponent_counsel']}
Opponent Profile: {case_data['opponent_profile']}
Key Legal Issues: {case_data['key_legal_issues']}

{"Document Content:" if has_document else "Additional Context:"}
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
