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
import requests
from bs4 import BeautifulSoup

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

# Indian Kanoon API Configuration
INDIAN_KANOON_BASE_URL = "https://api.indiankanoon.org"
INDIAN_KANOON_TOKEN = os.getenv("INDIAN_KANOON_API_TOKEN")
IK_HEADERS = {
    "Authorization": f"Token {INDIAN_KANOON_TOKEN}",
    "Accept": "application/json",
    "Content-Type": "application/x-www-form-urlencoded"
}

# Helper Functions for Indian Kanoon API
def html_to_text(html_content):
    """Convert HTML content to plain text"""
    if not html_content:
        return ""
    soup = BeautifulSoup(html_content, "html.parser")
    for br in soup.find_all("br"):
        br.replace_with("\n")
    return soup.get_text("\n").strip()

def search_indian_kanoon(query, max_results=10):
    """Search for cases on Indian Kanoon"""
    try:
        url = f"{INDIAN_KANOON_BASE_URL}/search/"
        payload = {
            "formInput": query,
            "pagenum": "0"
        }
        print(f"   🌐 POST {url}")
        print(f"   📤 Payload: {payload}")
        
        response = requests.post(url, headers=IK_HEADERS, data=payload, timeout=30)
        
        print(f"   📥 Status Code: {response.status_code}")
        
        if response.ok:
            result = response.json()
            print(f"   ✅ Response received: {len(str(result))} chars")
            return result
        else:
            print(f"   ❌ API Error: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
    except Exception as e:
        print(f"   ❌ Indian Kanoon search error: {e}")
        import traceback
        traceback.print_exc()
    return {"docs": []}

def get_case_metadata(docid):
    """Fetch metadata for a specific case document"""
    try:
        url = f"{INDIAN_KANOON_BASE_URL}/docmeta/{docid}/"
        print(f"      🔍 Fetching metadata from: {url}")
        response = requests.post(url, headers=IK_HEADERS, data={}, timeout=30)
        print(f"      📥 Metadata status: {response.status_code}")
        if response.ok:
            result = response.json()
            print(f"      ✅ Metadata keys: {list(result.keys()) if result else 'Empty'}")
            return result
        else:
            print(f"      ❌ Metadata error: {response.text[:100]}")
    except Exception as e:
        print(f"      ❌ Indian Kanoon metadata error: {e}")
    return {}

def get_case_document(docid):
    """Fetch full document content for a case"""
    try:
        url = f"{INDIAN_KANOON_BASE_URL}/doc/{docid}/"
        print(f"      🔍 Fetching document from: {url}")
        response = requests.post(url, headers=IK_HEADERS, data={}, timeout=30)
        print(f"      📥 Document status: {response.status_code}")
        if response.ok:
            data = response.json()
            print(f"      ✅ Document keys: {list(data.keys()) if data else 'Empty'}")
            # Extract HTML content and convert to text
            html_content = data.get('doc', '') or data.get('content', '') or data.get('html', '')
            text_content = html_to_text(html_content)
            print(f"      📝 Document text length: {len(text_content)} chars")
            return text_content
        else:
            print(f"      ❌ Document error: {response.text[:100]}")
    except Exception as e:
        print(f"      ❌ Indian Kanoon document error: {e}")
    return ""


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


PREDICTION_PROMPT = """You are an expert litigation prediction AI with deep knowledge of Indian legal system, court procedures, case outcomes, and legal costs across different jurisdictions in Maharashtra.

Based on the provided case information and supporting documents, analyze and predict the litigation outcome with THREE different prediction models:

1. **BALANCED**: Realistic, moderate predictions based on typical case outcomes
2. **AGGRESSIVE**: Optimistic predictions assuming best-case scenarios (higher win probability, shorter duration, lower costs)
3. **CONSERVATIVE**: Cautious predictions assuming challenges and uncertainties (lower win probability, longer duration, higher costs)

Consider these factors:
- Case type, jurisdiction, and court level
- Strength of legal claims vs defenses
- Quality and completeness of evidence/documentation
- Opponent counsel experience and profile
- Historical case outcome patterns in similar cases
- Court backlog and typical case durations in the jurisdiction
- Legal complexity and claim amount
- Key legal issues and their precedents

Provide your prediction in EXACTLY this JSON structure with ALL THREE models:
{{
  "balanced": {{
    "win_probability": <number between 0-100>,
    "confidence_level": <number between 0-100>,
    "discovery_timeline": "<estimated time like '3-6 months'>",
    "mediation_timeline": "<estimated time like '8-10 months'>",
    "duration": "<estimated time like '6-12 months' or '1-2 years'>",
    "estimated_costs": "₹<min> - ₹<max>",
    "risk_tag": "<Low Risk/Medium Risk/High Risk>",
    "key_factors": [
      {{
        "factor_name": "<concise factor name>",
        "impact_type": "Positive or Negative",
        "category": "<Legal/Procedural/Evidence/Financial/Case Merit>",
        "impact_score": <number between -20 and +20>,
        "description": "<one sentence explanation of how this factor affects the case>"
      }}
    ]
  }},
  "aggressive": {{
    "win_probability": <number between 0-100>,
    "confidence_level": <number between 0-100>,
    "discovery_timeline": "<estimated time like '3-6 months'>",
    "mediation_timeline": "<estimated time like '8-10 months'>",
    "duration": "<estimated time like '6-12 months' or '1-2 years'>",
    "estimated_costs": "₹<min> - ₹<max>",
    "risk_tag": "<Low Risk/Medium Risk/High Risk>",
    "key_factors": [
      {{
        "factor_name": "<concise factor name>",
        "impact_type": "Positive or Negative",
        "category": "<Legal/Procedural/Evidence/Financial/Case Merit>",
        "impact_score": <number between -20 and +20>,
        "description": "<one sentence explanation of how this factor affects the case>"
      }}
    ]
  }},
  "conservative": {{
    "win_probability": <number between 0-100>,
    "confidence_level": <number between 0-100>,
    "discovery_timeline": "<estimated time like '3-6 months'>",
    "mediation_timeline": "<estimated time like '8-10 months' or '1-2 years'>",
    "duration": "<estimated time like '6-12 months' or '1-2 years'>",
    "estimated_costs": "₹<min> - ₹<max>",
    "risk_tag": "<Low Risk/Medium Risk/High Risk>",
    "key_factors": [
      {{
        "factor_name": "<concise factor name>",
        "impact_type": "Positive or Negative",
        "category": "<Legal/Procedural/Evidence/Financial/Case Merit>",
        "impact_score": <number between -20 and +20>,
        "description": "<one sentence explanation of how this factor affects the case>"
      }}
    ]
  }}
}}

Guidelines for each field:
- **win_probability**: 0-100 percentage. Analyze case merits, evidence strength, legal precedents, and all available information
  - Aggressive: More optimistic assessment (+10-20% higher than balanced)
  - Conservative: More cautious assessment (-10-20% lower than balanced)
- **confidence_level**: 0-100 percentage representing certainty in your prediction
  - Higher confidence: Strong documentation, clear legal grounds, favorable precedents, comprehensive evidence
  - Medium confidence: Moderate evidence, some uncertainties in legal interpretation, adequate documentation
  - Lower confidence: Insufficient documentation, unclear legal position, unpredictable factors, missing evidence
  - Aggressive: Higher confidence assuming best-case scenarios
  - Conservative: Lower confidence accounting for uncertainties and challenges
- **discovery_timeline**: Time from case filing to complete discovery phase (document exchange, depositions, interrogatories, evidence gathering)
  - Scale this based on case complexity, court level, and jurisdiction backlog
  - Discovery duration should be proportional to total case duration
  - Aggressive: Efficient, streamlined discovery process
  - Conservative: Extended discovery with potential delays and complications
- **mediation_timeline**: Time from filing to mediation/settlement attempts
  - Usually occurs after discovery is substantially complete or in progress
  - Should be chronologically after discovery_timeline begins
  - Consider court's mediation processes and jurisdiction practices
  - Aggressive: Earlier mediation attempts for faster resolution
  - Balanced: Standard mediation timeline
  - Conservative: Later mediation with full preparation and extended timelines
- **duration**: Total case duration from filing to final Trial/Settlement conclusion
  - This is the FINAL endpoint of the entire case
  - Must be LONGER than both discovery_timeline and mediation_timeline
  - Consider: court level, jurisdiction backlog, case complexity, evidence volume, legal issues complexity
  - Scale appropriately - simple cases can resolve quickly, complex cases can take many years
  - Aggressive: Optimistic timeline assuming efficient processes
  - Conservative: Extended timeline accounting for delays, appeals, and complications
- **estimated_costs**: Legal fees in Indian Rupees (₹). Scale based on:
  - Court level and forum
  - Case complexity and duration
  - Claim amount and stakes involved
  - Jurisdiction (metro vs smaller cities)
  - Opponent counsel reputation and firm size
  - Expected litigation intensity
  - Aggressive: Lower cost estimates assuming efficient resolution

  - Conservative: Higher cost estimates (accounting for delays/complications)
- **risk_tag**: Overall risk assessment based on all factors
  - Low Risk: High win probability, strong case, manageable costs and duration
  - Medium Risk: Moderate win probability, some uncertainties, reasonable costs
  - High Risk: Lower win probability, weak case, or very high costs/duration
  - Tag should holistically match the prediction profile considering win probability, costs, duration, and confidence level
  - Aggressive model: Likely lower risk tags
  - Conservative model: Likely higher risk tags
- **key_factors**: Array of EXACTLY 5 key factors that influenced your prediction (REQUIRED for all three models)
  - Each model should have its own unique set of 5 factors that explain THAT specific prediction
  - **factor_name**: Short, clear name (2-4 words max) like "Jurisdiction Precedents", "Evidence Strength", "Opposing Counsel Track Record"
  - **impact_type**: Either "Positive" (helps your case) or "Negative" (hurts your case)
  - **category**: Must be one of these EXACT values:
    - "Legal" - precedents, statutes, case law, legal doctrines
    - "Procedural" - court processes, timelines, jurisdiction practices, opposing counsel tactics
    - "Evidence" - documentation quality, witness testimony, proof strength
    - "Financial" - claim amount, cost considerations, damages valuation
    - "Case Merit" - overall case strength, legal issues complexity, factual disputes
  - **impact_score**: Number from -20 to +20 representing relative impact strength
    - Positive scores (+1 to +20): Factors helping your case (higher = stronger positive impact)
    - Negative scores (-1 to -20): Factors hurting your case (lower = stronger negative impact)
    - Score magnitude shows relative importance (±15-20 = major factor, ±10-14 = moderate, ±5-9 = minor, ±1-4 = minimal)
    - Scores do NOT need to sum to any specific total - each represents independent relative strength
  - **description**: ONE clear sentence (under 15 words) explaining how this factor affects the outcome
  - IMPORTANT: Generate factors based on the ACTUAL case details provided - reference specific evidence, jurisdiction, legal issues, etc.
  - Distribute categories appropriately - don't use same category for all 5 factors
  - Mix positive and negative factors for balanced analysis (typically 3-4 positive, 1-2 negative OR vice versa depending on case strength)

Return ONLY the JSON object with all three models. No additional text or explanations."""


KEYWORD_EXTRACTION_PROMPT = """You are a legal research assistant analyzing a case to extract search keywords for finding similar precedents.

Based on the case information provided (including both structured fields and document content), extract 3-5 relevant legal keywords and phrases that would be most effective for searching Indian case law databases.

Consider:
- Key legal concepts and doctrines mentioned in case details AND document
- Specific laws, acts, or sections referenced in either source
- Type of legal dispute and specific claims
- Important legal terms and precedents mentioned in the document
- Facts and circumstances that distinguish this case

Return ONLY a JSON object in this format:
{
  "keywords": ["keyword1", "keyword2", "keyword3"]
}

Keep keywords concise and legally specific. Prioritize terms that appear in both the case fields and document content. No explanations, just JSON."""


PRECEDENT_ANALYSIS_PROMPT = """You are an expert legal analyst evaluating a precedent case's relevance to the current litigation.

You will receive FULL DOCUMENTS for both cases - analyze them comprehensively.

Current Case Summary (includes structured data and FULL document evidence):
{current_case}

Precedent Case to Analyze (FULL DOCUMENT):
Title: {precedent_title}
Court: {precedent_court}
Date: {precedent_year}
Complete Case Document: {precedent_content}

Compare these TWO COMPLETE DOCUMENTS and analyze deeply:

1. **similarity_score**: How similar is this precedent to the current case? (0-100)
   - Deep analysis: Compare legal doctrines, cited laws, case facts, arguments, evidence patterns
   - Consider: specific sections cited, legal reasoning, factual circumstances, parties involved
   - Look for: similar statutory provisions, comparable fact patterns, analogous legal arguments
   - 90-100: Highly similar (nearly identical legal issues, facts, and statutory basis)
   - 70-89: Very similar (same legal principles, comparable facts, related provisions)
   - 50-69: Moderately similar (related legal area, some factual overlap)
   - Below 50: Weakly similar (different issues but same legal domain)

2. **outcome**: What was the outcome? Choose ONE:
   - "Settled in favor of plaintiff"
   - "Plaintiff victory"
   - "Defendant victory"
   - "Settled"
   - "Dismissed"
   - "Partially favored plaintiff"

3. **relevance_explanation**: Brief 2-3 sentence explanation of how this precedent case influences the current prediction. 
   - Cite specific legal principles, statutory provisions, or factual parallels from BOTH documents
   - Explain how the precedent's outcome/reasoning applies to the current case

Return ONLY a JSON object:
{{
  "similarity_score": 85,
  "outcome": "Plaintiff victory",
  "relevance_explanation": "Both cases involve Section 420 IPC for cheating with similar fraudulent representations. The precedent established that dishonest inducement applies even in commercial disputes, directly supporting the current complaint."
}}

No additional text."""



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


def fetch_and_analyze_precedents(case_data, case_context):
    """Fetch and analyze top 3 precedent cases from Indian Kanoon"""
    precedents = []
    
    print("\n" + "="*80)
    print("PRECEDENT SEARCH DEBUG LOG")
    print("="*80)
    
    # Debug: Check token
    if not INDIAN_KANOON_TOKEN:
        print("❌ ERROR: INDIAN_KANOON_API_TOKEN not found in .env file")
        print("   Please add: INDIAN_KANOON_API_TOKEN=your_token_here")
        return precedents
    
    if INDIAN_KANOON_TOKEN == "your_indian_kanoon_token_here":
        print("❌ ERROR: INDIAN_KANOON_API_TOKEN is still placeholder")
        print("   Please replace with your actual Indian Kanoon API token")
        return precedents
    
    print(f"✅ API Token configured: {INDIAN_KANOON_TOKEN[:10]}...")
    
    try:
        # Step 1: Extract keywords using LLM
        # Combine form fields with document content for richer keyword extraction
        case_summary = f"""Case Title: {case_data.get('case_title', 'N/A')}
Case Type: {case_data.get('case_type', 'N/A')}
Key Legal Issues: {case_data.get('key_legal_issues', 'N/A')}
Court Level: {case_data.get('court_level', 'N/A')}
Jurisdiction: {case_data.get('jurisdiction', 'N/A')}
Opponent Profile: {case_data.get('opponent_profile', 'N/A')}

Additional Context from Uploaded Document:
{case_context[:1500] if case_context and len(case_context) > 50 else 'No additional document context available'}"""
        
        print(f"\n📝 Extracting keywords from case data...")
        print(f"   Case Type: {case_data.get('case_type', 'N/A')}")
        print(f"   Key Issues: {case_data.get('key_legal_issues', 'N/A')[:100]}...")
        
        keyword_response = call_llm(KEYWORD_EXTRACTION_PROMPT, case_summary)
        print(f"\n🤖 LLM Keyword Response:\n{keyword_response}")
        
        keywords_data = json.loads(keyword_response)
        search_query = " ".join(keywords_data.get("keywords", []))
        
        if not search_query:
            search_query = f"{case_data.get('case_type', '')} {case_data.get('key_legal_issues', '')}"[:100]
            print(f"⚠️  No keywords extracted, using fallback query")
        
        print(f"\n🔍 Search Query: '{search_query}'")
        
        # Step 2: Search Indian Kanoon
        print(f"\n🌐 Calling Indian Kanoon API...")
        search_results = search_indian_kanoon(search_query, max_results=10)
        
        print(f"\n📥 API Response received")
        print(f"   Response type: {type(search_results)}")
        print(f"   Response keys: {search_results.keys() if isinstance(search_results, dict) else 'Not a dict'}")
        
        docs = search_results.get("docs", [])
        print(f"   Number of docs found: {len(docs)}")
        
        if not docs:
            print("\n❌ No precedent cases found in Indian Kanoon")
            print("   Possible reasons:")
            print("   - Search query too specific")
            print("   - API returned empty results")
            print("   - Network/API error")
            return precedents
        
        print(f"\n✅ Found {len(docs)} potential precedents")
        for i, doc in enumerate(docs[:3], 1):
            print(f"   {i}. {doc.get('title', 'Untitled')[:60]}... (ID: {doc.get('tid', 'N/A')})")
        
        # Debug: Print complete structure of first doc to see available fields
        if docs:
            print(f"\n🔍 DEBUG - First document structure:")
            print(f"   Available keys: {list(docs[0].keys())}")
            for key, value in list(docs[0].items())[:10]:  # Show first 10 fields
                value_preview = str(value)[:100] if value else 'None'
                print(f"   {key}: {value_preview}")
        
        # Step 3: Analyze top 3 cases
        print(f"\n⚙️  Analyzing top 3 cases...")
        for idx, doc in enumerate(docs[:3], 1):
            try:
                docid = str(doc.get("tid", ""))  # Indian Kanoon uses 'tid' not 'docid'
                if not docid:
                    print(f"   ⚠️  Case {idx}: No tid found, skipping")
                    continue
                
                print(f"\n   📄 Analyzing Case {idx} (ID: {docid})...")
                
                # Fetch metadata
                metadata = get_case_metadata(docid)
                
                raw_title = metadata.get("title", "") or doc.get("title", "Untitled Case")
                
                # Extract date from title (usually at end: "on DD Month, YYYY")
                import re
                precedent_date = "Unknown"
                precedent_title = raw_title
                
                # Match patterns like "on 25 July, 2015" or "on 13 May, 2008"
                date_match = re.search(r'\bon\s+(\d{1,2}\s+\w+,?\s+\d{4})', raw_title)
                if date_match:
                    precedent_date = date_match.group(1).strip()
                    # Remove the date part from title
                    precedent_title = raw_title[:date_match.start()].strip()
                else:
                    # Fallback to year from publishdate
                    precedent_date = metadata.get("publishdate", "") if metadata.get("publishdate") else doc.get("publishdate", "Unknown")
                
                # Fetch COMPLETE document content first to extract court info
                doc_content = get_case_document(docid)
                
                # Smart court extraction from document content
                precedent_court = "Unknown Court"
                doc_preview = doc_content[:2000] if doc_content else ""
                
                # Priority 1: Supreme Court
                if "SUPREME COURT OF INDIA" in doc_preview.upper():
                    precedent_court = "Supreme Court of India"
                
                # Priority 2: High Courts (specific)
                elif "HIGH COURT" in doc_preview.upper():
                    # Common patterns: "Delhi High Court", "High Court of Delhi", etc.
                    high_court_patterns = [
                        r'(Delhi High Court|High Court of Delhi)',
                        r'(Bombay High Court|High Court of Bombay)',
                        r'(Calcutta High Court|High Court of Calcutta)',
                        r'(Madras High Court|High Court of Madras)',
                        r'(Karnataka High Court|High Court of Karnataka)',
                        r'(Kerala High Court|High Court of Kerala)',
                        r'(Punjab and Haryana High Court|High Court of Punjab and Haryana)',
                        r'(Allahabad High Court|High Court of Allahabad)',
                        r'(Rajasthan High Court|High Court of Rajasthan)',
                        r'(Gujarat High Court|High Court of Gujarat)',
                        r'(\w+\s+High Court)',  # Generic: "XYZ High Court"
                    ]
                    
                    for pattern in high_court_patterns:
                        match = re.search(pattern, doc_preview, re.IGNORECASE)
                        if match:
                            precedent_court = match.group(1)
                            break
                
                # Priority 3: District/Sessions/Other Courts
                elif any(court_type in doc_preview.upper() for court_type in ["DISTRICT COURT", "SESSIONS COURT", "ADDITIONAL SESSIONS JUDGE", "JUDICIAL MAGISTRATE"]):
                    # Extract court location from document
                    location_match = re.search(r'(District Court|Sessions Court|Court).*?(Delhi|Mumbai|Bangalore|Chennai|Kolkata|Hyderabad|Pune|[\w\s]+)', doc_preview, re.IGNORECASE)
                    if location_match:
                        precedent_court = f"District Court, {location_match.group(2).strip()}"
                    else:
                        # Try to find location from title or document
                        for line in doc_preview.split('\n')[:10]:
                            if "DELHI" in line.upper():
                                precedent_court = "District Court, New Delhi"
                                break
                            elif any(city in line.upper() for city in ["MUMBAI", "BANGALORE", "CHENNAI", "KOLKATA"]):
                                precedent_court = f"District Court, {line.split()[-1] if line.strip() else 'Unknown'}"
                                break
                
                # If still not found, try generic extraction but avoid judge names
                if precedent_court == "Unknown Court":
                    for line in doc_preview.split('\n')[:15]:
                        line_clean = line.strip()
                        # Skip lines with judge names (contain MS., MR., JUSTICE, HON'BLE with person names)
                        if any(avoid in line_clean.upper() for avoid in ["MS.", "MR.", "MRS.", "HON'BLE", "JUSTICE", "ASJ", "ACJ", "ADJ"]):
                            continue
                        if "COURT" in line_clean.upper() and len(line_clean) < 100:
                            precedent_court = line_clean
                            if precedent_court.startswith("IN THE "):
                                precedent_court = precedent_court[7:]
                            break
                
                print(f"      Title: {precedent_title[:60]}...")
                print(f"      Court: {precedent_court}")
                print(f"      Date: {precedent_date}")
                
                # Use full content (limit to 12000 chars to stay within token limits)
                # This gives LLM much more context for accurate similarity analysis
                full_precedent_content = doc_content[:12000] if doc_content else doc.get("headline", "No content available")
                
                print(f"      Content length: {len(full_precedent_content)} chars (full document)")
                
                # Analyze with LLM - provide FULL case context for both sides
                current_case = f"""Case Title: {case_data.get('case_title')}
Case Type: {case_data.get('case_type')}
Key Issues: {case_data.get('key_legal_issues')}
Court: {case_data.get('court_level')} in {case_data.get('jurisdiction')}
Opponent Profile: {case_data.get('opponent_profile')}
Claim Amount: ₹{case_data.get('claim_amount', 'N/A')}

Full Document Context (first 5000 chars):
{case_context[:5000] if case_context and len(case_context) > 50 else 'No document context'}"""
                
                analysis_prompt = PRECEDENT_ANALYSIS_PROMPT.format(
                    current_case=current_case,
                    precedent_title=precedent_title,
                    precedent_court=precedent_court,
                    precedent_year=precedent_date,  # Use full date instead of just year
                    precedent_content=full_precedent_content
                )
                
                print(f"      🤖 Requesting LLM analysis...")
                analysis_response = call_llm(analysis_prompt, "")
                
                # Parse LLM response - handle markdown code blocks
                response_text = analysis_response.strip() if analysis_response else "{}"
                if '```json' in response_text:
                    response_text = response_text.split('```json')[1].split('```')[0]
                elif '```' in response_text:
                    response_text = response_text.split('```')[1].split('```')[0]
                
                print(f"      📄 LLM response: {response_text[:150]}...")
                
                analysis = json.loads(response_text.strip())
                
                print(f"      ✅ Similarity: {analysis.get('similarity_score', 0)}%")
                print(f"         Outcome: {analysis.get('outcome', 'Unknown')}")
                
                # Store precedent data
                precedents.append({
                    "case_name": precedent_title,
                    "court": str(precedent_court),  # Ensure it's always a string
                    "year": str(precedent_date),    # Store full date (e.g., "25 July, 2015")
                    "similarity": analysis.get("similarity_score", 0),
                    "outcome": analysis.get("outcome", "Unknown"),
                    "description": analysis.get("relevance_explanation", "")
                })
                
            except Exception as e:
                print(f"   ❌ Error analyzing case {idx} (ID: {docid}): {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Step 4: Calculate weights for exactly 3 precedents
        print(f"\n⚖️  Calculating weights...")
        print(f"   Total precedents collected: {len(precedents)}")
        
        if len(precedents) == 3:
            print(f"   ✅ Exactly 3 precedents - calculating weights")
            # Calculate weights based on similarity and court hierarchy
            court_weights = {
                "Supreme Court": 1.5,
                "High Court": 1.2,
                "supreme": 1.5,
                "high": 1.2,
            }
            
            for prec in precedents:
                court_multiplier = 1.0
                court_name = str(prec.get("court", "")).lower()  # Convert to string safely
                
                for key, weight in court_weights.items():
                    if key.lower() in court_name:
                        court_multiplier = weight
                        break
                
                # Raw score: similarity * court importance
                prec["raw_score"] = prec["similarity"] * court_multiplier
                print(f"   - {prec['case_name'][:40]}...: similarity={prec['similarity']}% × multiplier={court_multiplier} = raw_score={prec['raw_score']}")
            
            # Normalize weights to sum to 100%
            total_raw = sum(p["raw_score"] for p in precedents)
            if total_raw > 0:
                for prec in precedents:
                    calculated_weight = (prec["raw_score"] / total_raw) * 100
                    # Ensure weight never exceeds similarity
                    prec["weight"] = min(int(calculated_weight), prec["similarity"])
                
                # Adjust weights to ensure they sum to 100%
                current_sum = sum(p["weight"] for p in precedents)
                if current_sum != 100:
                    # Distribute remainder to highest similarity case
                    precedents[0]["weight"] += (100 - current_sum)
                    
                print(f"   ✅ Final weights sum to 100%")
            else:
                # Fallback: equal weights
                for i, prec in enumerate(precedents):
                    prec["weight"] = 33 if i < 2 else 34
                print(f"   ⚠️  Using fallback equal weights")
            
            # Remove raw_score from final output
            for prec in precedents:
                prec.pop("raw_score", None)
        elif len(precedents) < 3:
            print(f"   ⚠️  Only {len(precedents)} precedents found (need 3)")
        
        print(f"\n🎯 FINAL RESULT: {len(precedents)} precedent cases ready")
        for i, prec in enumerate(precedents, 1):
            print(f"   {i}. {prec['case_name'][:50]}... (Weight: {prec.get('weight', 'N/A')}%, Similarity: {prec['similarity']}%)")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR in precedent analysis: {e}")
        import traceback
        traceback.print_exc()
        print("="*80 + "\n")
    
    return precedents


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
    
    # Store uploaded filename in session
    session['uploaded_filename'] = filename
    
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
    uploaded_filename = session.get('uploaded_filename', None)
    return render_template("form.html", fields=fields, uploaded_filename=uploaded_filename)


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
    
    # Process uploaded documents from Document Upload section
    uploaded_files = request.files.getlist('supporting_documents')
    doc_texts = []
    doc_names = request.form.getlist('doc_names')
    
    if uploaded_files:
        for file in uploaded_files:
            if file and file.filename:
                # Save to temp location
                temp_dir = tempfile.mkdtemp()
                filename = secure_filename(file.filename)
                filepath = os.path.join(temp_dir, filename)
                file.save(filepath)
                
                # Extract text based on file type
                if filename.lower().endswith('.pdf'):
                    text = extract_text_from_pdf(filepath)
                elif filename.lower().endswith('.docx'):
                    text = extract_text_from_docx(filepath)
                else:
                    text = ""
                
                if text.strip():
                    doc_texts.append({
                        'filename': filename,
                        'text': text
                    })
                
                # Clean up temp file
                shutil.rmtree(temp_dir)
    
    # Check if document was uploaded from Upload Document section (auto-extract)
    has_document = 'session_id' in session
    
    # Build comprehensive context from all sources
    all_context_parts = []
    
    # Add uploaded document context (from Upload Document section)
    if has_document:
        session_id = session['session_id']
        chunks = load_chunks_from_disk(session_id)
        if chunks:
            upload_context = retrieve_context(session_id, chunks, top_k=15)
            all_context_parts.append(f"Primary Document (Auto-Extracted):\n{upload_context}")
    
    # Add supporting documents context (from Document Upload section)
    if doc_texts:
        for doc in doc_texts:
            all_context_parts.append(f"Supporting Document - {doc['filename']}:\n{doc['text'][:5000]}")
    
    # Combine all context
    if all_context_parts:
        context = "\n\n---\n\n".join(all_context_parts)
    else:
        # No documents - use form data only
        context = f"""Case Title: {case_data['case_title']}
Case Type: {case_data['case_type']}
Jurisdiction: {case_data['jurisdiction']}
Court Level: {case_data['court_level']}
Claim Amount: {case_data['claim_amount']}
Opponent Counsel: {case_data['opponent_counsel']}
Opponent Profile: {case_data['opponent_profile']}
Key Legal Issues: {case_data['key_legal_issues']}"""
    
    # Combine case data with context for prediction
    full_context = f"""Case Information:
Case Title: {case_data['case_title']}
Case Type: {case_data['case_type']}
Jurisdiction: {case_data['jurisdiction']}
Court Level: {case_data['court_level']}
Claim Amount: ₹{case_data['claim_amount']}
Opponent Counsel: {case_data['opponent_counsel']}
Opponent Profile: {case_data['opponent_profile']}
Key Legal Issues: {case_data['key_legal_issues']}

{'Document Evidence:' if (has_document or doc_texts) else 'No Documents Provided'}
{context}
"""
    
    # Call LLM for litigation prediction
    print("\n" + "="*80)
    print("GENERATING LITIGATION PREDICTION...")
    print("="*80)
    
    llm_response = call_llm(PREDICTION_PROMPT, full_context)
    
    # Parse LLM prediction response
    try:
        # Extract JSON from response (in case LLM adds extra text)
        response_text = llm_response.strip() if llm_response else "{}"
        if '```json' in response_text:
            response_text = response_text.split('```json')[1].split('```')[0]
        elif '```' in response_text:
            response_text = response_text.split('```')[1].split('```')[0]
        
        predictions = json.loads(response_text.strip())
        
        # Print all predictions to terminal
        print("\n" + "="*80)
        print("LITIGATION PREDICTION RESULTS - ALL MODELS")
        print("="*80)
        
        for model in ['balanced', 'aggressive', 'conservative']:
            if model in predictions:
                pred = predictions[model]
                print(f"\n{model.upper()} MODEL:")
                print(f"  Win Probability: {pred.get('win_probability', 'N/A')}%")
                print(f"  Confidence Level: {pred.get('confidence_level', 'N/A')}%")
                print(f"  Discovery Timeline: {pred.get('discovery_timeline', 'N/A')}")
                print(f"  Mediation Timeline: {pred.get('mediation_timeline', 'N/A')}")
                print(f"  Trial/Settlement Duration: {pred.get('duration', 'N/A')}")
                print(f"  Estimated Legal Costs: {pred.get('estimated_costs', 'N/A')}")
                print(f"  Risk Tag: {pred.get('risk_tag', 'N/A')}")
        
        print("="*80 + "\n")
        
        # Ensure all three models exist with defaults
        default_prediction = {
            "win_probability": 50,
            "confidence_level": 65,
            "discovery_timeline": "4-6 months",
            "mediation_timeline": "8-10 months",
            "duration": "1-2 years",
            "estimated_costs": "₹1,00,000 - ₹3,00,000",
            "risk_tag": "Medium Risk"
        }
        
        if 'balanced' not in predictions:
            predictions['balanced'] = default_prediction.copy()
        if 'aggressive' not in predictions:
            predictions['aggressive'] = {
                "win_probability": 70,
                "confidence_level": 75,
                "discovery_timeline": "2-4 months",
                "mediation_timeline": "5-7 months",
                "duration": "6-12 months",
                "estimated_costs": "₹75,000 - ₹2,00,000",
                "risk_tag": "Low Risk"
            }
        if 'conservative' not in predictions:
            predictions['conservative'] = {
                "win_probability": 35,
                "confidence_level": 55,
                "discovery_timeline": "6-9 months",
                "mediation_timeline": "10-14 months",
                "duration": "1.5-3 years",
                "estimated_costs": "₹1,50,000 - ₹4,50,000",
                "risk_tag": "High Risk"
            }
        
        # Store all predictions
        analysis = predictions
        
        # Fetch and analyze precedent cases
        print("\n" + "="*80)
        print("FETCHING SUPPORTING PRECEDENTS FROM INDIAN KANOON...")
        print("="*80)
        
        precedents = fetch_and_analyze_precedents(case_data, full_context)
        
        # Add precedents to each model's analysis
        for model in ['balanced', 'aggressive', 'conservative']:
            if model in analysis:
                analysis[model]['precedents'] = precedents
            
    except Exception as e:
        print(f"\nError parsing prediction response: {e}")
        print(f"Raw LLM response: {llm_response}\n")
        # Provide default predictions for all three models
        analysis = {
            "balanced": {
                "win_probability": 50,
                "confidence_level": 65,
                "discovery_timeline": "4-6 months",
                "mediation_timeline": "8-10 months",
                "duration": "1-2 years",
                "estimated_costs": "₹1,00,000 - ₹3,00,000",
                "risk_tag": "Medium Risk"
            },
            "aggressive": {
                "win_probability": 70,
                "confidence_level": 75,
                "discovery_timeline": "2-4 months",
                "mediation_timeline": "5-7 months",
                "duration": "6-12 months",
                "estimated_costs": "₹75,000 - ₹2,00,000",
                "risk_tag": "Low Risk"
            },
            "conservative": {
                "win_probability": 35,
                "confidence_level": 55,
                "discovery_timeline": "6-9 months",
                "mediation_timeline": "10-14 months",
                "duration": "1.5-3 years",
                "estimated_costs": "₹1,50,000 - ₹4,50,000",
                "risk_tag": "High Risk"
            }
        }
    
    # Store prediction in session
    session['case_data'] = case_data
    session['analysis'] = analysis
    
    # Store document names in session
    if not doc_names and doc_texts:
        doc_names = [doc['filename'] for doc in doc_texts]
    session['uploaded_documents'] = doc_names
    
    return redirect(url_for("results"))


@app.route("/results")
def results():
    if 'analysis' not in session:
        return redirect(url_for("index"))
    
    case_data = session.get('case_data', {})
    analysis = session['analysis']
    documents = session.get('uploaded_documents', [])
    
    return render_template("results.html", case_data=case_data, analysis=analysis, documents=documents)


@app.route("/delete-uploaded-doc", methods=["POST"])
def delete_uploaded_doc():
    """Delete the uploaded filename from session"""
    if 'uploaded_filename' in session:
        del session['uploaded_filename']
    return jsonify({"success": True})


if __name__ == "__main__":
    app.run(debug=True)
