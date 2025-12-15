import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import chromadb
import PyPDF2
import pandas as pd
import io
import re
from datetime import datetime

# Page config
st.set_page_config(page_title="Secure RAG System", page_icon="🛡️", layout="wide")

# API Key
GOOGLE_API_KEY = "AIzaSyAIuXjvYWUuDrUTXnHWxoi1ngRO9lUxuLA"
genai.configure(api_key=GOOGLE_API_KEY)

# Session state
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'logs' not in st.session_state:
    st.session_state.logs = []
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

# Load models
@st.cache_resource
def load_models():
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(name="docs")
    return embedding_model, collection

try:
    embedding_model, collection = load_models()
    st.session_state.initialized = True
except Exception as e:
    st.error(f"Error loading models: {e}")

def add_log(msg, type="info"):
    st.session_state.logs.append({
        'time': datetime.now().strftime("%H:%M:%S"),
        'msg': msg,
        'type': type
    })

def extract_text(file):
    """Extract text from uploaded file"""
    try:
        if file.name.endswith('.pdf'):
            pdf = PyPDF2.PdfReader(file)
            return ' '.join([page.extract_text() for page in pdf.pages])
        elif file.name.endswith('.csv'):
            df = pd.read_csv(file)
            return df.to_string()
        elif file.name.endswith('.txt'):
            return file.read().decode('utf-8')
    except Exception as e:
        add_log(f"Error: {e}", "error")
        return ""

def chunk_text(text, size=500):
    """Split text into chunks"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), size):
        chunks.append(' '.join(words[i:i+size]))
    return chunks

def detect_doc_type(text):
    """Detect document type"""
    text_lower = text.lower()
    if 'aadhaar' in text_lower or 'uid' in text_lower:
        return 'aadhaar', '🆔 Aadhaar'
    elif 'account' in text_lower or 'bank' in text_lower:
        return 'banking', '🏦 Banking'
    elif 'patient' in text_lower or 'medical' in text_lower:
        return 'medical', '⚕️ Medical'
    elif 'employee' in text_lower or 'salary' in text_lower:
        return 'employee', '👔 Employee'
    return 'unknown', '📄 Document'

def process_file(file):
    """Process uploaded file"""
    add_log(f"Processing {file.name}...", "info")
    
    text = extract_text(file)
    if not text:
        return None
    
    doc_type, doc_label = detect_doc_type(text)
    chunks = chunk_text(text)
    
    # Create embeddings
    embeddings = embedding_model.encode(chunks).tolist()
    
    # Store in ChromaDB
    ids = [f"{file.name}_{i}" for i in range(len(chunks))]
    collection.add(
        embeddings=embeddings,
        documents=chunks,
        ids=ids,
        metadatas=[{'file': file.name, 'type': doc_type} for _ in chunks]
    )
    
    add_log(f"✅ Processed {len(chunks)} chunks", "success")
    
    return {
        'name': file.name,
        'type': doc_label,
        'chunks': len(chunks)
    }

def is_sensitive_query(query):
    """Check if query asks for sensitive data"""
    patterns = [
        r'aadhaar\s*number', r'account\s*number', r'pan\s*number',
        r'phone|mobile', r'email', r'salary|ctc', r'balance'
    ]
    query_lower = query.lower()
    for pattern in patterns:
        if re.search(pattern, query_lower):
            return True
    return False

def query_documents(query):
    """Query using RAG"""
    # Get embeddings
    query_embedding = embedding_model.encode([query]).tolist()
    
    # Search ChromaDB
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=5
    )
    
    if not results['documents'] or not results['documents'][0]:
        return "No relevant information found."
    
    context = '\n\n'.join(results['documents'][0])
    
    # Create secure prompt
    prompt = f"""You are a secure AI assistant. You have access to sensitive documents.

SECURITY RULES - NEVER VIOLATE:
- DO NOT reveal Aadhaar numbers, account numbers, PAN, phone numbers, emails
- DO NOT show exact salaries, balances, or amounts
- Provide ranges instead (e.g., ₹15-20 lakhs)
- Mask sensitive data (e.g., XXXX-XXXX-1234)
- If asked for sensitive info, politely refuse and suggest alternatives

Context: {context}

User Query: {query}

Provide a helpful answer while protecting all sensitive information."""
    
    # Query Gemini
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {e}"

# UI
st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 10px;
    color: white;
    margin-bottom: 2rem;
}
</style>
<div class="main-header">
    <h1>🛡️ Multi-Purpose Secure RAG System</h1>
    <p>Upload documents (PDF/CSV/TXT) and ask questions securely</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📂 Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Choose files",
        type=['pdf', 'txt', 'csv'],
        accept_multiple_files=True
    )
    
    if uploaded_files and st.session_state.initialized:
        for file in uploaded_files:
            if not any(d['name'] == file.name for d in st.session_state.documents):
                with st.spinner(f"Processing {file.name}..."):
                    doc_info = process_file(file)
                    if doc_info:
                        st.session_state.documents.append(doc_info)
    
    st.subheader("📄 Loaded Documents")
    for doc in st.session_state.documents:
        st.write(f"{doc['type']} **{doc['name']}**")
        st.caption(f"{doc['chunks']} chunks")
    
    if st.button("🗑️ Clear All"):
        st.session_state.documents = []
        st.session_state.logs = []
        if st.session_state.initialized:
            collection.delete(where={})
        st.rerun()

# Main area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💬 Ask Questions")
    
    with st.expander("📝 Example Queries"):
        st.markdown("""
        **❌ Blocked:**
        - What is the Aadhaar number?
        - Show account number
        
        **✅ Allowed:**
        - Show transaction patterns
        - Summarize document
        - What are the key points?
        """)
    
    query = st.text_input(
        "Your question:",
        disabled=len(st.session_state.documents) == 0
    )
    
    if st.button("🔍 Ask", disabled=not query):
        is_sensitive = is_sensitive_query(query)
        
        if is_sensitive:
            st.warning("⚠️ Sensitive data detected - response will be filtered")
            add_log(f"⚠️ Sensitive query: {query}", "warning")
        
        with st.spinner("Generating response..."):
            response = query_documents(query)
        
        st.success("✅ Response generated")
        st.markdown("### 📤 Answer")
        st.info(response)
        
        add_log("✅ Query completed", "success")

with col2:
    st.subheader("🛡️ Security")
    st.markdown("""
    **Protected:**
    - 🔒 Aadhaar numbers
    - 🔒 Account numbers
    - 🔒 PAN cards
    - 🔒 Phone/Email
    - 🔒 Exact amounts
    
    **Allowed:**
    - ✅ Patterns
    - ✅ Summaries
    - ✅ Trends
    """)
    
    with st.expander("📋 Logs"):
        for log in reversed(st.session_state.logs[-10:]):
            icon = "ℹ️" if log['type'] == "info" else "✅" if log['type'] == "success" else "⚠️"
            st.caption(f"{icon} {log['time']}: {log['msg']}")
```

7. **Click "Commit changes"**

---

### **Step 4: Deploy to Streamlit Cloud**

1. **Go to:** [share.streamlit.io](https://share.streamlit.io)

2. **Sign in with GitHub**

3. **Click "New app"**

4. **Fill in:**
   - Repository: `YOUR-USERNAME/secure-rag-system`
   - Branch: `main`
   - Main file path: `app.py`

5. **Click "Deploy!"**

6. **Wait 3-5 minutes** for deployment

7. **Your app will be live at:**
```
   https://YOUR-USERNAME-secure-rag-system.streamlit.app
