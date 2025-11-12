from flask import Flask, render_template, request, redirect, url_for, session, flash, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import os
import json
import pandas as pd
import re
import chromadb
import csv
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnablePassthrough
from datetime import datetime
from pypdf import PdfReader
import docx2txt
from builtins import enumerate

import torch
import pickle
import faiss
import requests
import unicodedata
import numpy as np
from transformers import BertTokenizer, BertModel

# === IMPORTS POUR L'ANALYSE DE CV ===
import spacy
from pdfminer.high_level import extract_text
from docx import Document as DocxDocument
import pytesseract
from PIL import Image

app = Flask(__name__)
app.secret_key = 'changez_cette_cle_secrete_en_production_123456789'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs('uploads/cvs', exist_ok=True)
os.makedirs('data', exist_ok=True)

USERS_FILE = 'data/users.json'
JOBS_FILE = 'data/jobs.json'
APPLICATIONS_FILE = 'data/applications.json'
CSV_JOBS_FILE = 'C:\\Users\\Guide Info\\Downloads\\new_jobs.csv'
CHROMA_DB_PATH = './chroma_db_jobs'

# === CONFIGURATION CHATBOT ===
DATA_PATH = "C:\\Users\\Guide Info\\Downloads\\CVs_sans_doublons_Final.xlsx"
INDEX_PATH = "C:\\Users\\Guide Info\\Downloads\\faiss_index_cvs.index"
CHUNKS_PATH = "C:\\Users\\Guide Info\\Downloads\\chunk_df.pkl"

OPENAI_API_KEY = "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"
OPENAI_API_BASE = "https://api.groq.com/openai/v1"
MODEL = "llama-3.1-8b-instant"
TOP_K = 5
DEFAULT_TEMPERATURE = 0.25

# === INITIALISATION BERT ===
device = torch.device("cpu")
torch.set_grad_enabled(False)

TOKENIZER_ID = "bert-base-uncased"
MODEL_ID = "bert-base-uncased"

print("🔹 Chargement du modèle BERT...")
tokenizer = BertTokenizer.from_pretrained(TOKENIZER_ID)
bert_model = BertModel.from_pretrained(MODEL_ID).to(device)
bert_model.eval()
print("✅ Modèle BERT chargé.")

# === INITIALISATION SPACY POUR L'ANALYSE DE CV ===
try:
    nlp_en = spacy.load('en_core_web_sm')
    nlp_fr = spacy.load('fr_core_news_sm')
    print("✅ Modèles SpaCy chargés pour l'analyse de CV")
except OSError as e:
    print("Error: SpaCy models missing. Please run:")
    print("python -m spacy download en_core_web_sm")
    print("python -m spacy download fr_core_news_sm")
    raise e

# === CONFIGURATION POUR L'ANALYSE DE CV ===
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'jpg', 'jpeg', 'png'}
GROQ_API_KEY = "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"
GROQ_API_BASE = "https://api.groq.com/openai/v1"
SELECTED_MODEL = "llama-3.1-8b-instant"

# === FONCTIONS CHATBOT ===
def embed_text(text: str):
    """Encode un texte avec BERT et normalise le vecteur"""
    if not isinstance(text, str) or not text.strip():
        return np.zeros((1, bert_model.config.hidden_size), dtype="float32")
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True).to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        vec = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    norm = np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12
    return (vec / norm).astype("float32")

def _normalize_text(s):
    if not isinstance(s, str):
        return ""
    s0 = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("utf-8")
    return s0.lower().strip()

def _format_bullets(answer):
    if not isinstance(answer, str):
        return ""
    txt = re.sub(r"(🧠|👤|📊)", r"\n\1 ", answer)
    txt = re.sub(r"\s*(?:•|-|\d+\.)\s*", "\n• ", txt)
    txt = re.sub(r"\n{2,}", "\n", txt).strip()
    return txt

# === CHARGEMENT FAISS + CHUNKS ===
if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
    with open(CHUNKS_PATH, "rb") as f:
        chunk_df = pickle.load(f)
    chunk_df = chunk_df.dropna(subset=["Nom", "Chunk"])
    faiss_index = faiss.read_index(INDEX_PATH)
    print(f"✅ Index FAISS chargé ({faiss_index.ntotal} vecteurs, dimension {faiss_index.d}).")
else:
    print("⚠️  Index FAISS ou chunk_df.pkl introuvable - chatbot désactivé")
    faiss_index = None
    chunk_df = None

def retrieve_context(query, top_k=TOP_K):
    if faiss_index is None or chunk_df is None:
        return "Aucun résumé pertinent trouvé.", pd.DataFrame(columns=["Nom", "Chunk", "sim"])
    
    q_emb = embed_text(query)
    if faiss_index.ntotal == 0 or q_emb.size == 0:
        return "Aucun résumé pertinent trouvé.", pd.DataFrame(columns=["Nom", "Chunk", "sim"])
    D, I = faiss_index.search(q_emb, top_k * 10)
    valid_idx = [i for i in I[0] if i >= 0]
    if not valid_idx:
        return "Aucun résumé pertinent trouvé.", pd.DataFrame(columns=["Nom", "Chunk", "sim"])
    res = chunk_df.iloc[valid_idx].copy()
    res["sim"] = [D[0][k] for k, i in enumerate(I[0]) if i in valid_idx]
    res = res.sort_values("sim", ascending=False).head(top_k)
    ctx = "\n".join([f"• {r['Nom']} — {r.get('Chunk', '')[:150]}" for _, r in res.iterrows()])
    return (ctx if ctx.strip() else "Aucun résumé pertinent trouvé."), res[["Nom", "Chunk", "sim"]]

def call_groq_api_chatbot(prompt, temperature=DEFAULT_TEMPERATURE):
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    data = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 400, "temperature": temperature}
    r = requests.post(f"{OPENAI_API_BASE}/chat/completions", headers=headers, json=data, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()

FEW_SHOT = """Tu es RecruitBot, un assistant RH professionnel et concis.
🧩 Format attendu :
🧠 <phrase de synthèse>
👤 - **Nom** — compétences clés
📊 <phrase de conclusion>
"""

def build_prompt(query, context):
    return f"{FEW_SHOT}\nQuestion : {query}\nContexte : {context}"

def hr_llm_chat(query):
    ctx, files = retrieve_context(query)
    prompt = build_prompt(query, ctx)
    answer = call_groq_api_chatbot(prompt)
    return _format_bullets(answer), files

# === FONCTIONS POUR L'ANALYSE DE CV ===
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_file(file_path):
    if not os.path.exists(file_path):
        return "Error: File not found."
    ext = file_path.lower().split('.')[-1]
    try:
        if ext == 'pdf':
            return extract_text(file_path)
        elif ext == 'docx':
            doc = DocxDocument(file_path)
            return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        elif ext in ['jpg', 'jpeg', 'png']:
            img = Image.open(file_path).convert('L')
            return pytesseract.image_to_string(img, lang='eng+fra')
        else:
            return "Unsupported format."
    except Exception as e:
        return f"Extraction error: {str(e)}"

def clean_text(text, full_names=[]):
    if not text or "Error" in text:
        return text

    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')

    # Emails
    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    emails = re.findall(email_pattern, text)
    for email in emails:
        text = text.replace(email, "EMAIL_PROTECTED")

    # Téléphones (format suisse)
    phone_pattern = r"\+41\(\d\)\d{2}\s+\d{6,7}"
    phones = re.findall(phone_pattern, text)
    for i, phone in enumerate(phones):
        text = text.replace(phone, f"PHONE_{i}_PROTECTED")

    # Protéger noms
    for name in full_names:
        text = text.replace(name, name.replace(' ', '_'))

    # Nettoyage ponctuation
    text = re.sub(r"[^\\w\\s@.+\\-\\(\\)_]", "", text)
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenisation
    doc_fr = nlp_fr(text)
    doc_en = nlp_en(text)
    cleaned_fr = ' '.join([t.text for t in doc_fr if not t.is_stop and not t.is_punct])
    cleaned_en = ' '.join([t.text for t in doc_en if not t.is_stop and not t.is_punct])
    cleaned_text = cleaned_fr if len(cleaned_fr) > len(cleaned_en) else cleaned_en

    # Restaurer
    for email in emails:
        cleaned_text = cleaned_text.replace("EMAIL_PROTECTED", email)
    for i, phone in enumerate(phones):
        cleaned_text = cleaned_text.replace(f"PHONE_{i}_PROTECTED", phone)
    for name in full_names:
        cleaned_text = cleaned_text.replace(name.replace(' ', '_'), name)

    return cleaned_text.strip()

def extract_name(raw_text):
    match = re.search(r"(\b[A-Z][a-z]+)\s+([A-Z][a-z]+)", raw_text)
    return match.group(0) if match else "Unknown"

def extract_email(raw_text):
    match = re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", raw_text)
    return match.group(0) if match else "Not found"

def extract_phone(raw_text):
    match = re.search(r"\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}", raw_text)
    return match.group(0) if match else "Not found"

def extract_skills_automatically(raw_text):
    if not raw_text or not isinstance(raw_text, str):
        return []

    exclude_terms = [
        'variety', 'ability', 'experience', 'project', 'formation', 'course', 'month', 'weeks',
        'service', 'manager', 'doctor', 'internal', 'and', 'in', 'at', 'to', 'under', 'of', 'the',
        'personal', 'title', 'college', 'university', 'degree', 'nationality', 'status', 'location',
        'address', 'tel', 'phone', 'email', 'summary', 'citizen', 'visa', 'fmh', 'ers', 'curriculum'
    ]

    # Appel Groq API
    prompt = (
        "Extract only technical skills from this resume. "
        "Return as comma-separated list. "
        "No explanations. No titles. No degrees."
    )
    result = call_groq_api_cv(prompt, raw_text[:2000])
    if result and isinstance(result, str):
        skills = [s.strip() for s in result.split(',') if s.strip() and len(s) > 1]
        skills = [s for s in skills if not any(ex in s.lower() for ex in exclude_terms)]
        if skills:
            return sorted(skills)

    # Fallback NLP
    skills = set()
    doc = nlp_en(raw_text) if 'summary' in raw_text.lower() else nlp_fr(raw_text)

    for chunk in doc.noun_chunks:
        skill = chunk.text.strip()
        if (len(skill) > 2 and
            not any(ex in skill.lower() for ex in exclude_terms) and
            not re.match(r'.*@.*\..*', skill) and
            not re.match(r'\+\d+.*', skill)):
            skills.add(skill)

    return sorted(list(skills))

def extract_education_from_resume(raw_text):
    if not raw_text or not isinstance(raw_text, str):
        return []

    pattern = r"(?i)(B\.?Sc\.?|M\.?Sc\.?|Ph\.?D\.?|Bachelor|Master|Doctorat|Licence|Diplôme|Maturité|Titre FMH)"
    matches = re.findall(pattern, raw_text)
    cleaned = [re.sub(r'\s+', ' ', m.strip()) for m in matches]
    cleaned = [re.sub(r'^\d{4}.*', '', m) for m in cleaned]
    cleaned = [m for m in cleaned if not any(ex in m.lower() for ex in ['database', 'gui', 'jenkins'])]
    cleaned = list(set(cleaned))
    return sorted(cleaned) if cleaned else []

def call_groq_api_cv(prompt, text):
    url = f"{GROQ_API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": SELECTED_MODEL,
        "messages": [{"role": "user", "content": f"{prompt}\n\n{text}"}],
        "max_tokens": 512,
        "temperature": 0.7
    }
    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()["choices"][0]["message"]["content"]
        return re.sub(r'^(Here is|Extracted|The following).*?:', '', result, flags=re.I).strip()
    except Exception as e:
        return f"API Error: {str(e)}"

def generate_recommendations(name, skills, education):
    skills_str = skills if skills else "No skills detected"
    education_str = education if education else "No education detected"

    prompt = f"""
    Profile:
    - Name: {name}
    - Skills: {skills_str}
    - Education: {education_str}

    Provide recommendations in English:
    • 3 suitable job positions
    • 3 skills to improve
    • 2 training courses or certifications
    """
    return call_groq_api_cv(prompt, "")

# === INIT FICHIERS ===
def init_json_files():
    for file in [USERS_FILE, JOBS_FILE, APPLICATIONS_FILE]:
        if not os.path.exists(file):
            with open(file, 'w') as f:
                if file == USERS_FILE:
                    default_rh = {
                        "id": "1", "name": "Admin RH", "email": "admin@rh.com",
                        "password": generate_password_hash("admin123"), "role": "rh"
                    }
                    json.dump([default_rh], f)
                else:
                    json.dump([], f)

init_json_files()

app.jinja_env.globals['enumerate'] = enumerate

# === EMBEDDINGS + LLM ===
embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="mistral")

# === CHARGER OU CRÉER CHROMA DB DEPUIS CSV ===
def load_or_create_chroma_db():
    if not os.path.exists(CSV_JOBS_FILE):
        print(f"CSV non trouvé : {CSV_JOBS_FILE}")
        return False

    df = pd.read_csv(CSV_JOBS_FILE)
    documents = []
    metadatas = []
    ids = []

    for idx, row in df.iterrows():
        title = str(row.get('Job Title', 'Inconnu'))
        desc = str(row.get('Job Description', ''))
        skills = str(row.get('Skills', ''))

        content = f"{title}\n{desc}\n{skills}".strip()

        job_id = f"job_{idx}"

        documents.append(content)
        metadatas.append({
            'id': job_id,
            'title': title,
            'description': desc,
            'skills': skills
        })
        ids.append(job_id)

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    try:
        client.delete_collection("jobs_collection")
    except:
        pass

    from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
    
    ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text"
    )
    
    collection = client.create_collection(
        name="jobs_collection",
        embedding_function=ollama_ef
    )
    
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    
    print(f"{len(documents)} jobs chargés dans Chroma")
    return True

def extract_cv_summary(cv_text):
    prompt_template = """
    Extraire une liste complète des technologies, langages, outils, frameworks, etc., mentionnés dans le texte du CV fourni. Formatez-le comme une liste numérotée par catégorie.

    Texte du CV:
    {cv_text}

    Résumé:
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["cv_text"])
    chain = PROMPT | llm
    response = chain.invoke({"cv_text": cv_text[:2000]})
    return response.content.strip()

# === UTILITAIRES ===
def load_json(file):
    with open(file, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(file, data):
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def extract_cv_text(file_path):
    if file_path.lower().endswith('.pdf'):
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    elif file_path.lower().endswith(('.docx', '.doc')):
        return docx2txt.process(file_path).strip()
    raise ValueError("Format non supporté")

# === SCORE ===
def get_score(cv_text, job):
    job_text = f"{job['title']} {job['description']} {job['skills']}".strip()

    prompt = PromptTemplate.from_template(
        """Tu es un expert RH. Donne UNIQUEMENT un nombre entre 0 et 100.

RÈGLES :
- Pas de phrase
- Pas de %
- Un seul nombre
- Exemple : 87.3

POSTE :
{job}

CV :
{cv}

SCORE :"""
    )

    try:
        response = (prompt | llm).invoke({
            "job": job_text[:1200],
            "cv": cv_text[:2500]
        })
        raw = response.content.strip()
        print(f"[SCORE] LLM → '{raw}'")

        match = re.search(r'\d+\.?\d*', raw)
        if match:
            score = float(match.group(0))
            return round(max(0, min(100, score)), 1)
        return 0.0
    except Exception as e:
        print(f"[ERREUR SCORE] {e}")
        return 0.0

def generate_quiz(cv_text, job, num_questions=4):
    job_title = job.get('title', 'Poste')
    job_skills = job.get('skills', '')
    job_desc = job.get('description', '')[:700]

    cv_skills_text = extract_cv_summary(cv_text)
    cv_skills = [s.strip().lower() for s in cv_skills_text.replace('\n', ',').split(',') if s.strip()]
    
    job_skills_list = [s.strip().lower() for s in job_skills.split(',') if s.strip()]
    gaps = [s for s in job_skills_list if s not in cv_skills]
    gaps_str = ', '.join(gaps[:5]) if gaps else "Aucune lacune détectée"

    prompt = f"""
Tu es un **coach RH expert**. Crée {num_questions} questions QCM **éducatives** pour aider un candidat à s'améliorer.

CV DU CANDIDAT :
{cv_skills_text}

POSTE VISÉ :
{job_title}
Compétences requises : {job_skills}
Description : {job_desc}

LACUNES DÉTECTÉES :
{gaps_str}

CRÉE :
- 2 questions sur les forces du CV
- 2 questions sur les gaps (pour apprendre)

FORMAT JSON STRICT :
[
  {{
    "question": "Quelle commande Git utilise-t-on pour fusionner ?",
    "options": ["A: git merge", "B: git push", "C: git pull", "D: git clone"],
    "reponse_correcte": "A",
    "explication": "git merge fusionne les branches. Tu as Git dans ton CV → bien joué !"
  }}
]
"""

    try:
        response = llm.invoke(prompt)
        raw = response.content.strip()
        print(f"[QUIZ IA] {raw[:200]}...")

        match = re.search(r'\[.*\]', raw, re.DOTALL)
        if match:
            quiz = json.loads(match.group(0))
            if len(quiz) >= 3:
                return quiz[:num_questions]
    except Exception as e:
        print(f"[QUIZ ERREUR] {e}")

    gap_example = gaps[0] if gaps else "nouvelles technologies"
    return [
        {
            "question": f"Avez-vous déjà utilisé {gap_example} ?",
            "options": [f"A: Oui, expert", f"B: Oui, débutant", "C: En cours d'apprentissage", "D: Non"],
            "reponse_correcte": "A",
            "explication": f"{gap_example} est requis pour ce poste. Si tu apprends, c'est un bon début !"
        },
        {
            "question": "Quelle est votre plus grande force technique ?",
            "options": ["A: Python", "B: Communication", "C: Design", "D: Gestion de projet"],
            "reponse_correcte": "A",
            "explication": "Python est dans ton CV → c'est un atout majeur !"
        },
        {
            "question": "Es-tu prêt à apprendre rapidement ?",
            "options": ["A: Oui, j'adore", "B: Oui", "C: Peut-être", "D: Non"],
            "reponse_correcte": "A",
            "explication": "L'envie d'apprendre compense les gaps !"
        },
        {
            "question": "Disponible dans 15 jours ?",
            "options": ["A: Oui", "B: Dans 1 mois", "C: Dans 2 mois", "D: Plus tard"],
            "reponse_correcte": "A",
            "explication": "Disponibilité rapide = gros avantage"
        }
    ]

# === FONCTIONS D'AUTHENTIFICATION ===
def is_authenticated():
    return 'user_id' in session

def get_current_user():
    if is_authenticated():
        return {
            'id': session.get('user_id'),
            'name': session.get('name'),
            'email': session.get('email'),
            'role': session.get('role'),
            'is_authenticated': True
        }
    return {'is_authenticated': False}

def login_required(role=None):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not is_authenticated():
                flash('Veuillez vous connecter pour accéder à cette page', 'error')
                return redirect(url_for('login'))
            
            if role and session.get('role') != role:
                flash('Accès non autorisé', 'error')
                return redirect(url_for('index'))
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# === FONCTIONS CHROMA DB CORRIGÉES ===
def get_chroma_collection():
    """Récupère la collection Chroma pour les jobs"""
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        
        from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
        
        ollama_ef = OllamaEmbeddingFunction(
            url="http://localhost:11434/api/embeddings",
            model_name="nomic-embed-text"
        )
        
        collection = client.get_collection(
            name="jobs_collection",
            embedding_function=ollama_ef
        )
        return collection
    except Exception as e:
        print(f"Erreur récupération collection Chroma: {e}")
        return None

def get_jobs_from_chroma():
    """Récupère tous les jobs depuis Chroma DB"""
    try:
        collection = get_chroma_collection()
        if collection is None:
            print("Collection Chroma non disponible")
            return []
        
        results = collection.get(include=["metadatas", "documents"])
        jobs = []
        
        for i, (meta, doc) in enumerate(zip(results['metadatas'], results['documents'])):
            job_id = meta.get('id') or f"job_{i}"
            jobs.append({
                'id': str(job_id),
                'title': meta.get('title', 'Sans titre'),
                'description': meta.get('description', ''),
                'skills': meta.get('skills', ''),
                'full_content': doc
            })
        
        print(f"{len(jobs)} jobs chargés depuis Chroma")
        return jobs
        
    except Exception as e:
        print(f"Erreur Chroma: {e}")
        return []

def add_job_to_chroma(document_id, text, metadata):
    """
    Ajoute un job à Chroma DB - VERSION CORRIGÉE
    """
    try:
        collection = get_chroma_collection()
        if collection is None:
            print("Collection Chroma non disponible")
            return False
        
        print(f"Ajout à Chroma - ID: {document_id}")
        
        collection.add(
            documents=[text],
            metadatas=[metadata],
            ids=[document_id]
        )
        
        print(f"Job {document_id} ajouté à Chroma DB avec succès")
        return True
        
    except Exception as e:
        print(f"Erreur lors de l'ajout à Chroma DB: {e}")
        import traceback
        traceback.print_exc()
        return False

def add_job_to_csv(title, skills, description):
    """
    Ajoute un job au fichier CSV avec gestion robuste des permissions
    """
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        csv_file_path = os.path.join(base_dir, 'data', 'jobs_data.csv')
        
        print(f"Tentative d'écriture dans: {csv_file_path}")
        
        os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
        
        file_exists = os.path.exists(csv_file_path)
        
        with open(csv_file_path, 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            
            if not file_exists:
                writer.writerow(['Job Title', 'Skills', 'Job Description'])
                print("En-tête CSV écrit")
            
            writer.writerow([title, skills, description])
            print(f"Job écrit dans CSV: {title}")
        
        return True
        
    except Exception as e:
        print(f"Erreur lors de l'écriture dans le CSV: {e}")
        return False

# === ROUTES PRINCIPALES ===
@app.route('/')
def index():
    return render_template('index-2.html', current_user=get_current_user())

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        role = request.form.get('role')
        users = load_json(USERS_FILE)
        user = next((u for u in users if u['email'] == email and u['role'] == role), None)
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['role'] = user['role']
            session['name'] = user['name']
            session['email'] = user['email']
            flash(f'Bienvenue {user["name"]}!', 'success')
            return redirect(url_for('reload_jobs' if role == 'rh' else 'candidat_dashboard'))
        flash('Identifiants incorrects', 'error')
    return render_template('login.html', current_user=get_current_user())

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        users = load_json(USERS_FILE)
        email = request.form.get('email')
        if any(u['email'] == email for u in users):
            flash('Email déjà utilisé', 'error')
            return render_template('register.html', current_user=get_current_user())
        new_user = {
            'id': str(len(users) + 1),
            'name': request.form.get('name'),
            'email': email,
            'password': generate_password_hash(request.form.get('password')),
            'role': request.form.get('role')
        }
        users.append(new_user)
        save_json(USERS_FILE, users)
        flash('Inscription réussie !', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', current_user=get_current_user())

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

# === ROUTES RH ===
@app.route('/rh/reload_jobs')
@login_required(role='rh')
def reload_jobs():
    success = load_or_create_chroma_db()
    flash('Jobs rechargés depuis CSV !' if success else 'Erreur CSV', 
          'success' if success else 'error')
    return redirect(url_for('rh_dashboard'))

@app.route('/rh/dashboard')
@login_required(role='rh')
def rh_dashboard():
    page = request.args.get('page', 1, type=int)
    search_query = request.args.get('search', '', type=str).strip()
    jobs_per_page = 9
    
    # Récupérer TOUS les jobs depuis Chroma
    all_jobs = get_jobs_from_chroma()
    applications = load_json(APPLICATIONS_FILE)
    
    # Filtrer les jobs par recherche si un terme est fourni (pour la pagination côté serveur)
    server_filtered_jobs = all_jobs
    if search_query:
        search_lower = search_query.lower()
        filtered_jobs = []
        for job in all_jobs:
            title_match = job.get('title', '').lower().find(search_lower) != -1
            description_match = job.get('description', '').lower().find(search_lower) != -1
            
            if title_match or description_match:
                filtered_jobs.append(job)
        server_filtered_jobs = filtered_jobs
    
    # Calculer les candidatures pour chaque job
    for job in all_jobs:
        job_candidatures = [a for a in applications if a.get('job_id') == job.get('id')]
        job['candidatures_count'] = len(job_candidatures)
    
    # Pour la pagination côté serveur, utiliser les jobs filtrés
    total_jobs = len(server_filtered_jobs)
    total_pages = (total_jobs + jobs_per_page - 1) // jobs_per_page if total_jobs > 0 else 1
    
    # S'assurer que la page demandée est valide
    page = max(1, min(page, total_pages))
    
    # Calculer les indices de début et fin pour la pagination
    start_index = (page - 1) * jobs_per_page
    end_index = start_index + jobs_per_page
    
    # Récupérer les jobs pour la page actuelle (pour l'affichage initial)
    jobs_for_page = server_filtered_jobs[start_index:end_index]
    
    return render_template('rh_dashboard.html', 
                         jobs=jobs_for_page,           # Jobs pour la page actuelle
                         all_jobs=all_jobs,            # TOUS les jobs pour la recherche côté client
                         current_user=get_current_user(),
                         page=page,
                         total_pages=total_pages,
                         total_jobs=total_jobs,
                         start_index=start_index + 1 if total_jobs > 0 else 0,
                         end_index=min(end_index, total_jobs),
                         search_query=search_query,
                         jobs_per_page=jobs_per_page)



@app.route('/download_cv/<path:filename>')
def download_cv(filename):
    return send_from_directory('uploads/cvs', filename, as_attachment=True)

@app.route('/rh/applications/<job_id>')
@login_required(role='rh')
def view_job_applications(job_id):
    jobs = get_jobs_from_chroma()
    job = next((j for j in jobs if j['id'] == job_id), None)
    if not job:
        flash('Offre introuvable', 'error')
        return redirect(url_for('rh_dashboard'))
    
    applications = load_json(APPLICATIONS_FILE)
    users = load_json(USERS_FILE)
    
    candidatures = []
    for a in applications:
        if a['job_id'] == job_id:
            user = next((u for u in users if u['id'] == a['user_id']), None)
            if user:
                a['user_email'] = user.get('email', 'No email provided')
            else:
                a['user_email'] = 'No email provided'
            
            if 'cv_path' in a:
                a['cv_filename'] = os.path.basename(a['cv_path'])
                a['cv_exists'] = os.path.exists(a['cv_path'])
            else:
                a['cv_exists'] = False
            
            candidatures.append(a)
    
    print(f"Candidatures pour le job {job_id}: {candidatures}")
    
    return render_template('view_job_applications.html', 
                         job=job, 
                         candidatures=candidatures,
                         current_user=get_current_user())

@app.route('/rh/add_job', methods=['GET', 'POST'])
@login_required(role='rh')
def add_job():
    if request.method == 'POST':
        job_title = request.form.get('title')
        job_description = request.form.get('description')
        job_skills = request.form.get('skills')
        
        if not all([job_title, job_description, job_skills]):
            flash('Veuillez remplir tous les champs', 'error')
            return render_template('add_job.html', current_user=get_current_user())
        
        job_id = str(int(datetime.now().timestamp()))
        
        print(f"=== DÉBUT AJOUT JOB ===")
        print(f"Titre: {job_title}")
        print(f"Skills: {job_skills}")
        print(f"Description: {job_description[:50]}...")
        
        chroma_success = add_job_to_chroma(
            document_id=job_id,
            text=f"{job_title} {job_description} {job_skills}",
            metadata={
                'id': job_id,
                'title': job_title,
                'description': job_description,
                'skills': job_skills,
                'created_by': session['user_id'],
                'created_at': datetime.now().isoformat(),
                'type': 'job_offer'
            }
        )
        
        csv_success = add_job_to_csv(job_title, job_skills, job_description)
        
        print(f"=== RÉSULTAT AJOUT ===")
        print(f"Chroma: {'SUCCÈS' if chroma_success else 'ÉCHEC'}")
        print(f"CSV: {'SUCCÈS' if csv_success else 'ÉCHEC'}")
        
        if chroma_success:
            if csv_success:
                flash('Offre ajoutée avec succès à Chroma DB et CSV', 'success')
            else:
                flash('Offre ajoutée à Chroma DB (erreur avec le fichier CSV)', 'warning')
            return redirect(url_for('rh_dashboard'))
        else:
            flash('Erreur lors de l\'ajout à la base de données', 'error')
            return render_template('add_job.html', current_user=get_current_user())
    
    return render_template('add_job.html', current_user=get_current_user())

@app.route('/rh/analyze_cv', methods=['GET', 'POST'])
@login_required(role='rh')
def analyze_cv():
    jobs = get_jobs_from_chroma()
    if not jobs:
        flash('Aucun job chargé. Clique sur Recharger', 'error')
        return render_template('analyze_cv.html', jobs=[], current_user=get_current_user())
    
    selected_job = None
    compatibility_score = None
    top_jobs = None
    
    if request.method == 'POST':
        cv_file = request.files.get('cv')
        job_id = request.form.get('job_id')
        if not cv_file or not job_id:
            flash('CV + poste requis', 'error')
            return render_template('analyze_cv.html', jobs=jobs, current_user=get_current_user())
        
        filename = secure_filename(cv_file.filename)
        cv_path = os.path.join('uploads/cvs', filename)
        cv_file.save(cv_path)
        cv_text = extract_cv_text(cv_path)
        
        job = next((j for j in jobs if j['id'] == job_id), None)
        if not job:
            flash('Job non trouvé', 'error')
            return render_template('analyze_cv.html', jobs=jobs, current_user=get_current_user())

        score = get_score(cv_text, job)

        top_jobs = []
        for j in jobs[:8]:
            try:
                s = get_score(cv_text, j)
                top_jobs.append({'job': j, 'score': s})
            except:
                continue
        top_jobs.sort(key=lambda x: x['score'], reverse=True)
        top_jobs = top_jobs[:5]
        
        selected_job = job
        compatibility_score = score
        
        flash('Analyse CV terminée avec succès!', 'success')
    
    return render_template('analyze_cv.html', 
                         jobs=jobs,
                         selected_job=selected_job,
                         compatibility_score=compatibility_score,
                         top_jobs=top_jobs,
                         current_user=get_current_user())

# === NOUVELLE ROUTE POUR L'ANALYSE AVANCÉE DES CV ===
@app.route('/candidate/analyse_cv_avancee', methods=['GET', 'POST'])
@login_required(role='candidat')
def analyse_cv_avancee():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # === TRAITEMENT AVANCÉ ===
            raw_text = extract_text_from_file(filepath)
            if "Error" in raw_text:
                flash(raw_text)
                os.remove(filepath)
                return redirect(request.url)

            name = extract_name(raw_text)
            email = extract_email(raw_text)
            phone = extract_phone(raw_text)

            skills_list = extract_skills_automatically(raw_text)
            education_list = extract_education_from_resume(raw_text)

            skills = ', '.join(skills_list) if skills_list else "No skills detected"
            education = ', '.join(education_list) if education_list else "No education detected"

            recommendations = generate_recommendations(name, skills, education)

            # Nettoyer
            if os.path.exists(filepath):
                os.remove(filepath)

            return render_template('result_analyse.html', 
                                 name=name,
                                 email=email,
                                 phone=phone,
                                 skills=skills,
                                 education=education,
                                 recommendations=recommendations,
                                 current_user=get_current_user())

    return render_template('analyse_cv_avancee.html', current_user=get_current_user())

# === ROUTES CANDIDAT ===
@app.route('/candidat/dashboard')
@login_required(role='candidat')
def candidat_dashboard():
    page = request.args.get('page', 1, type=int)
    search_query = request.args.get('search', '', type=str).strip()
    jobs_per_page = 9
    
    # Récupérer TOUS les jobs depuis Chroma
    all_jobs = get_jobs_from_chroma()
    applications = load_json(APPLICATIONS_FILE)
    
    # Filtrer les jobs par recherche si un terme est fourni (pour la pagination côté serveur)
    server_filtered_jobs = all_jobs
    if search_query:
        search_lower = search_query.lower()
        filtered_jobs = []
        for job in all_jobs:
            title_match = job.get('title', '').lower().find(search_lower) != -1
            description_match = job.get('description', '').lower().find(search_lower) != -1
            
            if title_match or description_match:
                filtered_jobs.append(job)
        server_filtered_jobs = filtered_jobs
    
    # Pour la pagination côté serveur, utiliser les jobs filtrés
    total_jobs = len(server_filtered_jobs)
    total_pages = (total_jobs + jobs_per_page - 1) // jobs_per_page if total_jobs > 0 else 1
    
    # S'assurer que la page demandée est valide
    page = max(1, min(page, total_pages))
    
    # Calculer les indices de début et fin pour la pagination
    start_index = (page - 1) * jobs_per_page
    end_index = start_index + jobs_per_page
    
    # Récupérer les jobs pour la page actuelle (pour l'affichage initial)
    jobs_for_page = server_filtered_jobs[start_index:end_index]
    
    # Récupérer les candidatures de l'utilisateur
    user_apps = [a for a in applications if a['user_id'] == session['user_id']]
    applied_job_ids = [a['job_id'] for a in user_apps]
    
    return render_template('candidat_dashboard.html', 
                         jobs=jobs_for_page,           # Jobs pour la page actuelle
                         all_jobs=all_jobs,            # TOUS les jobs pour la recherche côté client
                         applied_job_ids=applied_job_ids,
                         current_user=get_current_user(),
                         page=page,
                         total_pages=total_pages,
                         total_jobs=total_jobs,
                         start_index=start_index + 1 if total_jobs > 0 else 0,
                         end_index=min(end_index, total_jobs),
                         search_query=search_query,
                         jobs_per_page=jobs_per_page)
@app.route('/candidat/apply/<job_id>', methods=['GET', 'POST'])
@login_required(role='candidat')
def apply_job(job_id):
    jobs = get_jobs_from_chroma()
    job = next((j for j in jobs if j['id'] == job_id), None)
    if not job:
        flash('Offre non trouvée', 'error')
        return redirect(url_for('candidat_dashboard'))
    
    if request.method == 'POST':
        cv_file = request.files.get('cv')
        if not cv_file:
            flash('Veuillez uploader votre CV', 'error')
            return render_template('apply_job.html', job=job, current_user=get_current_user())
        
        filename = f"{session['user_id']}_{job_id}_{secure_filename(cv_file.filename)}"
        cv_path = os.path.join(app.config['UPLOAD_FOLDER'], 'cvs', filename)
        cv_file.save(cv_path)
        cv_text = extract_cv_text(cv_path)
        
        session['pending_application'] = {
            'job_id': job_id,
            'cv_path': cv_path
        }
        return redirect(url_for('take_quiz'))
    
    return render_template('apply_job.html', job=job, current_user=get_current_user())

@app.route('/candidat/quiz', methods=['GET', 'POST'])
@login_required(role='candidat')
def take_quiz():
    if 'pending_application' not in session:
        return redirect(url_for('candidat_dashboard'))
    
    pending = session['pending_application']
    jobs = get_jobs_from_chroma()
    job = next((j for j in jobs if j['id'] == pending['job_id']), None)
    if not job:
        flash('Offre introuvable', 'error')
        return redirect(url_for('candidat_dashboard'))

    cv_text = extract_cv_text(pending['cv_path'])
    quiz_questions = generate_quiz(cv_text, job)

    if request.method == 'POST':
        answers = [request.form.get(f'q{i}') for i in range(len(quiz_questions))]
        correct = 0
        
        for i, q in enumerate(quiz_questions):
            if i < len(answers) and answers[i] is not None:
                user_answer_index = int(answers[i])
                correct_answer_letter = q['reponse_correcte'].strip().upper()
                
                correct_answer_index = None
                for idx, opt in enumerate(q['options']):
                    if opt.startswith(correct_answer_letter + ':'):
                        correct_answer_index = idx
                        break
                
                if user_answer_index == correct_answer_index:
                    correct += 1

        quiz_score = round((correct / len(quiz_questions)) * 100, 1) if quiz_questions else 0
        cv_score = get_score(cv_text, job)

        applications = load_json(APPLICATIONS_FILE)
        applications.append({
            'id': str(len(applications) + 1),
            'user_id': session['user_id'],
            'user_name': session['name'],
            'job_id': job['id'],
            'job_title': job['title'],
            'cv_path': pending['cv_path'],
            'cv_score': cv_score,
            'quiz_score': quiz_score,
            'applied_at': datetime.now().isoformat()
        })
        save_json(APPLICATIONS_FILE, applications)
        session.pop('pending_application', None)

        return render_template('quiz_result.html',
                               job=job,
                               cv_score=cv_score,
                               quiz_score=quiz_score,
                               current_user=get_current_user())
    return render_template('quiz.html', 
                         job=job, 
                         quiz=quiz_questions,
                         current_user=get_current_user())

@app.route('/dashboard')
@login_required()
def dashboard():
    if session.get('role') == 'rh':
        return redirect(url_for('rh_dashboard'))
    elif session.get('role') == 'candidat':
        return redirect(url_for('candidat_dashboard'))
    else:
        flash('Rôle non reconnu', 'error')
        return redirect(url_for('index'))
    
@app.route('/profile')
@login_required()
def profile():
    return render_template('profile.html', current_user=get_current_user())

# === ROUTES CHATBOT ===
@app.route("/chat")
def chat():
    return render_template("chat.html", current_user=get_current_user())

@app.route("/ask-question", methods=["POST"])
def ask_question():
    try:
        data = request.get_json(force=True, silent=True) or {}
        question = (data.get("question") or "").strip()
        if not question:
            return jsonify({"answer": "Merci d'entrer une question.", "candidats": []})

        print(f"\n🟢 Question reçue : {question}")
        answer, files = hr_llm_chat(question)
        print(f"✅ Réponse générée : {answer[:150]}...")

        return jsonify({"answer": answer, "candidats": files["Nom"].tolist()})
    except Exception as e:
        import traceback
        print("❌ Traceback complet :")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/rh/insights_data")
@login_required(role="rh")
def rh_insights_data():
    """Retourne les statistiques RH sous forme JSON (pour le dashboard dynamique)"""
    jobs = get_jobs_from_chroma()
    applications = load_json(APPLICATIONS_FILE)

    job_stats = []
    total_scores = []
    for job in jobs:
        job_apps = [a for a in applications if a.get("job_id") == job.get("id")]
        job_stats.append({"title": job.get("title", "Sans titre"), "count": len(job_apps)})
        for a in job_apps:
            if "cv_score" in a:
                total_scores.append(a["cv_score"])

    top_jobs = sorted(job_stats, key=lambda x: x["count"], reverse=True)[:5]
    total_jobs = len(jobs)
    total_apps = len(applications)
    avg_score = round(sum(total_scores) / len(total_scores), 1) if total_scores else 0
    jobs_with_apps = len([j for j in job_stats if j["count"] > 0])

    score_dist = {
        "80-100": sum(1 for s in total_scores if s >= 80),
        "60-79": sum(1 for s in total_scores if 60 <= s < 80),
        "40-59": sum(1 for s in total_scores if 40 <= s < 60),
        "0-39": sum(1 for s in total_scores if s < 40),
    }

    timeline = sorted(applications, key=lambda a: a.get("applied_at", ""), reverse=True)[:6]
    timeline_data = [
        {
            "user": a.get("user_name", "Candidat inconnu"),
            "job": a.get("job_title", "Offre"),
            "date": a.get("applied_at", "")[:10],
        }
        for a in timeline
    ]

    return jsonify({
        "total_jobs": total_jobs,
        "total_apps": total_apps,
        "avg_score": avg_score,
        "jobs_with_apps": jobs_with_apps,
        "top_jobs": top_jobs,
        "score_dist": score_dist,
        "timeline": timeline_data,
    })

@app.route("/rh/insights")
@login_required(role="rh")
def rh_insights():
    """Page principale du tableau de bord dynamique RH"""
    return render_template("rh_insights.html", current_user=get_current_user())

if __name__ == '__main__':
    app.run(debug=True, port=5000)