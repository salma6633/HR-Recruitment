# app.py
from flask import Flask, render_template, request, flash, redirect, url_for
import os
import re
import unicodedata
import requests
from werkzeug.utils import secure_filename
from pdfminer.high_level import extract_text
from docx import Document
import pytesseract
from PIL import Image
import spacy

# === INITIALISATION FLASK ===
app = Flask(__name__)
app.secret_key = "cv_analyzer_secret"
app.config['UPLOAD_FOLDER'] = 'Uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

ALLOWED_EXTENSIONS = {'pdf', 'docx', 'jpg', 'jpeg', 'png'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# === CHARGER LES MODÈLES SPACY ===
try:
    nlp_en = spacy.load('en_core_web_sm')
    nlp_fr = spacy.load('fr_core_news_sm')
except OSError as e:
    print("Error: SpaCy models missing. Please run:")
    print("python -m spacy download en_core_web_sm")
    print("python -m spacy download fr_core_news_sm")
    raise e

# === CONFIG GROQ API ===
GROQ_API_KEY = "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"
GROQ_API_BASE = "https://api.groq.com/openai/v1"
SELECTED_MODEL = "llama-3.1-8b-instant"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# === EXTRACTION DE TEXTE ===
def extract_text_from_file(file_path):
    if not os.path.exists(file_path):
        return "Error: File not found."
    ext = file_path.lower().split('.')[-1]
    try:
        if ext == 'pdf':
            return extract_text(file_path)
        elif ext == 'docx':
            doc = Document(file_path)
            return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        elif ext in ['jpg', 'jpeg', 'png']:
            img = Image.open(file_path).convert('L')
            return pytesseract.image_to_string(img, lang='eng+fra')
        else:
            return "Unsupported format."
    except Exception as e:
        return f"Extraction error: {str(e)}"

# === NETTOYAGE DU TEXTE ===
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

# === EXTRACTION D'ENTITÉS ===
def extract_name(raw_text):
    match = re.search(r"(\b[A-Z][a-z]+)\s+([A-Z][a-z]+)", raw_text)
    return match.group(0) if match else "Unknown"

def extract_email(raw_text):
    match = re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", raw_text)
    return match.group(0) if match else "Not found"

def extract_phone(raw_text):
    match = re.search(r"\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}", raw_text)
    return match.group(0) if match else "Not found"

# === EXTRACTION DES COMPÉTENCES (CORRIGÉE) ===
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
    result = call_groq_api(prompt, raw_text[:2000])
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

    return sorted(list(skills))  # ← LISTE GARANTIE

# === EXTRACTION DE L'ÉDUCATION (CORRIGÉE) ===
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

# === APPEL API GROQ ===
def call_groq_api(prompt, text):
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

# === GÉNÉRATION DE RECOMMANDATIONS ===
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
    return call_groq_api(prompt, "")

# === ROUTES FLASK ===
@app.route('/analyse', methods=['GET', 'POST'])
def analyse():
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

            # === TRAITEMENT ===
            raw_text = extract_text_from_file(filepath)
            if "Error" in raw_text:
                flash(raw_text)
                os.remove(filepath)
                return redirect(request.url)

            name = extract_name(raw_text)
            email = extract_email(raw_text)
            phone = extract_phone(raw_text)

            # GARANTIR DES LISTES
            skills_list = extract_skills_automatically(raw_text)
            education_list = extract_education_from_resume(raw_text)

            skills = ', '.join(skills_list) if skills_list else "No skills detected"
            education = ', '.join(education_list) if education_list else "No education detected"

            recommendations = generate_recommendations(name, skills, education)

            # Nettoyer
            if os.path.exists(filepath):
                os.remove(filepath)

            # Passer uniquement les recommandations au template
            return render_template('result.html', recommendations=recommendations)

    return render_template('analyse.html')

if __name__ == '__main__':
    app.run(debug=True)