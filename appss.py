import os, re, pickle, faiss, torch, requests, unicodedata
import numpy as np, pandas as pd
from flask import Flask, request, jsonify, render_template
from transformers import BertTokenizer, BertModel

# ============================================================
# ⚙️ Configuration Flask
# ============================================================
app = Flask(__name__, template_folder="templates", static_folder="static")

DATA_PATH = "C:\\Users\\Guide Info\\Downloads\\CVs_sans_doublons_Final.xlsx"
INDEX_PATH = "C:\\Users\\Guide Info\\Downloads\\faiss_index_cvs.index"
CHUNKS_PATH = "C:\\Users\\Guide Info\\Downloads\chunk_df.pkl"

OPENAI_API_KEY = os.getenv("GROQ_API_KEY") or "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"
OPENAI_API_BASE = "https://api.groq.com/openai/v1"
MODEL = "llama-3.1-8b-instant"
TOP_K = 5
DEFAULT_TEMPERATURE = 0.25

# ============================================================
# 🧠 Initialisation BERT (modèle original : bert-base-uncased)
# ============================================================
device = torch.device("cpu")
torch.set_grad_enabled(False)

TOKENIZER_ID = "bert-base-uncased"
MODEL_ID = "bert-base-uncased"

print("🔹 Chargement du modèle BERT... (cela peut prendre un peu de temps)")
tokenizer = BertTokenizer.from_pretrained(TOKENIZER_ID)
bert_model = BertModel.from_pretrained(MODEL_ID).to(device)
bert_model.eval()
print("✅ Modèle chargé.")

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

# ============================================================
# 🧩 Fonctions utilitaires
# ============================================================
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

# ============================================================
# 🔍 Chargement FAISS + Chunks
# ============================================================
if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
    with open(CHUNKS_PATH, "rb") as f:
        chunk_df = pickle.load(f)
    chunk_df = chunk_df.dropna(subset=["Nom", "Chunk"])
    faiss_index = faiss.read_index(INDEX_PATH)
    print(f"✅ Index FAISS chargé ({faiss_index.ntotal} vecteurs, dimension {faiss_index.d}).")
else:
    raise RuntimeError("❌ L’index FAISS ou le fichier chunk_df.pkl est introuvable.")

# ============================================================
# 🔎 Recherche contextuelle
# ============================================================
def retrieve_context(query, top_k=TOP_K):
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

# ============================================================
# 🤖 Appel API Groq
# ============================================================
def call_groq_api(prompt, temperature=DEFAULT_TEMPERATURE):
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
    answer = call_groq_api(prompt)
    return _format_bullets(answer), files

# ============================================================
# 🌐 Routes Flask
# ============================================================
@app.route("/chat")
def index():
    return render_template("chat.html")

@app.route("/ask-question", methods=["POST"])
def ask_question():
    try:
        data = request.get_json(force=True, silent=True) or {}
        question = (data.get("question") or "").strip()
        if not question:
            return jsonify({"answer": "Merci d’entrer une question.", "candidats": []})

        print(f"\n🟢 Question reçue : {question}")
        answer, files = hr_llm_chat(question)
        print(f"✅ Réponse générée : {answer[:150]}...")

        return jsonify({"answer": answer, "candidats": files["Nom"].tolist()})
    except Exception as e:
        import traceback
        print("❌ Traceback complet :")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ============================================================
# 🚀 Lancer le serveur
# ============================================================
if __name__ == "__main__":
    app.run(debug=True, port=8000, use_reloader=True)
