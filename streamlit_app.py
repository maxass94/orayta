import streamlit as st
import numpy as np, gzip, json
from collections import defaultdict
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Orayta", page_icon="📜", layout="centered")

@st.cache_resource
def charger():
    versets = json.load(gzip.open("tanakh.json.gz", "rt", encoding="utf-8"))
    emb = np.load("orayta_emb.npy").astype("float32") / 127.0
    modele = SentenceTransformer("intfloat/multilingual-e5-small")
    livres = defaultdict(lambda: defaultdict(list))
    for i, v in enumerate(versets):
        livre, cv = v["ref"].rsplit(" ", 1)
        chap = cv.split(":")[0]
        livres[livre][chap].append((i, v))
    return versets, emb, modele, livres

with st.spinner("Chargement d'Orayta…"):
    versets, emb, modele, livres = charger()

def chercher(question, k=6):
    qe = modele.encode([f"query: {question}"], normalize_embeddings=True).astype("float32")[0]
    ids = np.argsort(-(emb @ qe))[:k]
    return [versets[i] for i in ids]

def cle_gemini():
    try:
        return st.secrets.get("GEMINI_KEY", "")
    except Exception:
        return ""

st.title("אורייתא · Orayta")

onglet_q, onglet_lire = st.tabs(["🔎 Demander à l'IA", "📖 Lire un texte"])

with onglet_q:
    st.caption("Posez une question — l'IA cherche dans les 23 110 versets du Tanakh et cite ses sources. "
               "Outil d'étude : vérifiez toujours dans le texte, ce n'est pas une décision halakhique.")
    q = st.text_input("Votre question de Torah", placeholder="ex : le repos du Chabbat")
    if q:
        sources = chercher(q)
        cle = cle_gemini()
        if cle:
            try:
                import google.generativeai as genai
                genai.configure(api_key=cle)
                contexte = "\n".join([f"[{s['ref']}] {s['en']}" for s in sources])
                prompt = ("Tu es un assistant d'étude de la Torah. Réponds à la question en français, "
                          "UNIQUEMENT à partir des versets fournis, en citant les références entre crochets. "
                          "Si les versets ne suffisent pas, dis-le. Rappelle en une phrase que c'est une aide "
                          "à l'étude, pas une décision halakhique (psak).\n\n"
                          f"Versets :\n{contexte}\n\nQuestion : {q}\n\nRéponse :")
                dispo = [m.name for m in genai.list_models()
                         if "generateContent" in getattr(m, "supported_generation_methods", [])]
                nom = next((x for x in dispo if "flash" in x.lower()), dispo[0] if dispo else "gemini-2.0-flash")
                rep = genai.GenerativeModel(nom).generate_content(prompt).text
                st.markdown("### Réponse")
                st.markdown(rep)
            except Exception as e:
                st.warning(f"Réponse rédigée indisponible pour l'instant ({e}). Voici les versets sources :")
        st.markdown("### Versets sources")
        for rang, s in enumerate(sources, 1):
            st.markdown(f"**{rang}. {s['ref']}**")
            st.markdown(f"<div dir='rtl' style='font-size:1.3em;line-height:1.9'>{s['he']}</div>",
                        unsafe_allow_html=True)
            if s.get("en"):
                st.write(s["en"])
            st.divider()

with onglet_lire:
    st.caption("Choisissez un livre et un chapitre pour lire le texte en hébreu et en traduction.")
    noms = sorted(livres.keys())
    defaut = noms.index("Genesis") if "Genesis" in noms else 0
    livre = st.selectbox("Livre", noms, index=defaut)
    chaps = sorted(livres[livre].keys(), key=lambda x: int(x))
    chap = st.selectbox("Chapitre", chaps)
    st.subheader(f"{livre} {chap}")
    for i, v in livres[livre][chap]:
        num = v["ref"].split(":")[-1]
        st.markdown(f"<div dir='rtl' style='font-size:1.35em;line-height:2'><b>{num}.</b> {v['he']}</div>",
                    unsafe_allow_html=True)
        if v.get("en"):
            st.caption(v["en"])
