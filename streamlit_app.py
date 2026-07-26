import streamlit as st
import numpy as np, gzip, json, html
from collections import defaultdict
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Orayta", page_icon="📜", layout="centered")

# ---------- Style « parchemin & grenat » ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Frank+Ruhl+Libre:wght@500;700&family=Spectral:ital,wght@0,400;0,600;1,400&display=swap');
.stApp { background:#f6efe2; }
#MainMenu, footer, header {visibility:hidden;}
.block-container { max-width: 820px; padding-top: 1.2rem; }
.orayta-head { text-align:center; padding: 14px 0 6px; }
.orayta-head h1 { font-family:'Frank Ruhl Libre',serif; color:#6d2b3a; font-size:2.6rem; margin:0; }
.orayta-head .sub { color:#8a7f6d; font-family:'Spectral',serif; font-style:italic; margin-top:2px; }
.orayta-head .rule { height:3px; width:120px; margin:10px auto 0; background:linear-gradient(90deg,transparent,#b5893a,transparent);}
.stTabs [data-baseweb="tab-list"] { justify-content:center; gap:8px; }
.stTabs [data-baseweb="tab"] { font-family:'Spectral',serif; font-size:1.05rem; }
.stTabs [aria-selected="true"] { color:#6d2b3a !important; }
.verset { background:#fffdf8; border:1px solid #eaded0; border-radius:14px; padding:16px 18px; margin:12px 0;
          box-shadow:0 1px 3px rgba(0,0,0,.05); }
.verset .ref { color:#6d2b3a; font-family:'Spectral',serif; font-weight:600; font-size:1.02rem; }
.verset .badge { display:inline-block; background:#b5893a; color:#fff; border-radius:50%; min-width:26px; height:26px;
                 line-height:26px; text-align:center; font-size:.78rem; font-weight:700; margin-inline-end:8px;}
.verset .he { direction:rtl; text-align:right; font-family:'Frank Ruhl Libre','SBL Hebrew',serif;
              font-size:1.55rem; line-height:2; color:#1c1512; margin:6px 0; }
.verset .tr { color:#4a4438; font-family:'Spectral',serif; font-size:1.02rem; }
.reponse { background:#f0e2df; border-inline-start:4px solid #6d2b3a; border-radius:12px; padding:16px 18px;
           font-family:'Spectral',serif; font-size:1.06rem; color:#2c2620; line-height:1.65; }
.reponse h3 { color:#6d2b3a; margin:0 0 8px; }
.mini { color:#8a7f6d; font-family:'Spectral',serif; font-size:.9rem; text-align:center; margin-bottom:4px;}
</style>
<div class="orayta-head">
  <h1>אורייתא · Orayta</h1>
  <div class="sub">La Torah, cherchée par le sens</div>
  <div class="rule"></div>
</div>
""", unsafe_allow_html=True)

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

def carte_verset(v, num=None):
    he = html.escape(v["he"])
    tr = html.escape(v.get("en", ""))
    badge = f"<span class='badge'>{num}</span>" if num else ""
    return (f"<div class='verset'><div class='ref'>{badge}{html.escape(v['ref'])}</div>"
            f"<div class='he'>{he}</div>" + (f"<div class='tr'>{tr}</div>" if tr else "") + "</div>")

onglet_q, onglet_lire = st.tabs(["🔎  Demander à l'IA", "📖  Lire un texte"])

with onglet_q:
    st.markdown("<div class='mini'>Posez une question — l'IA cherche dans les 23 110 versets du Tanakh et cite ses sources. "
                "Aide à l'étude, pas une décision halakhique.</div>", unsafe_allow_html=True)
    q = st.text_input("Votre question de Torah", placeholder="ex : que dit la Torah sur le repos du Chabbat ?",
                      label_visibility="collapsed")
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
                          "Si les versets ne suffisent pas, dis-le. Termine par une courte phrase rappelant "
                          "que c'est une aide à l'étude, pas une décision halakhique (psak).\n\n"
                          f"Versets :\n{contexte}\n\nQuestion : {q}\n\nRéponse :")
                dispo = [m.name for m in genai.list_models()
                         if "generateContent" in getattr(m, "supported_generation_methods", [])]
                fl = [x for x in dispo if "flash" in x.lower()]
                cand = ([x for x in fl if "latest" in x.lower()] + fl
                        + [x for x in dispo if "pro" in x.lower()] + dispo)
                rep, err = None, None
                for nom in dict.fromkeys(cand):
                    try:
                        rep = genai.GenerativeModel(nom).generate_content(prompt).text
                        break
                    except Exception as ex:
                        err = ex
                if rep is None:
                    raise err or Exception("aucun modèle Gemini disponible")
                st.markdown(f"<div class='reponse'><h3>Réponse</h3>{html.escape(rep).replace(chr(10),'<br>')}</div>",
                            unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"Réponse rédigée indisponible ({e}). Voici les versets sources :")
        st.markdown("<h4 style='color:#6d2b3a;font-family:Spectral,serif'>Versets sources</h4>", unsafe_allow_html=True)
        for rang, s in enumerate(sources, 1):
            st.markdown(carte_verset(s, rang), unsafe_allow_html=True)

with onglet_lire:
    st.markdown("<div class='mini'>Choisissez un livre et un chapitre pour lire le texte.</div>", unsafe_allow_html=True)
    noms = sorted(livres.keys())
    defaut = noms.index("Genesis") if "Genesis" in noms else 0
    c1, c2 = st.columns(2)
    livre = c1.selectbox("Livre", noms, index=defaut)
    chaps = sorted(livres[livre].keys(), key=lambda x: int(x))
    chap = c2.selectbox("Chapitre", chaps)
    st.markdown(f"<h3 style='color:#6d2b3a;font-family:Spectral,serif'>{html.escape(livre)} {chap}</h3>",
                unsafe_allow_html=True)
    for i, v in livres[livre][chap]:
        num = v["ref"].split(":")[-1]
        st.markdown(carte_verset(v, num), unsafe_allow_html=True)
