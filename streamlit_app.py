import streamlit as st
import numpy as np, gzip, json
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Orayta", page_icon="📜", layout="centered")

@st.cache_resource
def charger():
    versets = json.load(gzip.open("tanakh.json.gz", "rt", encoding="utf-8"))
    emb = np.load("orayta_emb.npy").astype("float32") / 127.0
    modele = SentenceTransformer("intfloat/multilingual-e5-small")
    return versets, emb, modele

with st.spinner("Chargement d'Orayta… (quelques secondes)"):
    versets, emb, modele = charger()

st.title("אורייתא · Orayta")
st.caption("Posez une question — l'IA cherche dans les 23 110 versets du Tanakh et cite ses sources. "
           "Outil d'étude : vérifiez toujours dans le texte, ce n'est pas une décision halakhique.")

q = st.text_input("Votre question de Torah", placeholder="ex : le repos du Chabbat")

if q:
    qe = modele.encode([f"query: {q}"], normalize_embeddings=True).astype("float32")[0]
    scores = emb @ qe
    ids = np.argsort(-scores)[:6]
    st.subheader("Versets trouvés")
    for rang, i in enumerate(ids, 1):
        v = versets[i]
        st.markdown(f"**{rang}. {v['ref']}**  ·  pertinence {scores[i]:.2f}")
        st.markdown(
            f"<div dir='rtl' style='font-size:1.35em; line-height:1.9'>{v['he']}</div>",
            unsafe_allow_html=True,
        )
        if v.get("en"):
            st.write(v["en"])
        st.divider()
