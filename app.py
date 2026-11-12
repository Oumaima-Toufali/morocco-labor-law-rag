from dotenv import load_dotenv
import os, chromadb, streamlit as st
from openai import OpenAI
from chromadb.utils import embedding_functions
load_dotenv()

@st.cache_resource
def get_collection():
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    return chromadb.PersistentClient("db").get_collection("docs", embedding_function=ef)

col = get_collection()
llm = OpenAI(base_url=os.environ["LLM_BASE_URL"], api_key=os.environ["LLM_API_KEY"])
MODEL = os.environ["LLM_MODEL"]

st.title("⚖️ Moroccan Labor Law Assistant")
st.caption("Answers are based on the Code du travail. This is not legal advice.")
q = st.text_input("Your question")

if q:
    res = col.query(query_texts=[q], n_results=8)
    context = "\n\n".join(
        f"[{m['source']}]\n{d}" for d, m in zip(res["documents"][0], res["metadatas"][0])
    )
    answer = llm.chat.completions.create(
        model=MODEL,
        max_tokens=2000,
        messages=[
            {"role": "system", "content": (
                "You are an assistant for Moroccan labor law. "
                "Answer ONLY from the provided context. "
                "Reply in the same language as the question. "
                "Cite the article numbers when they appear in the context. "
                "If the answer is not in the context, say you don't know. "
                "Do not give legal advice."
            )},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {q}"},
        ],
    )
    st.write(answer.choices[0].message.content)
    with st.expander("Retrieved chunks"):
        st.write(context)