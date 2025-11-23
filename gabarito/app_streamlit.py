import os
from dotenv import load_dotenv
import streamlit as st

# Imports para Gemini LLM e Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# --- 1. CARREGAMENTO DO AMBIENTE E OBTENÇÃO DA CHAVE ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Ajustado o caminho do .env para o que funcionou
DOTENV_PATH = os.path.join(BASE_DIR, '.env') 
load_dotenv(dotenv_path=DOTENV_PATH) 

API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    # Se a chave não for encontrada, o Streamlit exibirá uma mensagem de erro e parará.
    st.error("ERRO: A chave GEMINI_API_KEY não foi encontrada. Verifique seu arquivo .env.")
    st.stop()
# ----------------------------------------------------

CAMINHO_DB = "../faiss_md_index"

prompt_template_str = """
Você é um assistente de IA especializado em responder perguntas sobre estruturas condicionais em Python.

Instrução: Utilize o **Contexto** fornecido para responder à **Pergunta** de forma precisa e concisa. Se o contexto não tiver a informação necessária, responda que não sabe.

--- CONTEXTO ---
{contexto}
---

Pergunta: {pergunta}
Resposta:
"""
prompt = PromptTemplate.from_template(prompt_template_str)

@st.cache_resource
def carregar_componentes_rag():
    """Carrega o LLM, Embeddings e o Banco de Dados FAISS, passando a chave explicitamente."""
    global API_KEY # Garante que o API_KEY seja usado dentro do cache
    
    try:
        # CORREÇÃO 1: Passando a chave explicitamente para o LLM
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0, google_api_key=API_KEY)
        
        # CORREÇÃO 1: Passando a chave explicitamente para os Embeddings
        funcao_embeddings = GoogleGenerativeAIEmbeddings(
            model="text-embedding-004", 
            task_type="RETRIEVAL_DOCUMENT",
            google_api_key=API_KEY
        )
        
        # O FAISS carrega corretamente usando a função de embeddings do Gemini
        db = FAISS.load_local(CAMINHO_DB, funcao_embeddings, allow_dangerous_deserialization=True)
        
        return llm, db
    except Exception as e:
        st.error(f"Erro ao carregar o RAG. Verifique o banco de dados. Erro: {e}")
        st.stop()
        return None, None 

def gerar_resposta(llm, db, pergunta):
    """Executa a busca (Retrieval) e a geração de resposta (Augmentation)."""
    
    resultados_docs = db.similarity_search_with_relevance_scores(pergunta, k=3)
    
    if not resultados_docs:
        return " A busca não retornou documentos relevantes. Tente outra pergunta."
    
    contexto = "\n\n".join([doc.page_content for doc, score in resultados_docs])
    
    prompt_final = prompt.format(contexto=contexto, pergunta=pergunta)
    resposta = llm.invoke(prompt_final)
    
    # CORREÇÃO 2: Acessando o atributo .content antes de .strip()
    return resposta.content.strip()


st.title("🤖 Assistente RAG de Estruturas Condicionais (Python)")
st.caption("Baseado no seu documento Markdown")

llm, db = carregar_componentes_rag()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if pergunta_usuario := st.chat_input("Faça sua pergunta sobre estruturas condicionais..."):
    
    st.session_state.messages.append({"role": "user", "content": pergunta_usuario})
    with st.chat_message("user"):
        st.markdown(pergunta_usuario)

    with st.chat_message("assistant"):
        with st.spinner("Buscando e gerando resposta..."):
            resposta_ia = gerar_resposta(llm, db, pergunta_usuario)
        
        st.markdown(resposta_ia)
    
    st.session_state.messages.append({"role": "assistant", "content": resposta_ia})