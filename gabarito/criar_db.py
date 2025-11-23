import os
from dotenv import load_dotenv

# --- 1. CARREGAMENTO E VERIFICAÇÃO DO AMBIENTE (Ordem Correta) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Note que foi mantido o '.env' no mesmo nível do script, como você confirmou
DOTENV_PATH = os.path.join(BASE_DIR, '.env') 
load_dotenv(dotenv_path=DOTENV_PATH) 

# --- VERIFICAÇÃO DO AMBIENTE E OBTENÇÃO DA CHAVE (DEPOIS do load_dotenv) ---
# A API_KEY só existe no ambiente DEPOIS de rodar load_dotenv
API_KEY = os.environ.get("GEMINI_API_KEY") 

chave_presente_g = "GEMINI_API_KEY" in os.environ
chave_presente_a = "GOOGLE_API_KEY" in os.environ
print(f"VERIFICAÇÃO DE CHAVE GEMINI_API_KEY: {chave_presente_g}")
print(f"VERIFICAÇÃO DE CHAVE GOOGLE_API_KEY: {chave_presente_a}")

if not API_KEY:
    raise ValueError("A chave GEMINI_API_KEY não foi encontrada no ambiente.")
# -------------------------------------------------------------------------

# --- 2. Imports e Variáveis ---
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

PASTA_BASE = "base/estruturas_condicionais.md"
PASTA_DB = "faiss_md_index" 

# ... (funções criar_db, carregar_documentos, dividir_chunks) ...
def criar_db():
    print("--- 1. Carregando documentos...")
    documentos = carregar_documentos()
    
    if not documentos:
        print("ERRO: Nenhum documento carregado. Verifique o caminho e o arquivo .md.")
        return

    print(f"Documentos carregados: {len(documentos)}")
    
    print("--- 2. Dividindo documentos em Chunks...")
    chunks = dividir_chunks(documentos) 
    print(f"Chunks criados: {len(chunks)}")
    
    print("--- 3. Criando o Banco de Dados Vetorial (FAISS)...")
    criar_vetor_db(chunks)
    print("Sucesso! Banco de Dados FAISS criado e salvo.")

def carregar_documentos():
    """Carrega o arquivo Markdown usando o UnstructuredMarkdownLoader."""
    try:
        carregador = UnstructuredMarkdownLoader(PASTA_BASE)
        documentos = carregador.load()
        return documentos
    except Exception as e:
        print(f"Erro ao carregar o arquivo: {e}")
        return []

def dividir_chunks(documentos):
    """Divide os documentos carregados em pedaços menores (chunks)."""
    separador_documentos = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=250,
        length_function=len,
        add_start_index=True,
    )
    chunks = separador_documentos.split_documents(documentos)
    return chunks

def criar_vetor_db(chunks):
    """Cria embeddings com Gemini e salva no FAISS."""
    
    # --- CORREÇÃO FINAL: PASSANDO A CHAVE EXPLICITAMENTE ---
    embeddings = GoogleGenerativeAIEmbeddings(
        model="text-embedding-004",
        task_type="RETRIEVAL_DOCUMENT",
        # Parâmetro CRÍTICO: Usa a chave carregada acima para autenticar
        google_api_key=API_KEY 
    )
    # --------------------------------------------------------
    
    db = FAISS.from_documents(chunks, embeddings)
    
    if not os.path.exists(PASTA_DB):
        os.makedirs(PASTA_DB)

    db.save_local(PASTA_DB)


if __name__ == "__main__":
    criar_db()