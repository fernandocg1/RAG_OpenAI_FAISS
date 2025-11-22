import os
from dotenv import load_dotenv

# Importações para o carregamento do arquivo
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Importações para a vetorização (FAISS e OpenAI)
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Carrega a chave da API da OpenAI do arquivo .env
load_dotenv(dotenv_path="../.env")

# --- Configurações ---
PASTA_BASE = "base/estruturas_condicionais.md"
PASTA_DB = "faiss_md_index" 

def criar_db():
    print("--- 1. Carregando documentos...")
    documentos = carregar_documentos()
    
    if not documentos:
        print("ERRO: Nenhum documento carregado. Verifique o caminho e o arquivo .md.")
        return

    print(f"Documentos carregados: {len(documentos)}")
    
    print("--- 2. Dividindo documentos em Chunks...")
    # Chamada corrigida para a função
    chunks = dividir_chunks(documentos) 
    print(f"Chunks criados: {len(chunks)}")
    
    print("--- 3. Criando o Banco de Dados Vetorial (FAISS)...")
    criar_vetor_db(chunks)
    print("Sucesso! Banco de Dados FAISS criado e salvo.")

# --- FUNÇÕES DE PROCESSAMENTO ---

def carregar_documentos():
    """Carrega o arquivo Markdown usando o UnstructuredMarkdownLoader."""
    try:
        # ... (suas verificações de caminho) ...
        # Use o UnstructuredMarkdownLoader
        carregador = UnstructuredMarkdownLoader(PASTA_BASE)
        documentos = carregador.load()
        return documentos
    except Exception as e:
        print(f"Erro ao carregar o arquivo: {e}")
        return []

# Nome da função corrigido para refletir a chamada em criar_db()
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

# Função vetorizar_chunks() renomeada para criar_vetor_db() e implementada
def criar_vetor_db(chunks):
    """Cria embeddings com OpenAI e salva no FAISS."""
    
    # 1. Inicializa o Embedding da OpenAI (modelo eficiente e barato)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # 2. Cria o índice FAISS a partir dos chunks
    db = FAISS.from_documents(chunks, embeddings)
    
    # 3. Salva o índice no disco
    if not os.path.exists(PASTA_DB):
        os.makedirs(PASTA_DB)

    db.save_local(PASTA_DB)


if __name__ == "__main__":
    criar_db()