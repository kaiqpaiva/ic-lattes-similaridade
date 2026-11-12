"""Configurações centrais do projeto."""

from pathlib import Path

RAIZ = Path(__file__).resolve().parent

# Pastas
DATA_DIR = RAIZ / "data"
BRUTOS_DIR = DATA_DIR / "brutos"          # HTML original do Lattes (coletado à parte)
PDF_DIR = DATA_DIR / "pdfs"               # PDF gerado a partir do HTML (coletado à parte)
MARKDOWN_DIR = DATA_DIR / "markdown"
REFERENCIA_DIR = DATA_DIR / "referencia"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
LOGS_DIR = RAIZ / "logs"
RESULTS_DIR = RAIZ / "results"

# Arquivos
PROFESSORES_CSV = DATA_DIR / "professores.csv"
ESPERADOS_CSV = REFERENCIA_DIR / "esperados.csv"   # professores esperados por referência (validação)
CONVERSAO_LOG = LOGS_DIR / "conversao.log"
CHUNKS_PARQUET = EMBEDDINGS_DIR / "chunks.parquet"
EMBEDDINGS_NPY = EMBEDDINGS_DIR / "embeddings_chunks.npy"
EMBEDDINGS_INFO = EMBEDDINGS_DIR / "embeddings_info.json"
REFERENCIAS_EMB_DIR = EMBEDDINGS_DIR / "referencias"   # um .npy por texto de referência
VALIDACAO_MD = RESULTS_DIR / "validacao.md"

# Etapa 1
URL_CORPO_DOCENTE = (
    "https://www.mackenzie.br/graduacao/sao-paulo-higienopolis/"
    "ciencia-da-computacao/corpo-docente"
)

# Etapa 5 - chunking
CHUNK_TAMANHO = 1200   # em caracteres (~300 tokens; o e5 aceita até 512)
CHUNK_OVERLAP = 200    # em caracteres
# Seções ignoradas no chunking:
# - Identificação, Endereço, Idiomas: só dados pessoais/contato
# - Bancas: lista de defesas de OUTROS alunos de que o professor participou (17% dos chunks)
# - Eventos: lista de congressos em que participou (5% dos chunks)
# Tirar Bancas e Eventos reduz o tempo dos embeddings e o ruído sobre o tema de pesquisa.
SECOES_IGNORADAS = ["Identificação", "Endereço", "Idiomas", "Bancas", "Eventos"]

# Etapa 5 - embeddings
MODELO_EMBEDDING = "intfloat/multilingual-e5-base"
PREFIXO_QUERY = "query: "       # usado nos textos de referência
PREFIXO_PASSAGE = "passage: "   # usado nos chunks dos currículos
BATCH_SIZE = 32
# "auto" usa a GPU (CUDA) se houver, em meia precisão (fp16); senão usa a CPU.
DISPOSITIVO = "auto"

# Etapa 6
TOP_K = 5                       # quantos chunks entram na "média do top-k"
TOP_PROFESSORES_DETALHE = 10    # quantos professores aparecem em top_chunks.md
CHUNKS_POR_PROFESSOR = 3        # quantos trechos mostrar de cada um
