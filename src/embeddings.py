"""Etapa 5b: gera embeddings com intfloat/multilingual-e5.

O e5 foi treinado com prefixos: "query: " para a pergunta/consulta e
"passage: " para os textos onde se busca. Aqui os textos de referência são
as queries e os chunks dos currículos são as passages.

Os vetores são normalizados (norma 1), então o produto escalar já é o cosseno.
Tudo é salvo em disco; se nada mudou, não recalcula.
"""

import json

import numpy as np
import pandas as pd

import config
from src.chunking import dividir_por_tamanho
from src.referencia import listar_referencias

_modelo = None


def escolher_dispositivo():
    import torch

    if config.DISPOSITIVO != "auto":
        dispositivo = config.DISPOSITIVO
    elif torch.cuda.is_available():
        dispositivo = "cuda"
    else:
        dispositivo = "cpu"
    return dispositivo


def carregar_modelo():
    global _modelo
    if _modelo is None:
        from sentence_transformers import SentenceTransformer
        dispositivo = escolher_dispositivo()
        print(f"Carregando modelo {config.MODELO_EMBEDDING} em {dispositivo}...")
        _modelo = SentenceTransformer(config.MODELO_EMBEDDING, device=dispositivo)
        if dispositivo == "cuda":
            # meia precisão: metade da memória e bem mais rápido na GPU
            _modelo.half()
    return _modelo


def codificar(textos, prefixo):
    modelo = carregar_modelo()
    com_prefixo = []
    for texto in textos:
        com_prefixo.append(prefixo + texto)

    vetores = modelo.encode(
        com_prefixo,
        batch_size=config.BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    resultado = vetores.astype(np.float32)
    return resultado


def info_atual(quantidade):
    info = {
        "modelo": config.MODELO_EMBEDDING,
        "chunk_tamanho": config.CHUNK_TAMANHO,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "secoes_ignoradas": config.SECOES_IGNORADAS,
        "quantidade": quantidade,
    }
    return info


def cache_valido(info):
    valido = False
    if config.EMBEDDINGS_NPY.exists() and config.EMBEDDINGS_INFO.exists():
        salvo = json.loads(config.EMBEDDINGS_INFO.read_text(encoding="utf-8"))
        if salvo == info:
            valido = True
    return valido


def gerar_embeddings_chunks():
    tabela = pd.read_parquet(config.CHUNKS_PARQUET)
    info = info_atual(len(tabela))

    if cache_valido(info):
        print(f"Embeddings dos chunks já existem ({len(tabela)}); usando o cache.")
        vetores = np.load(config.EMBEDDINGS_NPY)
    else:
        textos = []
        for secao, texto in zip(tabela["secao"], tabela["texto"]):
            textos.append(secao + "\n" + texto)
        print(f"Gerando embeddings de {len(textos)} chunks...")
        vetores = codificar(textos, config.PREFIXO_PASSAGE)
        np.save(config.EMBEDDINGS_NPY, vetores)
        config.EMBEDDINGS_INFO.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Salvo em {config.EMBEDDINGS_NPY} ({vetores.shape})")

    return vetores


def gerar_embedding_referencia(nome, texto):
    """Embedding de um texto de referência (com cache).

    Se o texto for maior que um chunk, ele é dividido e o resultado é a média
    dos pedaços, normalizada de novo.
    """
    config.REFERENCIAS_EMB_DIR.mkdir(parents=True, exist_ok=True)
    caminho_npy = config.REFERENCIAS_EMB_DIR / f"{nome}.npy"
    caminho_info = config.REFERENCIAS_EMB_DIR / f"{nome}.json"
    info = {"modelo": config.MODELO_EMBEDDING, "prefixo": config.PREFIXO_QUERY, "texto": texto}

    usar_cache = False
    if caminho_npy.exists() and caminho_info.exists():
        salvo = json.loads(caminho_info.read_text(encoding="utf-8"))
        if salvo == info:
            usar_cache = True

    if usar_cache:
        resultado = np.load(caminho_npy)
    else:
        pedacos = dividir_por_tamanho(texto, config.CHUNK_TAMANHO, config.CHUNK_OVERLAP)
        vetores = codificar(pedacos, config.PREFIXO_QUERY)
        media = vetores.mean(axis=0)
        norma = np.linalg.norm(media)
        if norma > 0:
            media = media / norma
        resultado = media.astype(np.float32)
        np.save(caminho_npy, resultado)
        caminho_info.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Embedding da referência '{nome}' salvo ({len(pedacos)} pedaço(s))")

    return resultado


def gerar_embeddings_referencias():
    """nome da referência -> vetor."""
    vetores = {}
    for nome, texto in listar_referencias():
        vetores[nome] = gerar_embedding_referencia(nome, texto)
    return vetores


def etapa_embeddings():
    config.EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    gerar_embeddings_chunks()
    referencias = gerar_embeddings_referencias()
    print(f"{len(referencias)} textos de referência com embedding.")
