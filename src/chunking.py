"""Etapa 5a: divide os markdowns dos currículos em chunks.

Primeiro separa por seção do Lattes (títulos "## "). Seções curtas viram um
chunk só; seções longas são cortadas por tamanho, com overlap.
"""

import csv

import pandas as pd

import config
from src.utils import nome_arquivo


def dividir_em_secoes(markdown):
    """Devolve uma lista de (nome_da_secao, texto). O que vem antes da 1ª seção é o 'Resumo'."""
    secoes = []
    secao_atual = "Resumo"
    linhas_atuais = []

    for linha in markdown.splitlines():
        if linha.startswith("## "):
            texto = "\n".join(linhas_atuais).strip()
            if texto != "":
                secoes.append((secao_atual, texto))
            secao_atual = linha[3:].strip()
            linhas_atuais = []
        elif linha.startswith("# "):
            # nome do professor: não entra no texto
            continue
        else:
            linhas_atuais.append(linha)

    texto = "\n".join(linhas_atuais).strip()
    if texto != "":
        secoes.append((secao_atual, texto))

    return secoes


def dividir_por_tamanho(texto, tamanho, overlap):
    """Corta o texto em janelas de até `tamanho` caracteres, repetindo `overlap` caracteres.

    Sempre que possível o corte acontece num espaço/quebra de linha, para não partir palavras.
    """
    pedacos = []
    texto = " ".join(texto.split())

    if len(texto) <= tamanho:
        pedacos.append(texto)
    else:
        inicio = 0
        while inicio < len(texto):
            fim = inicio + tamanho
            if fim < len(texto):
                espaco = texto.rfind(" ", inicio + tamanho // 2, fim)
                if espaco != -1:
                    fim = espaco
            else:
                fim = len(texto)

            pedaco = texto[inicio:fim].strip()
            if pedaco != "":
                pedacos.append(pedaco)

            if fim >= len(texto):
                break

            proximo = fim - overlap
            espaco = texto.find(" ", proximo, fim)
            if espaco != -1:
                proximo = espaco + 1
            if proximo <= inicio:
                proximo = fim
            inicio = proximo

    return pedacos


def chunkar_markdown(markdown, tamanho, overlap, secoes_ignoradas):
    """Devolve uma lista de dicts {secao, texto} para um currículo."""
    chunks = []
    for secao, texto in dividir_em_secoes(markdown):
        if secao in secoes_ignoradas:
            continue
        texto = texto.replace("**", "")  # negrito do Markdown é só ruído para o modelo
        for pedaco in dividir_por_tamanho(texto, tamanho, overlap):
            chunks.append({"secao": secao, "texto": pedaco})
    return chunks


def mapa_nomes():
    """arquivo (ex.: 'ana_claudia_rossi') -> nome como está no CSV.

    O CSV é opcional (fica fora do git). Sem ele, o mapa vem vazio e o nome
    sai do próprio markdown (ver nome_do_markdown).
    """
    mapa = {}
    if config.PROFESSORES_CSV.exists():
        with open(config.PROFESSORES_CSV, newline="", encoding="utf-8") as arquivo:
            for linha in csv.DictReader(arquivo):
                mapa[nome_arquivo(linha["nome"])] = linha["nome"]
    return mapa


def nome_do_markdown(markdown, padrao):
    """Nome do professor a partir da linha '# Nome' do markdown; se não houver, usa o padrão."""
    nome = padrao
    encontrado = False
    for linha in markdown.splitlines():
        if not encontrado and linha.startswith("# "):
            nome = linha[2:].strip()
            encontrado = True
    return nome


def etapa_chunking():
    config.EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    nomes = mapa_nomes()

    linhas = []
    for caminho_md in sorted(config.MARKDOWN_DIR.glob("*.md")):
        arquivo = caminho_md.stem
        markdown = caminho_md.read_text(encoding="utf-8")
        if arquivo in nomes:
            professor = nomes[arquivo]
        else:
            professor = nome_do_markdown(markdown, arquivo)
        chunks = chunkar_markdown(
            markdown, config.CHUNK_TAMANHO, config.CHUNK_OVERLAP, config.SECOES_IGNORADAS
        )
        numero = 0
        for chunk in chunks:
            linhas.append({
                "professor": professor,
                "arquivo": arquivo,
                "chunk_id": numero,
                "secao": chunk["secao"],
                "texto": chunk["texto"],
            })
            numero = numero + 1

    tabela = pd.DataFrame(linhas)
    tabela.to_parquet(config.CHUNKS_PARQUET, index=False)

    print(f"{len(tabela)} chunks de {tabela['arquivo'].nunique()} professores "
          f"(tamanho={config.CHUNK_TAMANHO}, overlap={config.CHUNK_OVERLAP})")
    print(f"Salvo em {config.CHUNKS_PARQUET}")
    return tabela
