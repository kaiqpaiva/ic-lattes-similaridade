"""Ponto de entrada. Uso: python main.py <etapa>

Etapas disponíveis:
    professores   Etapa 1 - coleta nomes e links do Lattes -> data/professores.csv
    converter     Etapa 3 - data/pdfs/*.pdf -> data/markdown/*.md
    referencia    Etapa 4 - lista os textos de referência em data/referencia/*.md
    chunking      Etapa 5a - data/markdown -> data/embeddings/chunks.parquet
    embeddings    Etapa 5b - gera e salva os embeddings (e5)
    similaridade  Etapa 6 - um ranking por referência + results/validacao.md
    pipeline      roda converter, chunking, embeddings e similaridade em sequência

A Etapa 2 (PDFs do Lattes) é feita à parte, porque o Lattes pede captcha.
"""

import sys


def rodar(etapa):
    if etapa == "professores":
        from src.scraping import etapa_professores
        etapa_professores()
    elif etapa == "converter":
        from src.conversao import etapa_converter
        etapa_converter()
    elif etapa == "referencia":
        from src.referencia import etapa_referencia
        etapa_referencia()
    elif etapa == "chunking":
        from src.chunking import etapa_chunking
        etapa_chunking()
    elif etapa == "embeddings":
        from src.embeddings import etapa_embeddings
        etapa_embeddings()
    elif etapa == "similaridade":
        from src.similaridade import etapa_similaridade
        etapa_similaridade()
    elif etapa == "pipeline":
        for parte in ["converter", "chunking", "embeddings", "similaridade"]:
            print(f"\n===== {parte} =====")
            rodar(parte)
    else:
        print(f"Etapa desconhecida: {etapa}")
        print(__doc__)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
    else:
        rodar(sys.argv[1])


if __name__ == "__main__":
    main()
