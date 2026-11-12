"""Etapa 4: textos de referência.

Cada arquivo data/referencia/<nome>.md é um texto que será comparado com os
currículos (um ranking por texto). O arquivo esperados.csv diz, para cada
referência, se ela é "positivo" (deveria dar nota alta para alguns professores)
ou "negativo" (não deveria combinar com ninguém), e quais professores se espera
ver no topo. Isso serve para validar o método.
"""

import csv

import config


def listar_referencias():
    """Devolve uma lista de (nome, texto), um item por arquivo .md em data/referencia/."""
    referencias = []
    for caminho in sorted(config.REFERENCIA_DIR.glob("*.md")):
        texto = caminho.read_text(encoding="utf-8").strip()
        if texto != "":
            referencias.append((caminho.stem, texto))

    if len(referencias) == 0:
        raise FileNotFoundError(
            f"Nenhum texto de referência em {config.REFERENCIA_DIR}. "
            "Crie pelo menos um arquivo .md com o texto a comparar."
        )

    return referencias


def ler_esperados():
    """referencia -> {"tipo": "positivo"/"negativo", "professores": [nomes]}."""
    esperados = {}
    if config.ESPERADOS_CSV.exists():
        with open(config.ESPERADOS_CSV, newline="", encoding="utf-8") as arquivo:
            for linha in csv.DictReader(arquivo):
                professores = []
                for nome in linha["professores_esperados"].split(";"):
                    nome = nome.strip()
                    if nome != "":
                        professores.append(nome)
                esperados[linha["referencia"]] = {"tipo": linha["tipo"], "professores": professores}
    return esperados


def etapa_referencia():
    referencias = listar_referencias()
    esperados = ler_esperados()
    for nome, texto in referencias:
        tipo = "?"
        if nome in esperados:
            tipo = esperados[nome]["tipo"]
        print(f"- {nome} ({tipo}, {len(texto.split())} palavras): {texto[:90]}...")
    return referencias
