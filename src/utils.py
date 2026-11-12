"""Funções pequenas usadas por mais de uma etapa."""

import unicodedata
from datetime import datetime


def nome_arquivo(nome):
    """'Ana Cláudia Rossi' -> 'ana_claudia_rossi' (seguro para nome de arquivo)."""
    sem_acento = unicodedata.normalize("NFKD", nome)
    sem_acento = sem_acento.encode("ascii", "ignore").decode("ascii")

    caracteres = []
    for caractere in sem_acento.lower():
        if caractere.isalnum():
            caracteres.append(caractere)
        else:
            caracteres.append("_")

    resultado = "".join(caracteres)
    while "__" in resultado:
        resultado = resultado.replace("__", "_")
    resultado = resultado.strip("_")
    return resultado


def registrar_log(caminho, mensagem):
    """Acrescenta uma linha com data/hora no arquivo de log e mostra no terminal."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    agora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    linha = f"{agora} | {mensagem}"
    with open(caminho, "a", encoding="utf-8") as arquivo:
        arquivo.write(linha + "\n")
    print(f"  [log] {mensagem}")
