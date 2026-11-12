"""Etapa 3: converte os PDFs do Lattes em Markdown.

Usa o pymupdf4llm (rápido, sem modelos pesados) e depois normaliza os títulos:
só as seções do Lattes viram "## Seção"; os falsos títulos que o conversor cria
(anos, números de itens, nomes de empresas) voltam a ser texto em negrito.
"""

import unicodedata

import pymupdf4llm

import config
from src.utils import registrar_log

# Seções de primeiro nível do currículo Lattes
SECOES_LATTES = [
    "Identificação",
    "Endereço",
    "Formação acadêmica/titulação",
    "Pós-doutorado",
    "Formação Complementar",
    "Atuação Profissional",
    "Linhas de pesquisa",
    "Projetos de pesquisa",
    "Projetos de extensão",
    "Projetos de desenvolvimento",
    "Outros Projetos",
    "Membro de corpo editorial",
    "Revisor de periódico",
    "Revisor de projeto de fomento",
    "Áreas de atuação",
    "Idiomas",
    "Prêmios e títulos",
    "Produções",
    "Patentes e registros",
    "Inovação",
    "Educação e Popularização de C & T",
    "Bancas",
    "Eventos",
    "Orientações",
    "Outras informações relevantes",
]

CONVERSOR = "pymupdf4llm"


def simplificar(texto):
    """Minúsculas, sem acento, sem '*' e sem espaços nas pontas (para comparar títulos)."""
    texto = texto.replace("*", "").strip().lower()
    texto = unicodedata.normalize("NFKD", texto)
    texto = texto.encode("ascii", "ignore").decode("ascii")
    resultado = " ".join(texto.split())
    return resultado


def montar_mapa_secoes():
    mapa = {}
    for secao in SECOES_LATTES:
        mapa[simplificar(secao)] = secao
    return mapa


MAPA_SECOES = montar_mapa_secoes()


def normalizar_titulos(markdown):
    """Deixa como título só o nome do professor (#) e as seções do Lattes (##)."""
    linhas_saida = []
    for linha in markdown.splitlines():
        conteudo = linha.strip()
        nivel = 0
        while nivel < len(conteudo) and conteudo[nivel] == "#":
            nivel = nivel + 1
        texto = conteudo[nivel:].strip()
        chave = simplificar(texto)

        if chave in MAPA_SECOES:
            linhas_saida.append("## " + MAPA_SECOES[chave])
        elif nivel == 1:
            linhas_saida.append("# " + texto.replace("*", "").strip())
        elif nivel > 1:
            linhas_saida.append("**" + texto.replace("*", "").strip() + "**")
        elif chave.startswith("pagina gerada pelo sistema curriculo lattes"):
            linhas_saida.append("")
        else:
            linhas_saida.append(linha.rstrip())

    resultado = "\n".join(linhas_saida).strip() + "\n"
    return resultado


def pdf_para_markdown(caminho_pdf):
    bruto = pymupdf4llm.to_markdown(str(caminho_pdf), show_progress=False)
    resultado = normalizar_titulos(bruto)
    return resultado


def etapa_converter():
    config.MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)

    pendentes = []
    for caminho_pdf in sorted(config.PDF_DIR.glob("*.pdf")):
        caminho_md = config.MARKDOWN_DIR / f"{caminho_pdf.stem}.md"
        if not caminho_md.exists():
            pendentes.append(caminho_pdf)

    print(f"{len(pendentes)} PDFs para converter com {CONVERSOR}.")

    numero = 1
    for caminho_pdf in pendentes:
        caminho_md = config.MARKDOWN_DIR / f"{caminho_pdf.stem}.md"
        try:
            markdown = pdf_para_markdown(caminho_pdf)
            caminho_md.write_text(markdown, encoding="utf-8")
            print(f"  [{numero}/{len(pendentes)}] OK  {caminho_md.name}", flush=True)
        except Exception as erro:
            registrar_log(config.CONVERSAO_LOG, f"ERRO ao converter | {caminho_pdf.name} | {erro}")
        numero = numero + 1

    total = len(list(config.MARKDOWN_DIR.glob("*.md")))
    print(f"{total} arquivos Markdown em {config.MARKDOWN_DIR}")
