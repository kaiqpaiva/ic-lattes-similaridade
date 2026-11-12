"""Etapa 6: compara cada texto de referência com os chunks de cada professor.

Métricas por professor (todas são similaridade de cosseno):
- sim_topk_media:      média dos TOP_K chunks mais parecidos (métrica principal do ranking)
- sim_maxima:          o chunk mais parecido
- sim_media:           média de todos os chunks (penaliza currículos longos e variados)
- sim_curriculo_medio: cosseno com o "embedding médio" do currículo inteiro

Saídas:
- results/<referencia>/ranking.csv e results/<referencia>/top_chunks.md
- results/matriz_referencias.csv (professor x referência, sim_topk_media)
- results/validacao.md (os professores esperados apareceram no topo?)
"""

import numpy as np
import pandas as pd

import config
from src.embeddings import gerar_embeddings_referencias
from src.referencia import ler_esperados, listar_referencias

TOP_VALIDACAO = 5   # um professor esperado "acertou" se ficou entre os 5 primeiros


def carregar_dados():
    tabela = pd.read_parquet(config.CHUNKS_PARQUET)
    vetores = np.load(config.EMBEDDINGS_NPY)
    if len(tabela) != len(vetores):
        raise ValueError("chunks.parquet e embeddings_chunks.npy estão fora de sincronia; rode a etapa embeddings.")
    referencias = gerar_embeddings_referencias()
    return tabela, vetores, referencias


def metricas_professor(vetores_prof, sims_prof, referencia, top_k):
    ordenadas = np.sort(sims_prof)[::-1]
    k = min(top_k, len(ordenadas))

    media_curriculo = vetores_prof.mean(axis=0)
    norma = np.linalg.norm(media_curriculo)
    if norma > 0:
        media_curriculo = media_curriculo / norma

    metricas = {
        "sim_topk_media": float(ordenadas[:k].mean()),
        "sim_maxima": float(ordenadas[0]),
        "sim_media": float(sims_prof.mean()),
        "sim_curriculo_medio": float(media_curriculo @ referencia),
    }
    return metricas


def calcular_ranking(tabela, vetores, referencia):
    sims = vetores @ referencia
    tabela = tabela.copy()
    tabela["similaridade"] = sims

    linhas = []
    for arquivo, grupo in tabela.groupby("arquivo"):
        indices = grupo.index.to_numpy()
        metricas = metricas_professor(vetores[indices], sims[indices], referencia, config.TOP_K)
        linha = {
            "professor": grupo["professor"].iloc[0],
            "arquivo": arquivo,
            "n_chunks": len(grupo),
        }
        linha.update(metricas)
        linhas.append(linha)

    ranking = pd.DataFrame(linhas)
    ranking = ranking.sort_values("sim_topk_media", ascending=False).reset_index(drop=True)
    ranking.insert(0, "posicao", range(1, len(ranking) + 1))
    return ranking, tabela


def escrever_top_chunks(nome_ref, texto_ref, ranking, tabela_com_sims, caminho):
    linhas = []
    linhas.append(f"# Trechos que mais bateram com: {nome_ref}\n")
    linhas.append(f"> {texto_ref}\n")
    linhas.append(f"Modelo: `{config.MODELO_EMBEDDING}` | top-k: {config.TOP_K} | "
                  f"chunk: {config.CHUNK_TAMANHO} caracteres, overlap {config.CHUNK_OVERLAP}\n")

    topo = ranking.head(config.TOP_PROFESSORES_DETALHE)
    for _, prof in topo.iterrows():
        linhas.append(f"\n## {prof['posicao']}. {prof['professor']}\n")
        linhas.append(f"top-{config.TOP_K} média: **{prof['sim_topk_media']:.4f}** | "
                      f"máxima: {prof['sim_maxima']:.4f} | média: {prof['sim_media']:.4f} | "
                      f"currículo médio: {prof['sim_curriculo_medio']:.4f} | chunks: {prof['n_chunks']}\n")

        do_prof = tabela_com_sims[tabela_com_sims["arquivo"] == prof["arquivo"]]
        melhores = do_prof.sort_values("similaridade", ascending=False).head(config.CHUNKS_POR_PROFESSOR)
        for _, chunk in melhores.iterrows():
            trecho = chunk["texto"]
            if len(trecho) > 600:
                trecho = trecho[:600] + "..."
            linhas.append(f"\n**{chunk['similaridade']:.4f}** — *{chunk['secao']}*\n")
            linhas.append(f"> {trecho}\n")

    caminho.write_text("\n".join(linhas), encoding="utf-8")


def posicoes_esperados(ranking, professores):
    """Devolve uma lista de (nome, posição); posição None se o professor não está no ranking."""
    resultado = []
    for nome in professores:
        linha = ranking[ranking["professor"] == nome]
        if len(linha) == 0:
            resultado.append((nome, None))
        else:
            resultado.append((nome, int(linha["posicao"].iloc[0])))
    return resultado


def escrever_validacao(rankings, textos, esperados):
    linhas = []
    linhas.append("# Validação do método\n")
    linhas.append("Cada texto de referência foi escrito pensando em professores específicos (positivos) "
                  "ou em temas sem relação com computação (negativos). Se o método funciona, os "
                  f"esperados aparecem entre os {TOP_VALIDACAO} primeiros e os negativos têm notas "
                  "menores que os positivos.\n")
    linhas.append(f"Métrica usada: `sim_topk_media` (média dos {config.TOP_K} chunks mais parecidos).\n")

    linhas.append("\n## Resumo\n")
    linhas.append("| Referência | Tipo | 1º colocado | Nota do 1º | Esperados (posição) | Acertos no top "
                  f"{TOP_VALIDACAO} |")
    linhas.append("|---|---|---|---|---|---|")

    total_esperados = 0
    total_acertos = 0
    maiores_positivos = []
    maiores_negativos = []

    for nome_ref, ranking in rankings.items():
        info = esperados.get(nome_ref, {"tipo": "?", "professores": []})
        primeiro = ranking.iloc[0]
        nota_primeiro = float(primeiro["sim_topk_media"])

        if info["tipo"] == "negativo":
            maiores_negativos.append(nota_primeiro)
        elif info["tipo"] == "positivo":
            maiores_positivos.append(nota_primeiro)

        textos_pos = []
        acertos = 0
        for nome, posicao in posicoes_esperados(ranking, info["professores"]):
            if posicao is None:
                textos_pos.append(f"{nome} (fora)")
            else:
                textos_pos.append(f"{nome} ({posicao}º)")
                if posicao <= TOP_VALIDACAO:
                    acertos = acertos + 1
        total_esperados = total_esperados + len(info["professores"])
        total_acertos = total_acertos + acertos

        coluna_esperados = "—"
        coluna_acertos = "—"
        if len(info["professores"]) > 0:
            coluna_esperados = ", ".join(textos_pos)
            coluna_acertos = f"{acertos}/{len(info['professores'])}"

        linhas.append(f"| {nome_ref} | {info['tipo']} | {primeiro['professor']} | {nota_primeiro:.4f} | "
                      f"{coluna_esperados} | {coluna_acertos} |")

    linhas.append("")
    if total_esperados > 0:
        linhas.append(f"**Acertos:** {total_acertos} de {total_esperados} professores esperados ficaram "
                      f"no top {TOP_VALIDACAO}.\n")
    if len(maiores_positivos) > 0 and len(maiores_negativos) > 0:
        linhas.append(f"**Notas do 1º colocado:** positivos entre {min(maiores_positivos):.4f} e "
                      f"{max(maiores_positivos):.4f}; negativos entre {min(maiores_negativos):.4f} e "
                      f"{max(maiores_negativos):.4f}.\n")
        if max(maiores_negativos) < min(maiores_positivos):
            linhas.append("Os negativos ficaram abaixo de todos os positivos, como esperado.\n")
        else:
            linhas.append("Atenção: algum negativo teve nota maior que algum positivo.\n")

    linhas.append(f"\n## Top {TOP_VALIDACAO} de cada referência\n")
    for nome_ref, ranking in rankings.items():
        linhas.append(f"\n### {nome_ref}\n")
        linhas.append(f"> {textos[nome_ref]}\n")
        linhas.append("| Posição | Professor | top-k média | máxima | média |")
        linhas.append("|---|---|---|---|---|")
        for _, prof in ranking.head(TOP_VALIDACAO).iterrows():
            linhas.append(f"| {prof['posicao']} | {prof['professor']} | {prof['sim_topk_media']:.4f} | "
                          f"{prof['sim_maxima']:.4f} | {prof['sim_media']:.4f} |")
        linhas.append("")

    config.VALIDACAO_MD.write_text("\n".join(linhas), encoding="utf-8")


def etapa_similaridade():
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tabela, vetores, vetores_ref = carregar_dados()

    textos = {}
    for nome, texto in listar_referencias():
        textos[nome] = texto

    rankings = {}
    matriz = None
    for nome_ref, referencia in vetores_ref.items():
        ranking, tabela_com_sims = calcular_ranking(tabela, vetores, referencia)
        rankings[nome_ref] = ranking

        pasta = config.RESULTS_DIR / nome_ref
        pasta.mkdir(parents=True, exist_ok=True)
        ranking.to_csv(pasta / "ranking.csv", index=False, float_format="%.4f", encoding="utf-8")
        escrever_top_chunks(nome_ref, textos[nome_ref], ranking, tabela_com_sims, pasta / "top_chunks.md")

        coluna = ranking[["professor", "sim_topk_media"]].rename(columns={"sim_topk_media": nome_ref})
        if matriz is None:
            matriz = coluna
        else:
            matriz = matriz.merge(coluna, on="professor")

        print(f"\n== {nome_ref} ==")
        print(ranking.head(5)[["posicao", "professor", "sim_topk_media", "sim_maxima"]]
              .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    matriz = matriz.sort_values("professor")
    matriz.to_csv(config.RESULTS_DIR / "matriz_referencias.csv", index=False, float_format="%.4f", encoding="utf-8")
    escrever_validacao(rankings, textos, ler_esperados())

    print(f"\nRankings em {config.RESULTS_DIR}/<referencia>/ranking.csv")
    print(f"Relatório de validação em {config.VALIDACAO_MD}")
    return rankings
