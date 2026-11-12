# IC Lattes Similaridade

Projeto de estudo ligado à minha Iniciação Científica no Mackenzie AI Center
(IAM). Não é a IC em si: serve para praticar e entender, na prática, os
conceitos que ela usa (embeddings, chunking, similaridade de cosseno).

O projeto mede a similaridade semântica entre os currículos Lattes dos
professores de Ciência da Computação (FCI - Mackenzie) e textos de referência, e
gera um ranking por texto com os trechos de cada currículo que mais se
aproximam do tema.

Para validar o método, os textos de referência atuais são de teste: 5
"positivos", escritos a partir do perfil de professores específicos (o método
deveria colocá-los no topo), e 2 "negativos", de outras áreas (agronomia e
culinária), que não deveriam combinar com ninguém. O resultado da validação fica
em `results/validacao.md`.

## Instalação

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Tudo roda localmente e é gratuito. Na primeira execução, o modelo
`intfloat/multilingual-e5-base` (~1 GB) é baixado do Hugging Face.

## Etapas

| Etapa | Comando | Saída |
|---|---|---|
| 1. Professores | feita à parte, fora do código | `data/professores.csv` |
| 2. Currículos | feita à parte (o Lattes pede captcha) | `data/brutos/*.html`, `data/pdfs/*.pdf` |
| 3. PDF → Markdown | `python main.py converter` | `data/markdown/*.md` |
| 4. Textos de referência | um `.md` por texto em `data/referencia/` (`python main.py referencia` lista) | |
| 5. Chunking | `python main.py chunking` | `data/embeddings/chunks.parquet` |
| 5. Embeddings | `python main.py embeddings` | `data/embeddings/*.npy` |
| 6. Similaridade | `python main.py similaridade` | `results/<referencia>/`, `results/validacao.md` |

`python main.py pipeline` roda as etapas 3, 5 e 6 em sequência. Cada etapa
pula o que já foi feito (PDFs já convertidos, embeddings já calculados).

O repositório é público, então nada que identifique os professores fica no git:
nem os currículos, nem a lista de nomes e links (`data/professores.csv`), nem os
professores esperados na validação (`data/referencia/esperados.csv`). Esses
arquivos ficam só na máquina e no `dados_lattes.zip`.

## Rodando numa máquina com GPU

Os embeddings são a parte pesada: numa CPU fraca levam cerca de 1 hora; numa GPU
levam poucos minutos. O código usa a GPU sozinho quando ela existe
(`DISPOSITIVO = "auto"` em `config.py`), em meia precisão (fp16).

Leve o `dados_lattes.zip` (gerado localmente, ~1,5 MB, com `data/markdown/`,
`data/professores.csv` e `data/referencia/esperados.csv`) por Drive ou pendrive e
descompacte na raiz do projeto. Assim não precisa dos PDFs nem dos captchas:

```bash
git clone https://github.com/kaiqpaiva/ic-lattes-similaridade.git
cd ic-lattes-similaridade
unzip dados_lattes.zip
python -m venv .venv
source .venv/bin/activate        # no Windows: .venv\Scripts\activate
```

Instale o torch com CUDA (veja a versão do CUDA com `nvidia-smi`; para CUDA 12.6+
use `cu126`, para 12.8+ pode usar `cu128`) e o resto das dependências:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Se aparecer `True` e o nome da placa, rode:

```bash
python main.py chunking
python main.py embeddings
python main.py similaridade
```

Com uma GPU boa, dá para aumentar `BATCH_SIZE` em `config.py` (ex.: 128) e
testar o modelo maior trocando `MODELO_EMBEDDING` para
`intfloat/multilingual-e5-large`. Os resultados ficam em `results/`, fora do git
(têm nomes, notas e trechos dos currículos); `data/embeddings/` também fica fora
(é cache).

## Decisões

- **Conversão:** pymupdf4llm. Os PDFs têm texto real (não são imagem), então não
  precisa de OCR nem de modelos de layout; é bem mais rápido que o docling.
  Depois da conversão, só as seções oficiais do Lattes viram títulos `##`.
- **Chunking:** por seção do Lattes; seções longas são cortadas por tamanho com
  overlap. Identificação, Endereço, Idiomas, Bancas e Eventos são ignoradas.
  O nome de cada professor vem de `data/professores.csv`; sem esse arquivo, sai
  do título `# Nome` do próprio currículo.
- **Embeddings:** multilingual-e5 com os prefixos `query: ` (textos de referência) e
  `passage: ` (chunks). Vetores normalizados, então produto escalar = cosseno.
- **Ranking:** ordenado pela média dos top-k chunks. Também são calculadas a
  similaridade máxima, a média de todos os chunks e a similaridade com o
  embedding médio do currículo.
- **Validação:** `data/referencia/esperados.csv` diz, para cada texto, se ele é
  positivo ou negativo e quais professores se espera ver no topo.

Os parâmetros (tamanho do chunk, overlap, modelo, top-k) ficam em `config.py`.

Para usar outro texto (por exemplo, a descrição de um projeto), basta criar um
novo `.md` em `data/referencia/` e rodar `python main.py similaridade`.

## Estrutura

- `config.py`: caminhos e parâmetros
- `main.py`: ponto de entrada, uma etapa por comando
- `src/conversao.py`: Etapa 3
- `src/referencia.py`: Etapa 4
- `src/chunking.py`, `src/embeddings.py`: Etapa 5
- `src/similaridade.py`: Etapa 6
- `data/`: professores, currículos, markdown, textos de referência e embeddings
- `results/`: um ranking por texto de referência, a matriz professor x texto e a validação
