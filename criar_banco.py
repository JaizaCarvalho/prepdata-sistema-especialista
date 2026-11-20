"""
=====================================================================
criar_banco.py   (arquivo de Edmar)
=====================================================================

PARA QUE SERVE ESTE ARQUIVO?
Ele FABRICA os dados do projeto. Quando você roda
    python criar_banco.py
ele faz 4 coisas, nesta ordem:
    1. Inventa 120 "datasets de mentirinha" (dados SIMULADOS).
    2. Dá a cada um uma nota: baixo, medio ou alto "preparo necessario".
    3. Lê as regras do sistema especialista (arquivo rules.csv).
    4. Guarda tudo em dois lugares: dataset.csv e prepdata.db.

IMPORTANTE (tira a dúvida de "memória"):
Este script NÃO tem memória. Toda vez que roda, ele REFAZ tudo do zero
e SOBRESCREVE os arquivos antigos. Nada fica acumulado de uma execução
para outra. Se algo estiver errado, você corrige AQUI (ou no rules.csv)
e roda de novo: os arquivos novos substituem os velhos.

GLOSSÁRIO (termos que aparecem no código):
- Dataset ........ uma tabela de dados, tipo uma planilha do Excel.
- Registro ....... uma LINHA da tabela (aqui: um dataset de mentirinha).
- Atributo ....... uma COLUNA que descreve o registro (ex.: pct_nulos).
                   É o que o modelo "enxerga" para fazer a previsão.
- Rótulo ......... a resposta certa de cada linha (aqui: baixo/medio/alto).
                   É o que o modelo tenta aprender a prever.
- DataFrame ...... o nome que a biblioteca pandas dá a uma tabela na
                   memória do Python.
- CSV ............ arquivo de texto simples que guarda uma tabela
                   (valores separados por vírgula). Abre no Excel.
- SQLite ......... um "mini banco de dados" que vive dentro de UM arquivo
                   (prepdata.db). Não precisa instalar nada.
- Tabela (SQL) ... o equivalente a uma "aba" dentro do banco.
- NULL ........... "valor vazio". Nas regras, NULL significa "não importa".
- Seed (semente) . número que fixa o sorteio. Mesma seed = mesmos
                   números "aleatórios" sempre. Assim o grupo inteiro
                   gera o MESMO dataset.
- Simulado ....... inventado pelo computador. NÃO são dados reais.
"""

# ---------------------------------------------------------------------
# BLOCO 0: IMPORTAÇÕES
# "import" traz ferramentas prontas para usarmos neste arquivo.
# ---------------------------------------------------------------------
import sqlite3   # conversa com o banco SQLite (já vem com o Python)
import sys       # permite encerrar o programa com uma mensagem de erro

import numpy as np    # contas e sorteios com números ("np" é um apelido)
import pandas as pd   # trabalha com tabelas (DataFrame) ("pd" é o apelido)

# ---------------------------------------------------------------------
# BLOCO 1: CONFIGURAÇÕES
# Valores que você pode querer mudar ficam todos aqui no começo.
# ---------------------------------------------------------------------
ARQ_REGRAS = 'regras.csv'   # nome do arquivo com as regras do grupo.
                           # Se o grupo renomear para regras.csv, troque aqui.

N = 120                    # quantos registros (linhas) vamos criar.
                           # Se alguma classe ficar com menos de 10 linhas,
                           # aumente para 200 e rode de novo.

NIVEIS = ['baixo', 'medio', 'alto']   # as 3 classes possíveis, da menor
                                      # para a maior necessidade de preparo.
                                      # A posição importa: baixo=0, medio=1, alto=2.

# ---------------------------------------------------------------------
# BLOCO 2: GERADOR DE NÚMEROS ALEATÓRIOS (com seed)
# "rng" é a nossa "máquina de sortear". O 42 é a seed (semente): com a
# mesma seed, o sorteio sai igual em qualquer computador.
# ---------------------------------------------------------------------
rng = np.random.default_rng(42)

# ---------------------------------------------------------------------
# BLOCO 3: CRIAR OS ATRIBUTOS (as 4 colunas que descrevem cada dataset)
# Criamos um DataFrame (tabela) com N linhas e 4 colunas.
# ---------------------------------------------------------------------
df = pd.DataFrame({

    # n_linhas = quantas linhas o dataset inventado tem (de 50 a 500 mil).
    # Sorteamos em escala LOGARÍTMICA (np.log / np.exp) para sair muito
    # mais dataset pequeno do que gigante, como na vida real.
    # ".astype(int)" arredonda para número inteiro.
    'n_linhas': np.exp(rng.uniform(np.log(50), np.log(500000), N)).astype(int),

    # pct_nulos = percentual de valores faltando (nulos), de 0 a 60%.
    # "beta" é um tipo de sorteio que gera MUITOS valores baixos e poucos
    # altos (a maioria dos datasets tem poucos nulos).
    # np.round(..., 1) deixa com 1 casa decimal.
    'pct_nulos': np.round(rng.beta(1.2, 5, N) * 60, 1),

    # n_colunas_categoricas = quantas colunas de texto/categoria (sexo,
    # cidade...) o dataset tem. Sorteia um inteiro de 0 a 14.
    'n_colunas_categoricas': rng.integers(0, 15, N),

    # desbalanceamento_pct = o quanto a classe mais comum passa de 50%.
    # 0 = perfeitamente balanceado. Sorteia de 0 a 48, todos igualmente
    # prováveis ("uniform").
    'desbalanceamento_pct': np.round(rng.uniform(0, 48, N), 1),
})


# ---------------------------------------------------------------------
# BLOCO 4: CRITÉRIO DE ROTULAGEM (como damos a "resposta certa")
# Imagine um especialista fictício que dá PONTOS PARA CADA PROBLEMA do
# dataset. Mais problemas = mais pontos = mais preparo necessário.
# Este critério é o que vai no README (a professora quer ver).
# ---------------------------------------------------------------------
def pontuar(l):
    """Recebe UMA linha (l) e devolve a pontuação dela, de 0 a 6."""

    # NULOS: menos de 5% = 0 ponto | de 5% a 20% = 1 ponto | acima de 20% = 2.
    # Leitura do "if" numa linha só:  A if condição else B
    #   = "A, se a condição for verdadeira; senão, B".
    p = 0 if l.pct_nulos < 5 else (1 if l.pct_nulos <= 20 else 2)

    # CATEGÓRICAS: mais de 8 colunas categóricas = +1 ponto (senão +0).
    # "p += 1" significa "some 1 ao p".
    p += 1 if l.n_colunas_categoricas > 8 else 0

    # DESBALANCEAMENTO: menos de 15 = 0 | de 15 a 30 = +1 | acima de 30 = +2.
    p += 0 if l.desbalanceamento_pct < 15 else (1 if l.desbalanceamento_pct <= 30 else 2)

    # POUCAS LINHAS: menos de 500 linhas = +1 ponto.
    p += 1 if l.n_linhas < 500 else 0

    return p


def rotular(p):
    """Transforma a pontuação (p) em classe: 0-1 baixo, 2-3 medio, 4+ alto."""
    return NIVEIS[0] if p <= 1 else (NIVEIS[1] if p <= 3 else NIVEIS[2])


# Aplica o critério em TODAS as linhas e cria a coluna do rótulo.
# df.itertuples() entrega as linhas uma por uma; "for l in ..." repete
# para cada uma. O resultado (uma lista de 120 rótulos) vira a coluna nova.
df['preparo_necessario'] = [rotular(pontuar(l)) for l in df.itertuples()]

# ---------------------------------------------------------------------
# BLOCO 5: RUÍDO (casos de fronteira)
# Se TODOS os rótulos seguissem a fórmula à risca, o modelo acertaria
# quase tudo e isso não seria realista. Então trocamos o rótulo de ~8% das
# linhas para a classe VIZINHA (baixo<->medio ou medio<->alto),
# simulando especialistas que discordam em casos duvidosos.
# ---------------------------------------------------------------------
# rng.choice(...) sorteia QUAIS linhas (posições) vão mudar: 8% de N,
# "replace=False" garante que não sorteia a mesma linha duas vezes.
for i in rng.choice(N, size=int(N * 0.08), replace=False):
    # posição atual da classe (0, 1 ou 2) + (-1 ou +1) = classe vizinha
    k = NIVEIS.index(df.loc[i, 'preparo_necessario']) + int(rng.choice([-1, 1]))
    # min/max seguram o k entre 0 e 2 para não sair da lista de classes
    df.loc[i, 'preparo_necessario'] = NIVEIS[min(max(k, 0), 2)]

# ---------------------------------------------------------------------
# BLOCO 5.5: LIMPEZA OPCIONAL (para "cortar o que está errado")
# Se você quiser REMOVER linhas que não quer no dataset, faça AQUI, antes
# de salvar. Exemplos (tire o "#" do começo da linha para ativar):
#
#   df = df[df['pct_nulos'] <= 50]      # fica só com linhas com até 50% de nulos
#   df = df[df['n_linhas'] >= 100]      # descarta datasets com menos de 100 linhas
#
# Se remover linhas, ANOTE O MOTIVO no README (ex.: "valores irreais").
# NÃO remova linhas só porque o modelo errou nelas: isso é "maquiar" a
# acurácia e é considerado errado.
# ---------------------------------------------------------------------

# Mostra quantas linhas ficaram em cada classe (para você conferir).
# Se alguma tiver menos de 10, aumente o N lá no BLOCO 1.
print('Registros por classe:')
print(df['preparo_necessario'].value_counts())

# ---------------------------------------------------------------------
# BLOCO 6: LER AS REGRAS (antes de gravar qualquer coisa)
# Lemos as regras primeiro de propósito: se o arquivo estiver vazio ou
# faltando, o programa para com uma mensagem clara e NÃO deixa um banco
# pela metade.
# ---------------------------------------------------------------------
try:
    regras = pd.read_csv(ARQ_REGRAS)   # lê o CSV das regras como tabela
except FileNotFoundError:              # arquivo não existe
    sys.exit(f'ERRO: não achei {ARQ_REGRAS}. Rode de dentro da pasta do '
             'projeto e confira se deu git pull.')
except pd.errors.EmptyDataError:       # arquivo existe, mas está vazio
    sys.exit(f'ERRO: {ARQ_REGRAS} está vazio. Peça para a pessoa B subir '
             'as regras e dê git pull.')

# As colunas precisa_explicar e poucos_dados têm células vazias (= "não
# importa"). O tipo 'Int64' guarda números inteiros E aceita vazio sem
# transformar o 1 em 1.0. No banco, vazio vira NULL.
for c in ['precisa_explicar', 'poucos_dados']:
    regras[c] = regras[c].astype('Int64')

# ---------------------------------------------------------------------
# BLOCO 7: SALVAR (dataset.csv e prepdata.db)
# ---------------------------------------------------------------------
# 7a) CSV: arquivo de texto que vai na entrega. "index=False" evita
#     gravar uma coluna extra com o número da linha.
df.to_csv('dataset.csv', index=False)

# 7b) Banco SQLite: "connect" abre o arquivo prepdata.db (cria se não existir).
con = sqlite3.connect('prepdata.db')

# Cria a tabela 'dataset' com o conteúdo do df. "replace" = se já existia,
# apaga a antiga e põe a nova (por isso nada se acumula entre execuções).
df.to_sql('dataset', con, if_exists='replace', index=False)

# Cria a tabela 'regras' com SQL (a linguagem dos bancos de dados).
#  - DROP TABLE IF EXISTS: apaga a tabela velha, se existir.
#  - CREATE TABLE: define as colunas e o TIPO de cada uma.
#  - PRIMARY KEY: o id não pode repetir (R1, R2...).
#  - NOT NULL: recomendação e fonte são OBRIGATÓRIAS em toda regra.
con.executescript('''
DROP TABLE IF EXISTS regras;
CREATE TABLE regras (
    id TEXT PRIMARY KEY,
    classe TEXT,
    precisa_explicar INTEGER,
    poucos_dados INTEGER,
    recomendacao TEXT NOT NULL,
    fonte TEXT NOT NULL
);
''')

# Coloca as linhas do rules.csv dentro da tabela 'regras'.
# "append" = acrescenta nas linhas da tabela recém-criada.
regras.to_sql('regras', con, if_exists='append', index=False)

con.commit()   # confirma: só aqui as mudanças ficam gravadas de verdade
con.close()    # fecha o arquivo do banco (sempre feche depois de usar)

print(f'Pronto: dataset.csv e prepdata.db criados '
      f'({len(df)} registros, {len(regras)} regras).')