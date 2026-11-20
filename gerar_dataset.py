"""
gerar_dataset.py (arquivo de Edmar)

Gera o dataset simulado (120 registros) usado para treinar e avaliar a árvore
de decisão em modelo.py, salvando o resultado em dataset.csv.

Este script não tem memória: cada execução recria dataset.csv do zero com
semente fixa (seed = 42) para que o grupo inteiro obtenha os mesmos dados.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

# Garante que salva na pasta do projeto mesmo se executado de fora (ex.: botão Run)
os.chdir(Path(__file__).resolve().parent)

N = 120
NIVEIS = ['baixo', 'medio', 'alto']

rng = np.random.default_rng(42)

df = pd.DataFrame({
    'n_linhas': np.exp(rng.uniform(np.log(50), np.log(500000), N)).astype(int),
    'pct_nulos': np.round(rng.beta(1.2, 5, N) * 60, 1),
    'n_colunas_categoricas': rng.integers(0, 15, N),
    'desbalanceamento_pct': np.round(rng.uniform(0, 48, N), 1),
})


def pontuar(linha):
    """Recebe uma linha e devolve a pontuação de problemas, de 0 a 6."""
    p = 0 if linha.pct_nulos < 5 else (1 if linha.pct_nulos <= 20 else 2)
    p += 1 if linha.n_colunas_categoricas > 8 else 0
    p += 0 if linha.desbalanceamento_pct < 15 else (1 if linha.desbalanceamento_pct <= 30 else 2)
    p += 1 if linha.n_linhas < 500 else 0
    return p


def rotular(p):
    """Transforma a pontuação (p) em classe: 0-1 baixo, 2-3 medio, 4+ alto."""
    return NIVEIS[0] if p <= 1 else (NIVEIS[1] if p <= 3 else NIVEIS[2])


df['preparo_necessario'] = [rotular(pontuar(l)) for l in df.itertuples()]

# Casos de fronteira: troca o rótulo de ~8% dos registros para a classe vizinha
for i in rng.choice(N, size=int(N * 0.08), replace=False):
    k = NIVEIS.index(df.loc[i, 'preparo_necessario']) + int(rng.choice([-1, 1]))
    df.loc[i, 'preparo_necessario'] = NIVEIS[min(max(k, 0), 2)]

print('Registros por classe:')
print(df['preparo_necessario'].value_counts())

df.to_csv('dataset.csv', index=False)
print(f'Pronto: dataset.csv criado ({len(df)} registros).')
