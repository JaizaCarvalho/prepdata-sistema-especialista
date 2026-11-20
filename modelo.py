"""
modelo.py (Edmar)

Treina uma árvore de decisão (aprendizado supervisionado) que, a partir de
4 características de um dataset, prevê o preparo necessário: baixo, medio
ou alto. Ao rodar `python modelo.py` mostra:
  1. a acurácia no teste;
  2. a matriz de confusão (onde o modelo erra);
  3. as condições que a árvore aprendeu.

O main.py importa treinar().
"""

from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

BASE_DIR = Path(__file__).resolve().parent

# Nomes e ordem iguais aos do dataset e do main.py.
ATRIBUTOS = ['n_linhas', 'pct_nulos', 'n_colunas_categoricas', 'desbalanceamento_pct']

# Ordem fixa da matriz de confusão.
CLASSES = ['baixo', 'medio', 'alto']


def treinar(caminho_csv='dataset.csv', mostrar=True):
    """Lê o dataset do CSV, treina a árvore e devolve o modelo.

    mostrar=False evita imprimir os resultados (usado pelo main.py).
    """
    caminho = Path(caminho_csv) if Path(caminho_csv).exists() else BASE_DIR / caminho_csv
    df = pd.read_csv(caminho)

    # O rótulo (y) nunca entra em X, senão o modelo recebe a resposta pronta.
    X = df[ATRIBUTOS]
    y = df['preparo_necessario']

    # 25% para teste. random_state fixa o sorteio e stratify mantém a
    # proporção das classes no treino e no teste.
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)

    # max_depth=3 limita a árvore e evita que ela decore os dados (overfitting).
    modelo = DecisionTreeClassifier(max_depth=3, random_state=42)
    modelo.fit(X_tr, y_tr)

    pred = modelo.predict(X_te)

    if mostrar:
        print('Registros: treino =', len(X_tr), '| teste =', len(X_te))
        print('Acurácia:', accuracy_score(y_te, pred))

        # Linhas = classe real, colunas = prevista. A diagonal são os acertos.
        print('Matriz de confusão (linhas = real, colunas = previsto;'
              ' ordem: baixo, medio, alto):')
        print(confusion_matrix(y_te, pred, labels=CLASSES))

        # Compare os cortes com o critério do gerar_dataset.py: se forem
        # parecidos, o modelo aprendeu o padrão.
        print('Condições aprendidas pela árvore:')
        print(export_text(modelo, feature_names=ATRIBUTOS))

    return modelo


if __name__ == '__main__':
    treinar()
