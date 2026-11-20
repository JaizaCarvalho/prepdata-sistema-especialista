"""
=====================================================================
modelo.py   (arquivo de Edmar)
=====================================================================

PARA QUE SERVE ESTE ARQUIVO?
Ele cria e treina o MODELO DE IA do projeto: uma árvore de decisão que
olha as 4 características de um dataset e adivinha se ele precisa de
preparo "baixo", "medio" ou "alto".

Quando você roda  python modelo.py  ele mostra 3 coisas:
    1. A acurácia (quantas previsões deram certo no teste).
    2. A matriz de confusão (onde o modelo erra).
    3. As condições que a árvore aprendeu (as "perguntas" que ela faz).

O main.py (pessoa B) importa a função treinar() daqui. Por isso o nome
da função e os parâmetros (db, mostrar) NÃO podem mudar sem avisar o grupo.

ISTO É IA DE VERDADE? SIM. É "aprendizado supervisionado":
mostramos muitos exemplos COM a resposta certa e o computador descobre o
padrão sozinho. Esse momento de "aprender" é a linha  modelo.fit(...).

O MODELO TEM "MEMÓRIA" ENTRE EXECUÇÕES? NÃO.
Cada vez que treinar() roda, nasce um modelo NOVO que estuda o dataset do
zero e o resultado antigo é descartado. O que muda o aprendizado é
mudar os DADOS (criar_banco.py), não rodar de novo.

GLOSSÁRIO:
- Atributos (X) ... as entradas: o que é medido de cada caso.
- Rótulo (y) ...... a resposta certa que queremos prever.
- Treino ........... parte dos dados que o modelo USA para aprender.
- Teste ............ parte que o modelo NUNCA viu; serve para dar a nota.
                     (Testar com os dados do treino só mediria decoreba.)
- Árvore de decisão  modelo que faz perguntas em sequência
                     ("nulos <= 10%?") até chegar numa classe.
- Acurácia ......... acertos / total de previsões no teste.
- Matriz de confusão tabela que cruza classe REAL x classe PREVISTA.
"""

# ---------------------------------------------------------------------
# BLOCO 0: IMPORTAÇÕES (ferramentas prontas)
# ---------------------------------------------------------------------
import sqlite3                      # ler o banco prepdata.db
import pandas as pd                 # trabalhar com tabelas (DataFrame)
from sklearn.metrics import accuracy_score, confusion_matrix  # as "notas"
from sklearn.model_selection import train_test_split          # separa treino/teste
from sklearn.tree import DecisionTreeClassifier, export_text  # a árvore

# ---------------------------------------------------------------------
# BLOCO 1: CONSTANTES (valores fixos usados em vários lugares)
# ---------------------------------------------------------------------
# Os atributos (colunas de entrada). A ordem e os nomes têm que ser
# IGUAIS aos do dataset e aos usados no main.py.
ATRIBUTOS = ['n_linhas', 'pct_nulos', 'n_colunas_categoricas', 'desbalanceamento_pct']

# As classes na ordem baixo -> medio -> alto. Serve para a matriz de
# confusão sair sempre nessa ordem.
CLASSES = ['baixo', 'medio', 'alto']


# ---------------------------------------------------------------------
# BLOCO 2: A FUNÇÃO treinar()
# "def" cria uma função: um bloco de código com nome, que podemos chamar
# quando quisermos (o main.py chama treinar()).
#   db='prepdata.db' -> qual banco ler (valor padrão se ninguém disser nada)
#   mostrar=True     -> se True, imprime acurácia, matriz e árvore;
#                       o main.py chama com False para não poluir a tela.
# ---------------------------------------------------------------------
def treinar(db='prepdata.db', mostrar=True):

    # --- Passo 1: LER OS DADOS DO BANCO -------------------------------
    con = sqlite3.connect(db)                           # abre o banco
    df = pd.read_sql('SELECT * FROM dataset', con)      # "pegue TODAS as colunas
                                                        #  da tabela dataset"
    con.close()                                         # fecha o banco

    # Limpeza opcional de linhas que você não quer usar no treino
    # (tire o "#" para ativar). Prefira fazer isso no criar_banco.py,
    # para o dataset.csv da entrega ficar igual ao que o modelo usou:
    #   df = df[df['pct_nulos'] <= 50]

    # --- Passo 2: SEPARAR ENTRADAS (X) E RESPOSTA (y) -----------------
    X = df[ATRIBUTOS]               # só as 4 colunas de entrada
    y = df['preparo_necessario']    # só a coluna da resposta certa
    # REGRA DE OURO: o rótulo (y) NUNCA pode estar dentro do X. Se
    # estivesse, o modelo receberia a resposta pronta ("vazamento").

    # --- Passo 3: SEPARAR TREINO E TESTE ------------------------------
    # test_size=0.25  -> 25% dos dados (30 linhas) ficam guardados para
    #                    o teste; os outros 75% (90) são para treinar.
    # random_state=42 -> semente do sorteio: a divisão sai igual sempre.
    # stratify=y      -> mantém a mesma proporção de baixo/medio/alto
    #                    no treino e no teste (senão uma classe pequena
    #                    poderia ficar fora do teste por azar).
    # A função devolve 4 pedaços: X de treino, X de teste, y de treino,
    # y de teste ("tr" = treino, "te" = teste).
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)

    # --- Passo 4: CRIAR E TREINAR A ÁRVORE ----------------------------
    # max_depth=3 limita a árvore a 3 níveis de perguntas. Árvore sem
    # limite decora os dados (overfitting) e perde utilidade.
    modelo = DecisionTreeClassifier(max_depth=3, random_state=42)

    # fit = "aprenda". Aqui acontece o TREINAMENTO: o modelo olha as 90
    # linhas de treino com suas respostas e escolhe as melhores perguntas.
    modelo.fit(X_tr, y_tr)

    # --- Passo 5: TESTAR ----------------------------------------------
    # predict = "preveja". O modelo recebe só as ENTRADAS do teste (sem a
    # resposta) e dá seu palpite para cada uma das 30 linhas.
    pred = modelo.predict(X_te)

    # --- Passo 6: MOSTRAR OS RESULTADOS (se mostrar=True) -------------
    if mostrar:
        print('Registros: treino =', len(X_tr), '| teste =', len(X_te))

        # Compara palpites (pred) com respostas certas (y_te).
        # Ex.: 0.6 = acertou 60% das 30 linhas = 18 acertos.
        print('Acurácia:', accuracy_score(y_te, pred))

        # Matriz 3x3: LINHAS = classe real, COLUNAS = classe prevista,
        # na ordem baixo, medio, alto. A diagonal (cima-esquerda até
        # baixo-direita) são os ACERTOS; o resto são erros.
        # Ex.: número na linha "medio", coluna "alto" = casos que eram
        # médio mas o modelo disse alto.
        print('Matriz de confusão (linhas = real, colunas = previsto;'
              ' ordem: baixo, medio, alto):')
        print(confusion_matrix(y_te, pred, labels=CLASSES))

        # Mostra a árvore como texto: cada "|---" é uma pergunta
        # (ex.: pct_nulos <= 10.10) e "class:" é a decisão final.
        # Compare com o critério do criar_banco.py: se os cortes forem
        # parecidos, o modelo aprendeu o padrão.
        print('Condições aprendidas pela árvore:')
        print(export_text(modelo, feature_names=ATRIBUTOS))

    # --- Passo 7: DEVOLVER O MODELO TREINADO --------------------------
    # "return" entrega o modelo para quem chamou a função (o main.py usa
    # esse modelo para prever casos novos).
    return modelo


# ---------------------------------------------------------------------
# BLOCO 3: RODAR DIRETO
# Esta condição significa: "só execute o que está abaixo se o arquivo
# foi rodado diretamente (python modelo.py)". Quando o main.py
# IMPORTA este arquivo, esse trecho NÃO roda.
# ---------------------------------------------------------------------
if __name__ == '__main__':
    treinar()