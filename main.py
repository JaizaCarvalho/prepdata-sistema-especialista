import pandas as pd 
from motor import carregar_regras, decidir
# from modelo import treinar, ATRIBUTOS

# Atributos considerados do modelo 
ATRIBUTOS = ['n_linhas', 'pct_nulos', 'n_colunas_categoricas', 'desbalanceamento_pct']

def modelo_simulado(caso):
   """ 
   Função auxiliar de teste caso o modelo.py ainda não esteja pronto. Simula a previsão da árvore de decisão com base nas métricas do dataset. 
   """

    pct = caso.get('pct_nulos', 0)

    if pct < 5:
        return 'baixo'
    elif pct <= 20:
        return 'medio'
    else:
        return 'alto'

def consultar(caso, fatos, usar_modelo_real=False):
    """ 
    Integra a previsão da classe aos fatos de consulta e aciona o motor de inferência. 
    """