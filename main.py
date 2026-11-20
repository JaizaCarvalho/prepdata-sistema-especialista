import os
from pathlib import Path
import pandas as pd
from modelo import treinar, ATRIBUTOS
from motor import carregar_regras, decidir

# Garante que funciona mesmo rodando de fora da pasta do projeto (ex.: botao Run do VS Code)
os.chdir(Path(__file__).resolve().parent)

modelo = treinar(mostrar=False)
regras = carregar_regras()


def consultar(caso, fatos):
    # 1) Prever a classe do caso usando modelo.predict(pd.DataFrame([caso])[ATRIBUTOS])[0]
    classe = modelo.predict(pd.DataFrame([caso])[ATRIBUTOS])[0]
    # 2) Chamar decidir(...) passando a classe prevista, os fatos e as regras
    return classe, decidir(classe, fatos, regras)


if __name__ == '__main__':
    casos = [
        ("Caso 1", {'n_linhas': 800, 'pct_nulos': 32.0, 'n_colunas_categoricas': 11, 'desbalanceamento_pct': 38.0},
         {'precisa_explicar': 1, 'poucos_dados': 0}),
        ("Caso 2", {'n_linhas': 50000, 'pct_nulos': 12.0, 'n_colunas_categoricas': 4, 'desbalanceamento_pct': 20.0},
         {'precisa_explicar': 0, 'poucos_dados': 0}),
        ("Caso 3", {'n_linhas': 200000, 'pct_nulos': 2.0, 'n_colunas_categoricas': 3, 'desbalanceamento_pct': 5.0},
         {'precisa_explicar': 0, 'poucos_dados': 0}),
        ("Experimento (Caso 1 com precisa_explicar=0)",
         {'n_linhas': 800, 'pct_nulos': 32.0, 'n_colunas_categoricas': 11, 'desbalanceamento_pct': 38.0},
         {'precisa_explicar': 0, 'poucos_dados': 0}),
    ]

    for nome, caso, fatos in casos:
        classe, res = consultar(caso, fatos)
        print(f"\n=== {nome} ===")
        print("Classe prevista:", classe)
        print("Resultado:", res)

    # 3 testes do motor (consulte os IDs reais em regras.csv!)
    # Positivo: alto + precisa_explicar=1
    assert decidir('alto', {'precisa_explicar': 1, 'poucos_dados': 0}, regras)['regra'] == 'R4'
    # Negativo: baixo (não dispara regra de alto)
    assert decidir('baixo', {'precisa_explicar': 1, 'poucos_dados': 0}, regras)['regra'] == 'R1'
    # Informação ausente: alto sem saber precisa_explicar
    r_ausente = decidir('alto', {'precisa_explicar': None, 'poucos_dados': 0}, regras)
    assert r_ausente['status'] == 'nao_decidido' and r_ausente['perguntar'] == ['precisa_explicar']
    print("\nTodos os 3 testes do motor passaram!")