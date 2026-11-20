import pandas as pd

CONDICOES = [
    "classe",
    "precisa_explicar",
    "poucos_dados"
]

def carregar_regras(caminho_csv="regras.csv"):
    """
    Lê o arquivo regras.csv e converte para uma lista de dicionários.
    Valores vazios no CSV são convertidos para None ('não importa').
    """
    df = pd.read_csv(caminho_csv) 
    df = df.astype(object).where(pd.notnull(df), None)

    for col in ['precisa_explicar', 'poucos_dados']:
        valores = [
            None if pd.isna(x) else int(x)
            for x in df[col]
        ]
        df[col] = pd.Series(valores, index=df.index, dtype=object)

    return df.to_dict(orient='records')

def decidir(classe, fatos, regras):
    """ 
    Aplica a inferência lógica combinando a classe prevista e os fatos informados.
    """
    memoria = {'classe': classe, **fatos}

    aplicaveis = []
    pendentes = []

    for r in regras:
        exigidas = {c: r[c] for c in CONDICOES if r[c] is not None}

        if any(memoria.get(c) is not None and memoria[c] != v for c,v in exigidas.items()):
            continue
        faltando = [c for c in exigidas if memoria.get(c) is None]

        if faltando:
            pendentes.append((r, len(exigidas), faltando))
        else:
            aplicaveis.append((r, len(exigidas)))

    melhor = max(aplicaveis, key=lambda x: x[1], default=None)
    n_melhor = melhor[1] if melhor else -1

    precisa = sorted({f for _, n, fs in pendentes if n > n_melhor for f in fs})

    if precisa:
        return {
            'status': 'nao_decidido',
            'perguntar': precisa
                }
    if melhor is None:
        return {
            'status': 'sem_regras',
                'memoria': memoria
                }

    r = melhor[0]
    usados = {c: memoria[c] for c in CONDICOES if r[c] is not None}

    return {
        'status': 'ok',
        'regra': r['id'], 
        'recomendacao': r['recomendacao'],
        'fonte': r['fonte'],
        'explicacao': f"Fatos usados: {usados}. Regra {r['id']} aplicada (a mais específica entre {len(aplicaveis)} aplicável(is))."
            }

# Teste local do motor sem depender do modelo

if __name__ == '__main__':
    regras = carregar_regras()

    print("--- Teste 1: Positivo (Medio + Poucos Dados = 1) ---")
    res1 = decidir('medio', {'precisa_explicar': None, 'poucos_dados': 1}, regras)
    print(res1)

    print("\n--- Teste 2: Informação Ausente (Alto sem saber se precisa explicar) ---")
    res2 = decidir('alto',{'precisa_explicar':  None, 'poucos_dados': 0}, regras)
    print(res2)

    print("\n--- Teste 3: Positivo (baixo + outos fatos não importam) ---")
    res3 = decidir('baixo',{'precisa_explicar': 1, 'poucos_dados': 1},regras)
    print(res3)

    print("\n--- Teste 4: Sem Regra ---")
    res4 = decidir('muito_alto',{'precisa_explicar': None, 'poucos_dados': None},regras)
    print(res4)

    print("\n--- Teste 5: Uma regra por vez (Medio + Poucos Dados = 0 -> R3) ---")
    res5 = decidir('medio', {'precisa_explicar': None, 'poucos_dados': 0}, regras)
    print(res5)

    print("\n--- Teste 6: Alto + Precisa Explicar = 1 -> R4 ---")
    res6 = decidir('alto', {'precisa_explicar': 1, 'poucos_dados': None}, regras)
    print(res6)

    print("\n--- Teste 7: Alto + Precisa Explicar = 0 -> R5 ---")
    res7 = decidir('alto', {'precisa_explicar': 0, 'poucos_dados': None}, regras)
    print(res7)

    print("\n--- Teste 8: Medio sem fatos (deve perguntar só poucos_dados) ---")
    res8 = decidir('medio', {'precisa_explicar': None, 'poucos_dados': None}, regras)
    print(res8)

    print("\n--- Teste 9: Alto sem fatos (deve perguntar só precisa_explicar) ---")
    res9 = decidir('alto', {'precisa_explicar': None, 'poucos_dados': None}, regras)
    print(res9)

    print("\n--- Teste 10: Baixo sem fatos (decide R1 sem perguntar nada) ---")
    res10 = decidir('baixo', {'precisa_explicar': None, 'poucos_dados': None}, regras)
    print(res10)

    print("\n--- Teste 11: Classe com maiuscula (esperado: sem_regras) ---")
    res11 = decidir('Medio', {'precisa_explicar': None, 'poucos_dados': 1}, regras)
    print(res11)

    print("\n--- Teste 12: Fato com tipo errado, '1' string (esperado: sem_regras) ---")
    res12 = decidir('medio', {'precisa_explicar': None, 'poucos_dados': '1'}, regras)
    print(res12)