import sys
from itertools import product
from motor import carregar_regras, decidir


def verificar_base():
    regras = carregar_regras('rules.csv')

    regras_disparadas = set()
    lacunas = []
    pendentes = 0

    classes = ['baixo', 'medio', 'alto']
    valores_fato = [0, 1, None]

    print("=== AUDITORIA DA BASE DE CONHECIMENTO (verificar_base.py) ===\n")

    for c, e, p in product(classes, valores_fato, valores_fato):
        fatos = {'precisa_explicar': e, 'poucos_dados': p}
        res = decidir(c, fatos, regras)

        status = res['status']
        descricao = f"Fatos {c}, explic={e}, poucos={p}"

        if status == 'ok':
            regras_disparadas.add(res['regra'])
            print(f"{descricao} -> Disparou {res['regra']}")
        elif status == 'sem_regras':
            lacunas.append((c, fatos))
            print(f"{descricao} -> [AVISO] Sem regra aplicável!")
        elif status == 'nao_decidido':
            pendentes += 1
            print(f"{descricao} -> [PENDENTE] Perguntar: {res['perguntar']}")
        else:
            print(f"{descricao} -> [INESPERADO] {res}")

    print("\n--- RESUMO DA VERIFICAÇÃO ---")

    todas_regras = {r['id'] for r in regras}
    mortas = todas_regras - regras_disparadas

    if mortas:
        print(f"⚠️ Regras inalcançáveis (nunca disparam): {sorted(mortas)}")
    else:
        print("✅ Todas as regras cadastradas conseguem disparar adequadamente!")

    if lacunas:
        print(f"⚠️ Existem {len(lacunas)} combinações de fatos sem regra cadastrada.")
    else:
        print("✅ A base cobre todas as combinações de fatos testadas.")

    print(f"ℹ️  {pendentes} combinações ficaram pendentes (o motor pergunta ao usuário).")

    return not mortas and not lacunas


if __name__ == '__main__':
    ok = verificar_base()
    sys.exit(0 if ok else 1)