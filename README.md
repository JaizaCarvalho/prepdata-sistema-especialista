# PrepData — Sistema Especialista + Aprendizado Supervisionado

**Disciplina:** Inteligência Artificial e LFA (Trabalho A1)  
**Integrantes:** Edmar e Jaiza  

O **PrepData** recebe quatro características numéricas de um dataset (`n_linhas`, `pct_nulos`, `n_colunas_categoricas`, `desbalanceamento_pct`), prevê o nível de preparo necessário (`baixo`, `medio` ou `alto`) com uma **Árvore de Decisão** e combina essa previsão com fatos de contexto (`precisa_explicar` e `poucos_dados`) em um **motor de regras SE/ENTÃO** para recomendar a estratégia de preparo e modelagem com explicação e fonte.

---

## Como executar

```bash
python3 -m venv .venv
source .venv/bin/activate        # No Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

python gerar_dataset.py          # gera dataset.csv (120 registros simulados)
python modelo.py                 # treina e avalia a árvore de decisão
python motor.py                  # executa os 12 testes locais do motor
python main.py                   # executa 3 casos novos, experimento e 3 testes assert
python verificar_base.py         # audita as 27 combinações da base de regras
```

Para executar no Google Colab, abra [`notebook.ipynb`](notebook.ipynb) e clique em **Ambiente de execução → Executar tudo**.

---

## Arquitetura do sistema

- **Dados simulados ([`gerar_dataset.py`](gerar_dataset.py), [`dataset.csv`](dataset.csv)):** gera 120 registros simulados (`seed = 42`) com as quatro métricas de entrada e o rótulo `preparo_necessario`.
- **Modelo Supervisionado ([`modelo.py`](modelo.py)):** lê `dataset.csv` e treina uma `DecisionTreeClassifier(max_depth=3, random_state=42)` para prever `baixo`, `medio` ou `alto`.
- **Base de Conhecimento e Motor ([`regras.csv`](regras.csv), [`motor.py`](motor.py)):** implementa encadeamento para frente em um passo. A classe prevista pelo modelo entra na memória de trabalho junto com os fatos de contexto. Em caso de conflito, vence a regra **mais específica** (com mais condições preenchidas). Quando falta um fato necessário, o motor não assume valor padrão: retorna `status = 'nao_decidido'` indicando o que perguntar.
- **Integração ([`main.py`](main.py)):** conecta `modelo.py` e `motor.py` por meio da função `consultar(caso, fatos)`.

---

## Base de regras ([`regras.csv`](regras.csv))

Células vazias significam *"não importa"* (`None`).

| ID | `classe` | `precisa_explicar` | `poucos_dados` | Recomendação | Fonte |
|---|---|---|---|---|---|
| **R1** | `baixo` | *(vazio)* | *(vazio)* | Treinar direto um modelo simples e avaliar com matriz de confusão | Rules of Machine Learning (regra 4); scikit-learn: confusion_matrix |
| **R2** | `medio` | *(vazio)* | `1` | Imputar nulos; codificar categóricas e usar validação cruzada (imputador dentro de um Pipeline) | scikit-learn: Cross-validation; Imputation of missing values; OneHotEncoder |
| **R3** | `medio` | *(vazio)* | `0` | Imputar nulos; codificar categóricas e treinar com separação treino/teste | scikit-learn: train_test_split; Imputation of missing values; OneHotEncoder |
| **R4** | `alto` | `1` | *(vazio)* | Usar árvore de decisão (modelo interpretável) e mostrar as condições aprendidas; imputar nulos só se a versão usada exigir | scikit-learn: Decision Trees |
| **R5** | `alto` | `0` | *(vazio)* | Testar modelos mais flexíveis e comparar com validação cruzada para vigiar overfitting | James et al. (2013) ISL: secao 2.1.3 |

*As referências completas (título, autor/órgão, URL e resumo) estão em [`docs/fontes.md`](docs/fontes.md).*

---

## Critério de rotulagem (dados simulados)

Os **120 registros** (`medio: 63`, `alto: 30`, `baixo: 27`) são **simulados** por um critério de pontuação baseado nos problemas do dataset:

- `pct_nulos`: `< 5%` = 0 pt | `5% a 20%` = 1 pt | `> 20%` = 2 pts
- `n_colunas_categoricas`: `> 8` = 1 pt
- `desbalanceamento_pct` *(quanto a classe majoritária excede 50%; 0 = balanceado)*: `< 15` = 0 pt | `15 a 30` = 1 pt | `> 30` = 2 pts
- `n_linhas`: `< 500` = 1 pt
- **Faixas:** `0–1` pt = `baixo` | `2–3` pts = `medio` | `4+` pts = `alto`
- **Ruído de fronteira:** ~8% dos registros têm o rótulo trocado para uma classe vizinha, simulando discordância entre especialistas.

---

## Resultados do modelo ([`modelo.py`](modelo.py))

- **Divisão:** 90 registros para treino (75%) e 30 para teste (25%), com `stratify=y` e `random_state=42`.
- **Acurácia no teste:** `0.60` (18 acertos em 30).
- **Matriz de confusão** (linhas = real, colunas = previsto; ordem: `baixo`, `medio`, `alto`):

```text
[[5 2 0]
 [3 8 5]
 [0 2 5]]
```

Não houve nenhum erro extremo entre `baixo` e `alto`; todos os erros ocorreram entre classes vizinhas.

---

## Três casos novos, experimento e testes ([`main.py`](main.py))

- **Caso 1** (`800, 32.0%, 11, 38.0%`; `precisa_explicar=1, poucos_dados=0`) → previsto `alto`, regra **`R4`** (acerto).
- **Caso 2** (`50000, 12.0%, 4, 20.0%`; `precisa_explicar=0, poucos_dados=0`) → esperado `medio` (`R3`), previsto `baixo`, regra **`R1`** (erro de fronteira analisado em [`docs/analise_casos.md`](docs/analise_casos.md)).
- **Caso 3** (`200000, 2.0%, 3, 5.0%`; `precisa_explicar=0, poucos_dados=0`) → previsto `baixo`, regra **`R1`** (acerto).
- **Experimento** *(Caso 1 alterando apenas `precisa_explicar` de `1` para `0`)*: a classe prevista continua **`alto`**, mas a regra muda de `R4` para **`R5`**.
- **Testes do motor (`assert`):** positivo (`alto`, `precisa_explicar=1` → `R4`), negativo (`baixo`, `precisa_explicar=1` → `R1`) e informação ausente (`alto`, `precisa_explicar=None` → `nao_decidido`, perguntar `['precisa_explicar']`) — todos passaram, além dos 12 testes locais em [`motor.py`](motor.py).

---

## Verificação da base ([`verificar_base.py`](verificar_base.py))

Auditoria exaustiva das 27 combinações (`3 classes × 3 valores de precisa_explicar × 3 valores de poucos_dados`):
- **0 regras mortas** (todas de `R1` a `R5` disparam);
- **0 lacunas** (`sem_regras`);
- **6 combinações pendentes** (`nao_decidido`, solicitando `poucos_dados` para `medio` ou `precisa_explicar` para `alto`).

---

## Documentação complementar

- [`docs/estudo_inicial.md`](docs/estudo_inicial.md): respostas das 5 perguntas do estudo inicial e das 4 perguntas de análise.
- [`docs/analise_casos.md`](docs/analise_casos.md): tabela e análise dos 3 casos novos, discussão do erro no Caso 2 e resultado do experimento.
- [`docs/fontes.md`](docs/fontes.md): fontes técnicas detalhadas de cada regra.

---

## Limitações

- O rótulo é gerado por um critério aplicado aos mesmos 4 atributos (o ruído de 8% e a combinação de critérios atenuam o problema, mas não o eliminam).
- A amostra é pequena e simulada (120 registros, 30 no teste): a acurácia obtida não comprova eficácia em datasets reais.
- As 5 regras cobrem apenas um recorte didático das decisões de preparo de dados.
- O sistema não trata entradas fora do intervalo observado no treino.

---

## Declaração de Ética

Este projeto tem finalidade exclusivamente educacional. Todos os dados são simulados e não representam datasets reais. As regras seguem fontes técnicas citadas em [`docs/fontes.md`](docs/fontes.md), e a recomendação do sistema não substitui a análise de um profissional de dados. A acurácia obtida com poucos registros simulados não comprova eficácia em casos reais. Ferramentas de IA foram usadas como apoio de estudo e desenvolvimento; o código e a documentação foram revisados e compreendidos pelos integrantes do grupo.
