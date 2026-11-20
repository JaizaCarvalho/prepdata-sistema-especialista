# PrepData — Sistema Especialista + Aprendizado Supervisionado

**Problema em uma frase:** O PrepData recebe quatro características numéricas de um dataset (`n_linhas`, `pct_nulos`, `n_colunas_categoricas`, `desbalanceamento_pct`), prevê o nível de preparo necessário (`baixo`, `medio` ou `alto`) com uma árvore de decisão e, combinando essa classe com fatos de contexto (`precisa_explicar` e `poucos_dados`), aplica regras SE/ENTÃO para recomendar a estratégia de preparo e modelagem com explicação e fonte.

## Como executar
python3 -m venv .venv
source .venv/bin/activate        # No Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

python criar_banco.py            # gera dataset.csv e prepdata.db (120 registros, 5 regras)
python modelo.py                 # treina e avalia a árvore de decisão
python main.py                   # executa 3 casos novos, experimento e 3 testes assert
python verificar_base.py         # audita as 27 combinações da base de regras
```
Para executar no Google Colab, abra `notebook.ipynb` e clique em **Executar tudo**.

## Arquitetura
- **Dados e Banco (`criar_banco.py`, `dataset.csv`, `prepdata.db`):** gera 120 registros simulados e persiste as tabelas `dataset` e `regras` em SQLite.
- **Modelo Supervisionado (`modelo.py`):** `DecisionTreeClassifier(max_depth=3, random_state=42)` treinado sobre a tabela `dataset` de `prepdata.db`.
- **Base de Conhecimento e Motor (`regras.csv`, `motor.py`):** encadeamento para frente em um passo, com resolução de conflitos por **especificidade** (vence a regra com mais condições preenchidas) e tratamento explícito de informação ausente (`status = 'nao_decidido'`).
- **Integração (`main.py`):** conecta a classe prevista pelo modelo ao motor de regras via `consultar(caso, fatos)`.

## Tabela de regras (`regras.csv`)
| ID | `classe` | `precisa_explicar` | `poucos_dados` | Recomendação | Fonte |
|---|---|---|---|---|---|
| R1 | baixo | *(vazio)* | *(vazio)* | Treinar direto um modelo simples e avaliar com matriz de confusão | Rules of Machine Learning (regra 4); scikit-learn: confusion_matrix |
| R2 | medio | *(vazio)* | 1 | Imputar nulos; codificar categóricas e usar validação cruzada (imputador dentro de um Pipeline) | scikit-learn: Cross-validation; Imputation of missing values; OneHotEncoder |
| R3 | medio | *(vazio)* | 0 | Imputar nulos; codificar categóricas e treinar com separação treino/teste | scikit-learn: train_test_split; Imputation of missing values; OneHotEncoder |
| R4 | alto | 1 | *(vazio)* | Usar árvore de decisão (modelo interpretável) e mostrar as condições aprendidas; imputar nulos só se a versão usada exigir | scikit-learn: Decision Trees |
| R5 | alto | 0 | *(vazio)* | Testar modelos mais flexíveis e comparar com validação cruzada para vigiar overfitting | James et al. (2013) ISL: secao 2.1.3 |

*Mais detalhes e URLs das referências em `docs/fontes.md`.*

## Critério de rotulagem (dados simulados)
Os 120 registros (`medio: 63`, `alto: 30`, `baixo: 27`, `seed = 42`) são **simulados** por um critério de pontuação:
- `pct_nulos`: `< 5%` = 0 pt | `5% a 20%` = 1 pt | `> 20%` = 2 pts
- `n_colunas_categoricas`: `> 8` = 1 pt
- `desbalanceamento_pct` (quanto a classe majoritária excede 50%): `< 15` = 0 pt | `15 a 30` = 1 pt | `> 30` = 2 pts
- `n_linhas`: `< 500` = 1 pt
- **Faixas:** 0–1 pt = `baixo` | 2–3 pts = `medio` | 4+ pts = `alto`
- **Ruído:** ~8% dos registros têm o rótulo trocado para a classe vizinha, simulando casos de fronteira.

## Resultados do modelo (`modelo.py`)
- **Divisão:** 90 treino (75%) / 30 teste (25%), `stratify=y`, `random_state=42`.
- **Acurácia no teste:** `0.60` (18/30).
- **Matriz de confusão** (linhas = real, colunas = previsto; ordem: `baixo`, `medio`, `alto`):
  `[[5, 2, 0], [3, 8, 5], [0, 2, 5]]` — zero erros extremos entre `baixo` e `alto`.

## Três casos novos, experimento e testes (`main.py`)
- **Caso 1** (`800, 32%, 11, 38%`; `explicar=1, poucos=0`) → previsto `alto`, regra **`R4`** (acerto).
- **Caso 2** (`50000, 12%, 4, 20%`; `explicar=0, poucos=0`) → esperado `medio` (`R3`), previsto `baixo`, regra **`R1`** (erro de fronteira analisado em `docs/analise_casos.md`).
- **Caso 3** (`200000, 2%, 3, 5%`; `explicar=0, poucos=0`) → previsto `baixo`, regra **`R1`** (acerto).
- **Experimento** (Caso 1 com `precisa_explicar=0`): classe continua `alto`, mas regra muda de `R4` para **`R5`**.
- **Testes do motor (`assert`):** positivo (`alto`, `explicar=1` → `R4`), negativo (`baixo`, `explicar=1` → `R1`) e informação ausente (`alto`, `explicar=None` → `nao_decidido`, perguntar `['precisa_explicar']`) — todos passaram (além dos 12 testes locais em `motor.py`).

## Verificação da base (`verificar_base.py`)
Auditoria das 27 combinações (`3 classes × 3 valores de precisa_explicar × 3 valores de poucos_dados`):
- **0 regras mortas** (todas de `R1` a `R5` disparam);
- **0 lacunas (`sem_regras`)**;
- **6 combinações pendentes** (`nao_decidido`, pedindo o fato ausente).
- **Resolução de conflito:** especificidade (regra com mais condições preenchidas vence).

## Limitações
- O rótulo é gerado a partir de um critério sobre os mesmos 4 atributos (atenuado, mas não eliminado, pelo ruído de 8%).
- 120 registros simulados (30 no teste) formam uma amostra pequena; a acurácia não comprova eficácia em casos reais.
- Cinco regras cobrem apenas um recorte didático de decisões de preparo de dados.
- O sistema não trata entradas fora do intervalo observado no treino.

## Declaração de Ética
Este projeto tem finalidade exclusivamente educacional. Todos os dados são simulados e não representam datasets reais. As regras seguem fontes técnicas citadas em `docs/fontes.md`, e a recomendação do sistema não substitui a análise de um profissional de dados. A acurácia obtida com poucos registros simulados não comprova eficácia em casos reais. Ferramentas de IA foram usadas como apoio de estudo e desenvolvimento; o código e a documentação foram revisados e compreendidos pelos integrantes do grupo.