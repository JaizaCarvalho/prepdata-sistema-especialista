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

## Ética e Impactos da Inteligência Artificial

### 1. Descrição do projeto e papel da IA
O **PrepData** é um sistema especialista híbrido educacional criado para apoiar estudantes e profissionais iniciantes em Ciência de Dados a decidir qual estratégia de pré-processamento e modelagem adotar diante das características de um dataset (`n_linhas`, `pct_nulos`, `n_colunas_categoricas` e `desbalanceamento_pct`). Quem pode ser afetado por suas saídas são os próprios usuários que seguem a recomendação para tratar seus dados ou treinar modelos. A IA participa em duas frentes separadas: uma **Árvore de Decisão** (`DecisionTreeClassifier`, aprendizado supervisionado em `modelo.py`) prevê o nível de preparo necessário (`baixo`, `medio` ou `alto`) a partir de dados simulados, enquanto o **motor de regras** (`motor.py` + `regras.csv`) combina essa classe prevista com fatos de contexto informados pelo usuário (`precisa_explicar` e `poucos_dados`) para emitir a recomendação final. Assim, a classificação de risco/complexidade depende da previsão do modelo, mas **todas as recomendações técnicas** (`R1` a `R5`) decorrem exclusivamente das regras humanas escritas e validadas pela equipe com base em literatura técnica.

### 2. Análise ética do projeto

| Aspecto | Situação concreta, quem pode ser afetado e medida da equipe |
|---|---|
| **Privacidade e dados** | **Situação:** o sistema recebe apenas 4 métricas agregadas de um dataset (quantidade de linhas, percentual de nulos, quantidade de colunas categóricas e desbalanceamento) e foi treinado com 120 registros totalmente simulados em `gerar_dataset.py`, sem dados pessoais ou sensíveis. **Quem pode ser afetado:** caso alguém adapte o sistema para enviar amostras reais de bases corporativas ou pessoais, haveria risco se dados brutos fossem armazenados. **Medida:** o PrepData não recebe nem armazena registros individuais de pessoas, trabalha apenas com estatísticas numéricas agregadas e deixa explícito que a base de treino é simulada. |
| **Viés e equidade** | **Situação:** na base simulada, o critério de pontuação dá peso maior a `pct_nulos` e `desbalanceamento_pct`, e a árvore de profundidade 3 (`max_depth=3`) aprende cortes aproximados. Em um caso real de fronteira, como observado no **Caso 2** (`n_linhas=50000`, `pct_nulos=12%`, `desbalanceamento=20%`), o modelo previu `baixo` em vez de `medio`. **Quem pode ser afetado:** usuários com datasets grandes e problemas moderados podem receber recomendação simplificada demais (`R1` em vez de `R3`), ignorando nulos e variáveis categóricas — o que, em domínios sensíveis, pode prejudicar grupos minoritários mal representados. **Medida:** documentamos o erro do Caso 2 em `docs/analise_casos.md`, incluímos a matriz de confusão no `modelo.py` e mantemos regras explícitas para validar dados desbalanceados. |
| **Transparência** | **Situação:** o usuário recebe a recomendação final sem necessariamente saber por que ela foi escolhida ou se veio do modelo ou da regra. **Quem pode ser afetado:** o usuário que confia cegamente na resposta sem entender os critérios. **Medida:** o motor retorna sempre a explicação detalhada (`explicacao`) listando os fatos usados, o ID da regra aplicada (`R1`–`R5`), a fonte técnica (`fonte`), além de imprimir as condições da árvore (`export_text`) e recusar palpites quando falta informação (`status = 'nao_decidido'`). |
| **Supervisão e responsabilidade** | **Situação:** como visto no Caso 2, um erro estatístico da árvore propaga uma recomendação inadequada para o motor de regras. **Quem pode ser afetado:** o estudante ou analista que aplicar a recomendação sem revisar o dataset real. **Medida:** o sistema atua apenas como apoio didático (não executa alterações automáticas nos dados); a revisão da saída e a decisão final cabem sempre à pessoa usuária/profissional de dados, que pode corrigir os fatos (`precisa_explicar`, `poucos_dados`) ou revisar `regras.csv`. |
| **Impacto social** | **Situação:** um projeto educacional bem explicado ajuda iniciantes a evitar erros comuns de pré-processamento (como vazamento de dados ou ignorar validação cruzada). Por outro lado, se tratado como ferramenta definitiva em problemas reais de saúde, crédito ou seleção, pode induzir pipelines mal preparados. **Quem pode ser afetado:** iniciantes em aprendizado de máquina e, indiretamente, pessoas avaliadas por modelos treinados a partir de recomendações equivocadas. **Medida:** declarar expressamente o caráter exclusivamente educacional, a origem simulada dos dados e as limitações de amostra (120 registros). |

### 3. Dilema ético do projeto
**Dilema:** Ao perceber que a Árvore de Decisão erra casos de fronteira como o Caso 2 (acurácia de `0.60`), a equipe enfrenta a escolha entre:
1. **Ajustar manualmente hiperparâmetros ou remover casos difíceis do dataset** até a acurácia no teste subir artificialmente para perto de `1.00`: gera uma entrega visualmente "bonita", mas mascara a fragilidade do modelo em dados inéditos e induz confiança falsa no usuário.
2. **Manter a avaliação honesta, os 8% de ruído e documentar o erro de fronteira:** expõe que o modelo tem acurácia moderada (`0.60`), mas preserva a integridade científica e permite discutir onde o sistema falha e quando exige revisão humana.

A equipe optou pela **segunda alternativa**, alinhada ao princípio de integridade acadêmica e transparência (Lacerda Júnior, 2026): declarar limitações reais e não maquiar resultados é parte essencial do uso responsável de IA.

*(Nota sobre o debate em sala: a reflexão individual pós-debate será registrada por cada integrante após a discussão em aula, indicando qual decisão manteria ou mudaria e como verificaria sua eficácia.)*

---

## Declaração de uso de sistemas de IA

### IA incorporada ao projeto
O projeto utiliza **uma Árvore de Decisão (`DecisionTreeClassifier`, scikit-learn) integrada a um Sistema Especialista baseado em regras `SE/ENTÃO` (`motor.py` e `regras.csv`)** para **recomendar estratégias de preparo de dados e escolha de modelos**. O sistema recebe **quatro métricas numéricas do dataset (`n_linhas`, `pct_nulos`, `n_colunas_categoricas`, `desbalanceamento_pct`) e dois fatos de contexto (`precisa_explicar`, `poucos_dados`)** e gera **a classe de preparo prevista (`baixo`, `medio` ou `alto`), a recomendação técnica, a fonte e a explicação dos fatos e da regra aplicada**. A equipe verificou seu funcionamento por meio de **separação treino/teste (90/30) com matriz de confusão, 12 testes unitários no `motor.py`, 3 casos novos + experimento + 3 asserções (`assert`) no `main.py` e auditoria exaustiva das 27 combinações no `verificar_base.py`**. As limitações identificadas incluem **uso de 120 dados simulados gerados sobre os mesmos atributos, acurácia de `0.60` sujeita a erros de fronteira (como no Caso 2), cobertura restrita a 5 regras didáticas e ausência de tratamento para entradas fora do domínio de treino**. A revisão cabe **à pessoa usuária (estudante ou profissional de dados responsável pela análise)**.

### IA utilizada na elaboração do trabalho
Utilizei **assistentes de IA generativa (Antigravity/Gemini e DeepSeek)** para **apoio na estruturação dos scripts de dados/modelo, revisão da integração em `main.py`, organização da documentação em `docs/`, montagem do `notebook.ipynb` e revisão do texto deste `README.md`**. Aproveitei **sugestões de estrutura de código, comentários didáticos, casos de teste e revisão textual** após **executar e validar todos os scripts localmente no ambiente `.venv` (`gerar_dataset.py`, `modelo.py`, `motor.py`, `main.py` e `verificar_base.py`), conferir os resultados reais da matriz de confusão e dos testes `assert`, e revisar manualmente as fontes citadas**. Assumo a responsabilidade pelas escolhas, verificações e conteúdo da entrega.
