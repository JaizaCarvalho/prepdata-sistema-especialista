# Estudo Inicial e Perguntas de Análise — PrepData

Este documento reúne as respostas das cinco perguntas conceituais do estudo inicial sobre aprendizado supervisionado e das quatro perguntas de análise aplicadas especificamente ao projeto **PrepData** (Sistema Especialista híbrido para orientação de preparo de dados).

---

## Parte 1 — Respostas do Estudo Inicial

### 1. O que é aprendizado de máquina?
Aprendizado de máquina (*Machine Learning*) é uma área da Inteligência Artificial em que um programa **aprende padrões diretamente a partir de dados**, em vez de depender exclusivamente de regras programadas manualmente para cada situação. Em vez de um programador escrever cada condição fixa para classificar um problema, o algoritmo ajusta parâmetros internos com base em exemplos para conseguir generalizar e responder a casos novos.

### 2. O que caracteriza o aprendizado supervisionado?
No aprendizado supervisionado, o algoritmo é treinado com **exemplos que já trazem a resposta conhecida (rótulo)**. Durante o treinamento, o modelo analisa a relação entre as entradas e os rótulos corretos para aprender uma função de mapeamento. Depois, quando recebe um exemplo novo sem rótulo, ele usa o padrão aprendido para prever a resposta. No PrepData, o algoritmo supervisionado é uma árvore de decisão (`DecisionTreeClassifier`) que aprende a prever o nível de preparo necessário de um dataset (`baixo`, `medio` ou `alto`).

### 3. O que são atributos e rótulos?
- **Atributos (`X`):** são as variáveis de entrada medidas em cada exemplo — aquilo que o modelo observa para fazer a previsão. No PrepData, os 4 atributos são `n_linhas`, `pct_nulos`, `n_colunas_categoricas` e `desbalanceamento_pct`.
- **Rótulo (`y`):** é a resposta conhecida que queremos que o modelo aprenda a prever. No PrepData, o rótulo é a coluna `preparo_necessario` (`baixo`, `medio` ou `alto`). O rótulo **nunca** deve entrar junto com os atributos `X`, pois isso daria a resposta pronta ao modelo.

### 4. Qual é a diferença entre classificação e regressão?
- **Classificação:** o modelo prevê uma **categoria discreta** (classe), como `baixo`, `medio` ou `alto` (caso do PrepData), ou *spam* vs. *legítima*.
- **Regressão:** o modelo prevê um **valor numérico contínuo**, como o preço de um imóvel, a temperatura de amanhã ou o consumo de energia em kWh.

### 5. Por que separar os dados em treino e teste?
Separamos os dados em treino e teste para **avaliar a capacidade de generalização do modelo em exemplos que ele nunca viu**. Se avaliássemos o modelo nos mesmos dados usados para treiná-lo, estaríamos medindo apenas **memorização** (*overfitting*), obtendo uma impressão falsamente otimista do desempenho. No PrepData (`modelo.py`), dividimos os 120 registros simulados em **90 para treino (75%)** e **30 para teste (25%)**, usando `stratify=y` para manter a proporção das classes (`baixo`, `medio`, `alto`) nos dois conjuntos e `random_state=42` para garantir reprodutibilidade.

---

## Parte 2 — Perguntas de Análise Aplicadas ao PrepData

### 1. Qual parte do programa aprende com os dados e qual aplica regras humanas?
O PrepData é um **sistema híbrido**, no qual as responsabilidades são separadas em dois módulos distintos:
- **Aprende com os dados (`modelo.py`):** uma árvore de decisão (`DecisionTreeClassifier` com `max_depth=3`) treinada sobre o `dataset.csv` (90 exemplos de treino). Ela descobre automaticamente cortes nos 4 atributos numéricos (`n_linhas`, `pct_nulos`, `n_colunas_categoricas`, `desbalanceamento_pct`) e devolve uma **classe prevista** (`baixo`, `medio` ou `alto`).
- **Aplica regras humanas (`motor.py` + `regras.csv`):** o motor de inferência carrega as 5 regras `SE... ENTÃO...` (`R1` a `R5`), escritas pelo grupo com base em fontes técnicas em `regras.csv`. Ele recebe a classe prevista pelo modelo como um fato na memória de trabalho, combina essa classe com os fatos de contexto informados pelo usuário (`precisa_explicar` e `poucos_dados`), resolve eventuais conflitos por **especificidade** (a regra com mais condições preenchidas vence) e devolve a **recomendação**, a **fonte** e a **explicação** explícita.

A integração acontece em `main.py` por meio da função `consultar(caso, fatos)`.

### 2. Por que o fato de consulta não aparece entre os atributos do modelo? O que acontece ao mudar apenas esse fato?
Os fatos de consulta (`precisa_explicar` e `poucos_dados`) representam **informações de contexto declaradas pelo usuário** sobre a necessidade do projeto ou restrições da análise — não são métricas estatísticas medidas diretamente dentro da tabela de dados. Por isso, eles não entram como atributos (`X`) da árvore de decisão.

**O que acontece ao mudar apenas um fato de consulta:** a classe prevista pelo modelo **não muda** (porque os atributos do dataset continuam idênticos), mas o motor de regras pode disparar uma **regra diferente** e mudar completamente a recomendação.

**Exemplo real executado no PrepData (`main.py`):**
- No **Caso 1** (`n_linhas=800`, `pct_nulos=32.0`, `n_colunas_categoricas=11`, `desbalanceamento_pct=38.0`):
  - Com `precisa_explicar = 1` (`poucos_dados = 0`): o modelo prevê **`alto`** e o motor aplica **`R4`** (*"Usar árvore de decisão (modelo interpretável) e mostrar as condições aprendidas; imputar nulos só se a versão usada exigir"*).
  - Mantendo exatamente os mesmos atributos e alterando apenas `precisa_explicar = 0`: o modelo continua prevendo **`alto`**, mas o motor passa a aplicar **`R5`** (*"Testar modelos mais flexíveis e comparar com validação cruzada para vigiar overfitting"*).

### 3. Um resultado de 100% (ou acurácia alta) em poucos casos é suficiente para uso real? Por quê?
**Não.** Mesmo que um modelo atingisse 100% de acurácia em um conjunto pequeno de teste, isso **não comprova eficácia em cenários reais**, por quatro motivos principais:
1. **Amostra pequena tem alta variância:** no PrepData temos apenas 120 registros simulados (30 no teste). Em 30 exemplos, cada acerto ou erro desloca a acurácia em ~3,3 pontos percentuais; uma avaliação em poucos casos não representa a diversidade de datasets reais.
2. **Dados simulados e critério derivado dos mesmos atributos:** nossos rótulos foram gerados por uma função de pontuação sobre os mesmos 4 atributos usados no treino (com ~8% de ruído em casos de fronteira). Quando os dados são sintéticos e bem separáveis, uma acurácia alta pode apenas refletir que o modelo reaprendeu a fórmula artificial, não o mundo real.
3. **Acurácia sozinha esconde tipos de erro:** é preciso olhar a **matriz de confusão** para saber *onde* o modelo erra. No nosso resultado real (`acurácia = 0,60`), a matriz `[[5, 2, 0], [3, 8, 5], [0, 2, 5]]` mostra que todos os erros ocorreram entre classes vizinhas (`baixo`/`medio` e `medio`/`alto`), com **zero erros extremos** entre `baixo` e `alto`.
4. **Necessidade de validação adequada:** para uso real, seria necessário coletar um volume muito maior de datasets reais rotulados por especialistas independentes e avaliar o modelo com **validação cruzada** (*k-fold cross-validation*), sem ficar ajustando hiperparâmetros diretamente sobre o conjunto de teste.

### 4. Qual conhecimento foi difícil de transformar em regra? Quando a recomendação deveria ser suspensa? Que evidência mostraria que a base precisa de revisão?
- **Conhecimento difícil de transformar em regra:** o **conhecimento tácito** de profissionais de ciência de dados. Na prática, decidir como preparar dados envolve nuances difíceis de reduzir a poucos limiares fixos — por exemplo, saber se valores nulos são aleatórios ou informativos (MCAR/MAR/MNAR), se colunas categóricas têm alta cardinalidade, qual é o custo de um falso negativo no domínio do negócio ou se há vazamento de dados temporal. Cinco regras cobrem apenas um recorte didático dessas decisões.
- **Quando a recomendação deve ser suspensa:**
  1. **Informação ausente (`status = 'nao_decidido'`):** quando falta um fato necessário para decidir (por exemplo, classe `alto` com `precisa_explicar = None` ou `medio` com `poucos_dados = None`). O motor do PrepData nunca "chuta" um valor padrão: ele suspende a decisão e lista exatamente qual fato perguntar.
  2. **Fora do domínio (`status = 'sem_regras'`):** quando a entrada recebe uma classe ou tipo de dado desconhecido ou valores muito fora do intervalo visto no treino (`n_linhas` entre 53 e 399.447, `pct_nulos` até 46,6%, `n_colunas_categoricas` de 0 a 14, `desbalanceamento_pct` até 47,7%).
  3. **Uso em ambientes reais ou sensíveis:** como o sistema é educacional e treinado em dados simulados, qualquer decisão real de engenharia de dados deve ser revisada por um profissional.
- **Evidências de que a base precisa de revisão:**
  1. **Regras inalcançáveis (mortas) ou lacunas na cobertura:** detectadas pela auditoria automática em `verificar_base.py` (no estado atual, temos 0 regras mortas e 0 lacunas nas 27 combinações testadas).
  2. **Conflitos ou recomendações contraditórias** entre regras que disparam juntas sem justificativa clara de especificidade.
  3. **Discordância recorrente de especialistas** em casos reais ou desatualização técnica das recomendações (por exemplo, mudanças nas bibliotecas citadas).
  4. **Erros do classificador propagados para o motor** em casos importantes ou queda de desempenho ao receber dados novos (como observado na análise do Caso 2 em `docs/analise_casos.md`).
