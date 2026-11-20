# Análise dos Casos Novos, Experimento e Testes — PrepData

Este documento registra e analisa os resultados reais obtidos na execução integrada do **PrepData** (`modelo.py`, `motor.py`, `main.py` e `verificar_base.py`).

---

## 1. Contexto do Modelo Treinado (`modelo.py`)

A árvore de decisão (`DecisionTreeClassifier(max_depth=3, random_state=42)`) foi treinada sobre os 120 registros simulados de `dataset.csv` (90 para treino, 30 para teste, `stratify=y`, `random_state=42`):

- **Acurácia no conjunto de teste:** `0.60` (18 acertos em 30 registros).
- **Matriz de confusão** (linhas = classe real, colunas = classe prevista; ordem: `baixo`, `medio`, `alto`):

| Real \ Previsto | `baixo` | `medio` | `alto` | Total real |
|---|---:|---:|---:|---:|
| **`baixo`** | **5** | 2 | 0 | 7 |
| **`medio`** | 3 | **8** | 5 | 16 |
| **`alto`** | 0 | 2 | **5** | 7 |

Observa-se que **não houve nenhum erro extremo** (nenhum caso `baixo` previsto como `alto`, nem `alto` previsto como `baixo`); todos os 12 erros ocorreram entre classes vizinhas, principalmente envolvendo a classe intermediária `medio`.

- **Condições aprendidas pela árvore (`export_text`):**

```text
|--- desbalanceamento_pct <= 22.40
|   |--- pct_nulos <= 10.10
|   |   |--- n_linhas <= 784.00 -> class: medio
|   |   |--- n_linhas >  784.00 -> class: baixo
|   |--- pct_nulos >  10.10
|   |   |--- n_linhas <= 46421.50 -> class: medio
|   |   |--- n_linhas >  46421.50 -> class: baixo
|--- desbalanceamento_pct >  22.40
|   |--- n_linhas <= 2796.50
|   |   |--- desbalanceamento_pct <= 27.70 -> class: alto
|   |   |--- desbalanceamento_pct >  27.70 -> class: alto
|   |--- n_linhas >  2796.50
|   |   |--- pct_nulos <= 15.95 -> class: medio
|   |   |--- pct_nulos >  15.95 -> class: alto
```

---

## 2. Tabela-Resumo dos 3 Casos Novos (`main.py`)

> **Nota sobre os IDs das regras:** na base oficial do grupo (`regras.csv`), as regras são: `R1` (`baixo`), `R2` (`medio` + `poucos_dados=1`), `R3` (`medio` + `poucos_dados=0`), `R4` (`alto` + `precisa_explicar=1`) e `R5` (`alto` + `precisa_explicar=0`).

| Caso | Atributos (`n_linhas`, `pct_nulos`, `n_colunas_categoricas`, `desbalanceamento_pct`) | Fatos (`precisa_explicar`, `poucos_dados`) | Esperado pelo critério | Previsto pelo modelo | Regra aplicada | Resultado | Recomendação |
|---|---|---|---|---|---|---|---|
| **Caso 1** | `800`, `32.0%`, `11`, `38.0%` | `1`, `0` | `alto` (`R4`) | **`alto`** | **`R4`** | Acerto | Usar árvore de decisão (modelo interpretável) e mostrar as condições aprendidas; imputar nulos só se a versão usada exigir |
| **Caso 2** | `50000`, `12.0%`, `4`, `20.0%` | `0`, `0` | `medio` (`R3`) | **`baixo`** | **`R1`** | **Erro do modelo** | Treinar direto um modelo simples e avaliar com matriz de confusão |
| **Caso 3** | `200000`, `2.0%`, `3`, `5.0%` | `0`, `0` | `baixo` (`R1`) | **`baixo`** | **`R1`** | Acerto | Treinar direto um modelo simples e avaliar com matriz de confusão |

---

## 3. Análise Individual dos 3 Casos

### Caso 1 — Dataset pequeno/médio com muitos problemas e necessidade de explicação
- **Entrada:** `n_linhas=800`, `pct_nulos=32.0`, `n_colunas_categoricas=11`, `desbalanceamento_pct=38.0`; fatos: `precisa_explicar=1`, `poucos_dados=0`.
- **Critério do especialista (`gerar_dataset.py`):** nulos > 20% (+2) + categóricas > 8 (+1) + desbalanceamento > 30% (+2) + linhas >= 500 (0) = **5 pontos** → esperado `alto` e regra `R4`.
- **Saída real:** classe prevista **`alto`**, regra **`R4`**, recomendação *"Usar arvore de decisao (modelo interpretavel) e mostrar as condicoes aprendidas; imputar nulos so se a versao usada exigir"* (fonte: `scikit-learn: Decision Trees`).
- **Explicação em uma frase:** Como a árvore classificou o dataset como preparo `alto` (por ter desbalanceamento acima de 22,4% e 800 linhas) e o usuário informou `precisa_explicar=1`, o motor aplicou a regra `R4` para recomendar um modelo interpretável.

### Caso 2 — Dataset grande com problemas moderados (erro do modelo analisado)
- **Entrada:** `n_linhas=50000`, `pct_nulos=12.0`, `n_colunas_categoricas=4`, `desbalanceamento_pct=20.0`; fatos: `precisa_explicar=0`, `poucos_dados=0`.
- **Critério do especialista (`gerar_dataset.py`):** nulos entre 5% e 20% (+1) + categóricas <= 8 (0) + desbalanceamento entre 15% e 30% (+1) + linhas >= 500 (0) = **2 pontos** → esperado `medio` e regra `R3`.
- **Saída real:** classe prevista **`baixo`**, regra **`R1`**, recomendação *"Treinar direto um modelo simples e avaliar com matriz de confusao"* (fonte: `Rules of Machine Learning (regra 4); scikit-learn: confusion_matrix`).
- **Explicação em uma frase:** Embora o critério do especialista pontue o caso como `medio` (2 pontos), a árvore o classificou como `baixo` ao seguir `desbalanceamento_pct <= 22.40`, `pct_nulos > 10.10` e `n_linhas > 46421.50`, fazendo o motor aplicar `R1` em vez de `R3`.

### Caso 3 — Dataset grande e limpo
- **Entrada:** `n_linhas=200000`, `pct_nulos=2.0`, `n_colunas_categoricas=3`, `desbalanceamento_pct=5.0`; fatos: `precisa_explicar=0`, `poucos_dados=0`.
- **Critério do especialista (`gerar_dataset.py`):** nulos < 5% (0) + categóricas <= 8 (0) + desbalanceamento < 15% (0) + linhas >= 500 (0) = **0 pontos** → esperado `baixo` e regra `R1`.
- **Saída real:** classe prevista **`baixo`**, regra **`R1`**, recomendação *"Treinar direto um modelo simples e avaliar com matriz de confusao"*.
- **Explicação em uma frase:** Como o dataset tem pouquíssimos nulos (2%) e baixo desbalanceamento (5%), a árvore previu `baixo` e o motor disparou `R1`, orientando começar com um modelo simples e avaliá-lo com matriz de confusão.

---

## 4. Discussão sobre o Erro do Modelo no Caso 2

O erro no Caso 2 é didaticamente importante e **não deve ser escondido**, pois evidencia três propriedades reais de sistemas híbridos:

1. **Caso de fronteira:** o Caso 2 soma exatamente **2 pontos** — o limite inferior da classe `medio` (0–1 = `baixo`, 2–3 = `medio`, 4+ = `alto`). Casos na fronteira entre classes são naturalmente os mais difíceis de classificar, ainda mais porque o dataset simulado inclui **8% de ruído** nos rótulos vizinhos.
2. **Restrição de profundidade e poucos exemplos de treino:** a árvore foi limitada a `max_depth=3` (para evitar overfitting) e treinada com apenas **90 registros**. Com esse limite de cortes e poucos exemplos na região de `n_linhas > 46.421`, a árvore aproximou os limiares (`desbalanceamento_pct <= 22.40` e `pct_nulos <= 10.10`, em vez de `15.0` e `5.0`) e nem chegou a utilizar o atributo `n_colunas_categoricas`. Assim, classificou essa folha como `baixo` em vez de `medio`.
3. **Propagação do erro do ML para o Sistema Especialista:** o motor de inferência (`motor.py`) funcionou perfeitamente — dada a classe `baixo`, a regra correta na base é de fato `R1`. Porém, como a previsão do modelo errou a classe (`baixo` em vez de `medio`), a recomendação final deixou de orientar a imputação dos 12% de valores nulos e a codificação das 4 colunas categóricas (que seriam sugeridas por `R3`).

**Lição prática:** em arquiteturas híbridas, testar o motor isoladamente garante que a lógica das regras está correta, mas a confiabilidade da recomendação final também depende da qualidade do modelo estatístico. Aumentar a base de treino, usar validação cruzada e manter revisão humana em casos de fronteira reduzem esse risco.

---

## 5. Experimento: Alterando Apenas o Fato `precisa_explicar`

Para comprovar a separação entre o que o modelo aprende e o que as regras decidem, executamos novamente o **Caso 1** mantendo os **mesmos 4 atributos do dataset** (`n_linhas=800`, `pct_nulos=32.0`, `n_colunas_categoricas=11`, `desbalanceamento_pct=38.0`) e mudando **apenas** `precisa_explicar` de `1` para `0`:

| Execução | Atributos do dataset | `precisa_explicar` | `poucos_dados` | Classe prevista | Regra aplicada | Recomendação |
|---|---|---:|---:|---|---|---|
| **Caso 1 original** | `800, 32.0, 11, 38.0` | `1` | `0` | **`alto`** | **`R4`** | Usar árvore de decisão (modelo interpretável) e mostrar as condições aprendidas; imputar nulos só se a versão usada exigir |
| **Experimento** | `800, 32.0, 11, 38.0` | `0` | `0` | **`alto`** | **`R5`** | Testar modelos mais flexíveis e comparar com validação cruzada para vigiar overfitting |

**Conclusão do experimento:** a classe prevista permaneceu **`alto`** porque `precisa_explicar` não é atributo do modelo supervisionado. Ainda assim, a recomendação mudou completamente (de `R4` para `R5`), provando que o contexto declarado pelo usuário controla a decisão simbólica no motor de regras sem interferir na classificação estatística do dataset.

---

## 6. Validação dos 3 Testes Unitários do Motor (`main.py`)

Seguindo a boa prática de testar a infraestrutura de regras independentemente do modelo de ML (*Rules of Machine Learning*), `main.py` executa três asserções diretas sobre `decidir(...)`:

1. **Teste positivo (`alto`, `precisa_explicar=1`, `poucos_dados=0`):** dispara `R4` (`status='ok'`). **Passou.**
2. **Teste negativo (`baixo`, `precisa_explicar=1`, `poucos_dados=0`):** dispara `R1` e **não** dispara `R4` (apesar de `precisa_explicar=1`). **Passou.**
3. **Teste de informação ausente (`alto`, `precisa_explicar=None`, `poucos_dados=0`):** devolve `status='nao_decidido'` e `perguntar=['precisa_explicar']`, comprovando que o motor não assume `0` ou `1` quando o fato não foi informado. **Passou.**

Complementarmente, a auditoria exaustiva das 27 combinações (`3 classes × 3 valores de precisa_explicar × 3 valores de poucos_dados`) em `verificar_base.py` confirmou **0 regras mortas**, **0 lacunas (`sem_regras`)** e **6 estados pendentes** (exatamente os casos em que falta `poucos_dados` para `medio` ou falta `precisa_explicar` para `alto`).
