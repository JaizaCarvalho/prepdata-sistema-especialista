# Fontes Técnicas e Origem das Regras — PrepData

Este documento registra as fontes técnicas que fundamentam cada uma das cinco regras da base de conhecimento (`regras.csv`), além das referências de apoio usadas na arquitetura, testes e discussão de limitações.

> **Ambiente de referência utilizado na construção e conferência:** Python 3.14.4, `scikit-learn` 1.9.1, `pandas` 3.1.1 e `numpy` 2.5.5.

---

## Resumo da Base de Regras (`regras.csv`)

| ID | Condições (`SE`) | Recomendação (`ENTÃO`) | Fonte registrada em `regras.csv` |
|---|---|---|---|
| **R1** | `classe = baixo` | Treinar direto um modelo simples e avaliar com matriz de confusão | Rules of Machine Learning (regra 4); scikit-learn: confusion_matrix |
| **R2** | `classe = medio` E `poucos_dados = 1` | Imputar nulos; codificar categóricas e usar validação cruzada (imputador dentro de um Pipeline) | scikit-learn: Cross-validation; Imputation of missing values; OneHotEncoder |
| **R3** | `classe = medio` E `poucos_dados = 0` | Imputar nulos; codificar categóricas e treinar com separação treino/teste | scikit-learn: train_test_split; Imputation of missing values; OneHotEncoder |
| **R4** | `classe = alto` E `precisa_explicar = 1` | Usar árvore de decisão (modelo interpretável) e mostrar as condições aprendidas; imputar nulos só se a versão usada exigir | scikit-learn: Decision Trees |
| **R5** | `classe = alto` E `precisa_explicar = 0` | Testar modelos mais flexíveis e comparar com validação cruzada para vigiar overfitting | James et al. (2013) ISL: secao 2.1.3 |

---

## Detalhamento das Fontes por Regra

### Regra R1 (`classe = baixo`)
**Recomendação:** *Treinar direto um modelo simples e avaliar com matriz de confusão.*

1. **Rules of Machine Learning: Best Practices for ML Engineering (Regra #4)**
   - **Autor / Órgão:** Martin Zinkevich (Google)
   - **Versão / Data:** Guia oficial online (43 regras de engenharia de ML)
   - **URL:** https://developers.google.com/machine-learning/guides/rules-of-ml
   - **O que aproveitamos (com nossas palavras):** A Regra #4 (*"Keep the first model simple and get the infrastructure right"*) orienta a começar sempre por um modelo simples quando a situação permite, para validar rapidamente o fluxo de ponta a ponta e criar uma linha de base (*baseline*) antes de investir em técnicas complexas. Usamos esse princípio para justificar que, quando o dataset exige preparo `baixo`, o melhor primeiro passo é treinar um modelo simples diretamente.

2. **scikit-learn API Reference: `sklearn.metrics.confusion_matrix` (e User Guide: Classification Metrics)**
   - **Autor / Órgão:** scikit-learn developers
   - **Versão / Data:** Documentação estável (conferida na versão `1.9.1`)
   - **URL:** https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
   - **O que aproveitamos (com nossas palavras):** Mostra como construir e interpretar a matriz de confusão, cruzando classes reais nas linhas e classes previstas nas colunas. Aproveitamos essa referência para recomendar a matriz de confusão já na primeira avaliação, pois a acurácia isolada não mostra *quais* classes o modelo está confundindo.

---

### Regra R2 (`classe = medio` E `poucos_dados = 1`)
**Recomendação:** *Imputar nulos; codificar categóricas e usar validação cruzada (imputador dentro de um Pipeline).*

1. **scikit-learn User Guide: Cross-validation: evaluating estimator performance**
   - **Autor / Órgão:** scikit-learn developers
   - **Versão / Data:** Documentação estável (conferida na versão `1.9.1`)
   - **URL:** https://scikit-learn.org/stable/modules/cross_validation.html
   - **O que aproveitamos (com nossas palavras):** Explica que, quando há poucos dados, separar apenas uma única partição de validação desperdiça exemplos de treino e produz estimativas instáveis. A validação cruzada (*k-fold*) aproveita melhor o conjunto disponível. Também aproveitamos o alerta da documentação sobre **vazamento de dados** (*data leakage*): transformadores como imputadores devem ficar dentro de um `Pipeline`, para que estatísticas (média, mediana etc.) sejam calculadas apenas em cada fold de treino e nunca usando o fold de teste.

2. **scikit-learn User Guide: Imputation of missing values**
   - **Autor / Órgão:** scikit-learn developers
   - **Versão / Data:** Documentação estável (conferida na versão `1.9.1`)
   - **URL:** https://scikit-learn.org/stable/modules/impute.html
   - **O que aproveitamos (com nossas palavras):** Descreve por que muitos estimadores não aceitam valores ausentes (`NaN`) e apresenta estratégias de imputação (como `SimpleImputer`) para preencher lacunas em vez de descartar linhas inteiras — cuidado ainda mais crítico quando já existem poucos dados.

3. **scikit-learn API Reference: `sklearn.preprocessing.OneHotEncoder`**
   - **Autor / Órgão:** scikit-learn developers
   - **Versão / Data:** Documentação estável (conferida na versão `1.9.1`)
   - **URL:** https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
   - **O que aproveitamos (com nossas palavras):** Mostra como transformar colunas categóricas (texto/categorias) em colunas binárias numéricas sem criar uma falsa relação de ordem entre categorias, preparando os dados para os algoritmos supervisionados.

---

### Regra R3 (`classe = medio` E `poucos_dados = 0`)
**Recomendação:** *Imputar nulos; codificar categóricas e treinar com separação treino/teste.*

1. **scikit-learn API Reference: `sklearn.model_selection.train_test_split`**
   - **Autor / Órgão:** scikit-learn developers
   - **Versão / Data:** Documentação estável (conferida na versão `1.9.1`)
   - **URL:** https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
   - **O que aproveitamos (com nossas palavras):** Documenta a separação aleatória (e estratificada via `stratify`) dos dados em conjuntos de treino e teste. Quando o dataset tem volume suficiente (`poucos_dados = 0`), a separação treino/teste já permite avaliar o modelo em dados inéditos de forma rápida e direta após tratar nulos e categóricas.

2. **scikit-learn User Guide: Imputation of missing values** e **`OneHotEncoder`**
   - **Autor / Órgão:** scikit-learn developers
   - **URLs:** https://scikit-learn.org/stable/modules/impute.html e https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
   - **O que aproveitamos (com nossas palavras):** Os mesmos fundamentos da regra R2 para tratar valores ausentes e converter colunas categóricas antes do treinamento, garantindo que o modelo receba entradas limpas e numéricas.

---

### Regra R4 (`classe = alto` E `precisa_explicar = 1`)
**Recomendação:** *Usar árvore de decisão (modelo interpretável) e mostrar as condições aprendidas; imputar nulos só se a versão usada exigir.*

1. **scikit-learn User Guide: Decision Trees**
   - **Autor / Órgão:** scikit-learn developers
   - **Versão / Data:** Documentação estável (conferida na versão `1.9.1`)
   - **URL:** https://scikit-learn.org/stable/modules/tree.html
   - **O que aproveitamos (com nossas palavras):** Destaca que árvores de decisão são modelos "caixa branca" (*white box*): suas decisões podem ser inspecionadas e explicadas por meio de regras simples de corte (usando `export_text` ou `plot_tree`), o que atende diretamente ao requisito `precisa_explicar = 1`. Além disso, a documentação descreve o suporte nativo a valores ausentes em versões recentes do `DecisionTreeClassifier` (a partir do scikit-learn 1.3 para critérios compatíveis), motivando a observação prática de só imputar nulos se a versão da biblioteca em uso exigir.

---

### Regra R5 (`classe = alto` E `precisa_explicar = 0`)
**Recomendação:** *Testar modelos mais flexíveis e comparar com validação cruzada para vigiar overfitting.*

1. **An Introduction to Statistical Learning (ISL), Seção 2.1.3 — *The Trade-Off Between Prediction Accuracy and Model Interpretability***
   - **Autores / Editora:** Gareth James, Daniela Witten, Trevor Hastie e Robert Tibshirani (Springer)
   - **Versão / Data:** 1ª edição (2013; 2ª edição em 2021 e edição em Python — ISLP — em 2023)
   - **URL:** https://www.statlearning.com/
   - **O que aproveitamos (com nossas palavras):** A Seção 2.1.3 explica o compromisso (*trade-off*) entre flexibilidade/acurácia preditiva e interpretabilidade. Quando o objetivo principal é previsão e não é necessário explicar cada passo de forma simples (`precisa_explicar = 0`), faz sentido testar modelos mais flexíveis (como ensembles ou métodos não lineares). Ao mesmo tempo, o livro alerta que modelos mais flexíveis têm maior risco de *overfitting* (variância alta), exigindo comparação rigorosa por validação cruzada para escolher o modelo que realmente generaliza melhor.

---

## Referências Complementares (Arquitetura, Verificação e Ética)

Além das fontes diretamente ligadas às 5 regras, o projeto apoiou-se nas seguintes referências conceituais e técnicas:

- **CLIPS — A Tool for Building Expert Systems:** NASA Johnson Space Center (domínio público desde 1996). URL: https://www.clipsrules.net/ — Base conceitual para separar a base de conhecimento (`regras.csv`), a memória de trabalho (`memoria`), o motor de inferência (`motor.py`) e a estratégia de resolução de conflitos por especificidade.
- **Rules of Machine Learning (Regra #5 — *Test the infrastructure independently from the machine learning*):** Martin Zinkevich (Google). URL: https://developers.google.com/machine-learning/guides/rules-of-ml — Inspirou os testes unitários do motor de inferência em `motor.py` e `main.py`, bem como a auditoria exaustiva em `verificar_base.py`, sem depender da saída do classificador.
- **NIST AI Risk Management Framework (AI RMF 1.0, NIST AI 100-1):** National Institute of Standards and Technology (janeiro de 2023). URL: https://www.nist.gov/itl/ai-risk-management-framework — Referência para a discussão de transparência, explicabilidade, limitações de dados simulados e responsabilidade pelo uso educacional.
