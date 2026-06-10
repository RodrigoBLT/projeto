# AVD — Decisão de Investimento com Python

> Projeto de Análise e Visualização de Dados (AVD) — disciplina de Python  
> **Storytelling:** _Como a taxa Selic impacta a decisão entre Poupança, Renda Fixa e Renda Variável?_

---

## Estrutura do Projeto

```
projeto-avd/
├── Dockerfile              # Containeriza a aplicação Streamlit
├── docker-compose.yml      # Orquestração com volume para dados
├── requirements.txt        # Dependências Python
├── main.py                 # Fase 5: Dashboard Streamlit (entry point)
├── src/
│   ├── data_extraction.py  # Fase 1: Extração (yfinance + API BCB)
│   ├── etl.py              # Fase 2: ETL & Data Wrangling
│   ├── analysis.py         # Fase 3: Análise Estatística
│   ├── ml_model.py         # Fase 4 (ML): Regressão + Simulador
│   └── visualization.py   # Fase 4 (Gestalt): 7 gráficos Plotly
└── data/
    ├── ibovespa.csv        # Série histórica diária do IBOVESPA
    ├── selic.csv           # Taxa Selic mensal (BCB)
    ├── cache.csv           # Dataset consolidado (gerado pelo ETL)
    └── processed/          # Arquivos intermediários do pipeline
```

---

## Como Executar com Docker

**Pré-requisito:** Docker e Docker Compose instalados.

```bash
# 1. Acesse o diretório do projeto
cd projeto-avd

# 2. Suba o container (build automático na 1ª execução)
docker-compose up

# 3. Abra no navegador
http://localhost:8501
```

Para parar: `docker-compose down`  
Para rebuild após atualizar dependências: `docker-compose up --build`

---

## Como Executar Localmente (sem Docker)

```bash
# Pré-requisito: Python 3.11+
pip install -r requirements.txt
streamlit run main.py
```

---

## Storytelling: "Simulação de Decisão Financeira"

### O Problema
Todo investidor brasileiro enfrenta a mesma escolha: deixar na **Poupança**,
aplicar em **Renda Fixa** (CDB/Tesouro Direto) ou arriscar na **Renda Variável** via IBOVESPA.

### A Variável-Chave: Taxa Selic
A Selic é a âncora de todo o sistema financeiro brasileiro:

| Selic Alta (≥ 13%) | Selic Moderada (9–13%) | Selic Baixa (< 9%) |
|---|---|---|
| Renda fixa muito atrativa | Equilíbrio entre estratégias | Incentivo à bolsa |
| IBOVESPA sob pressão | Diversificação recomendada | IBOVESPA historicamente melhor |

### Fluxo do Dashboard

1. **Contexto** → Selic atual, IBOVESPA × Selic nos últimos 12 meses, scatter correlação
2. **Simulador** → Usuário insere valor/prazo/risco → vê retornos e recomendação
3. **Análise Estatística** → Pearson, distribuição, quartis, outliers, heatmap, MAs
4. **ML Recomendação** → Treina Ridge Regression → prediz retorno IBOVESPA → recomendação

---

## Modelo de Machine Learning

**Algoritmo:** Ridge Regression (regressão linear com regularização L2)

| Feature | Descrição |
|---|---|
| `selic_anual` | Taxa Selic anualizada no mês |
| `selic_lag1` | Selic do mês anterior |
| `selic_delta` | Variação da Selic |
| `ibov_lag1` | Retorno IBOVESPA do mês anterior |

**Target:** Retorno mensal do IBOVESPA (%)  
**Validação:** Split temporal 80/20 — sem data leakage

---

## Fontes de Dados

| Fonte | Dado | Método |
|---|---|---|
| **Yahoo Finance** | Histórico IBOVESPA (^BVSP) | `yfinance` |
| **Banco Central do Brasil** | Meta Selic (SGS série 11) | REST API |

> Dados cacheados em `data/` para evitar sobrecarga nas APIs.  
> Botão **🔄 Atualizar Dados** na sidebar força re-extração.

---

## As 5 Fases

| Fase | Arquivo | Conteúdo |
|---|---|---|
| 1 — Extração | `src/data_extraction.py` | yfinance + API BCB + fallback sintético |
| 2 — ETL | `src/etl.py` | Limpeza, MA20/MA50, volatilidade, merge mensal |
| 3 — Análise | `src/analysis.py` | Pearson, quartis, outliers IQR/Z-score, tendência |
| 4 — Gestalt | `src/visualization.py` | 7 gráficos Plotly com princípios Gestalt |
| 5 — Dashboard | `main.py` | 4 abas interativas, KPI cards, simulador, ML |

---

## Dependências

```
streamlit   pandas   numpy   requests
yfinance    plotly   scikit-learn   scipy
```
