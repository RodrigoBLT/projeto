"""
Fase 2 — ETL & Data Wrangling

Transformações:
  - Remoção de NaNs e normalização de datas
  - Cálculo de retornos diários, retorno acumulado, MA20, MA50, volatilidade
  - Reagrupamento mensal do IBOVESPA para alinhamento com Selic
  - Merge IBOVESPA × Selic → dataset consolidado
  - Engenharia de features (lags, delta) para o modelo ML

Trabalha inteiramente em memória — sem leitura ou escrita de CSV.
"""

import numpy as np
import pandas as pd


# ── Métricas IBOVESPA ─────────────────────────────────────────────────────────

def compute_ibovespa_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula indicadores técnicos a partir dos preços diários.
    MA20/MA50: médias móveis para identificar tendência.
    Volatilidade: desvio padrão anualizado dos retornos (janela 21d).
    """
    d = df[["ibovespa"]].copy().sort_index()
    d["retorno_diario"]  = d["ibovespa"].pct_change()
    d["retorno_acum"]    = (1 + d["retorno_diario"]).cumprod() - 1
    d["ma20"]            = d["ibovespa"].rolling(20).mean()
    d["ma50"]            = d["ibovespa"].rolling(50).mean()
    d["volatilidade"]    = d["retorno_diario"].rolling(21).std() * np.sqrt(252) * 100
    return d.dropna(subset=["retorno_diario"])


# ── Merge Mensal ──────────────────────────────────────────────────────────────

def merge_monthly(ibov: pd.DataFrame, selic: pd.DataFrame) -> pd.DataFrame:
    """
    Reagrupa IBOVESPA para frequência mensal (último pregão)
    e faz merge com Selic pelo índice temporal.
    """
    ibov_m = ibov["ibovespa"].resample("ME").last()
    ret_m  = ibov_m.pct_change().rename("ibov_retorno_mensal") * 100

    merged = pd.concat([ibov_m.rename("ibovespa_mensal"), ret_m, selic], axis=1)
    merged = merged.dropna()
    merged["ibov_retorno_anual"] = ((1 + merged["ibov_retorno_mensal"] / 100) ** 12 - 1) * 100

    merged["poupanca_mensal"] = merged["selic_anual"].apply(
        lambda r: 0.005 if r < 0.085 else ((1 + r) ** (1 / 12) - 1) * 0.70
    ) * 100

    merged["rf_mensal"] = merged["selic_mensal"] * 0.97 * 100

    return merged


# ── Features para ML ─────────────────────────────────────────────────────────

def build_ml_features(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Cria variáveis defasadas (lags) e delta da Selic como features
    para o modelo de regressão que prevê retorno do IBOVESPA.
    """
    d = merged.copy()
    d["selic_lag1"]  = d["selic_anual"].shift(1)
    d["selic_lag2"]  = d["selic_anual"].shift(2)
    d["selic_delta"] = d["selic_anual"].diff()
    d["ibov_lag1"]   = d["ibov_retorno_mensal"].shift(1)
    d["ibov_lag2"]   = d["ibov_retorno_mensal"].shift(2)
    d["selic_nivel"] = pd.cut(
        d["selic_anual"],
        bins=[0, 0.09, 0.12, 1.0],
        labels=[0, 1, 2],
    ).astype(float)
    return d.dropna()


# ── Pipeline Principal ────────────────────────────────────────────────────────

def run(ibov: pd.DataFrame, selic: pd.DataFrame) -> dict:
    """
    Executa o pipeline ETL completo sobre DataFrames já extraídos.
    Retorna todos os datasets processados sem gravar arquivos.
    """
    ibov_metrics = compute_ibovespa_metrics(ibov)
    merged       = merge_monthly(ibov, selic)
    ml_features  = build_ml_features(merged)
    return {
        "ibov":         ibov,
        "ibov_metrics": ibov_metrics,
        "selic":        selic,
        "merged":       merged,
        "ml_features":  ml_features,
    }
