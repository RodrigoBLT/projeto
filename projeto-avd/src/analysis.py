"""
Fase 3 — Análise Estatística Exploratória

Métricas calculadas:
  - Correlação de Pearson: Selic × retorno mensal do IBOVESPA
  - ANOVA entre regimes de Selic e retorno do IBOVESPA
  - Estatísticas descritivas: média, mediana, desvio padrão, assimetria, curtose
  - Análise de quartis e IQR
  - Detecção de outliers (regra de Tukey / Z-Score)
  - Análise de tendência linear do IBOVESPA
"""

import numpy as np
import pandas as pd
from scipy import stats


# ── Correlação de Pearson ─────────────────────────────────────────────────────

def pearson_selic_ibov(merged: pd.DataFrame) -> dict:
    """
    Calcula r de Pearson entre a taxa Selic anual e o retorno mensal do IBOVESPA.
    Interpreta o resultado e informa se é estatisticamente significativo (p < 0,05).
    """
    x = merged["selic_anual"].dropna()
    y = merged["ibov_retorno_mensal"].dropna()
    idx = x.index.intersection(y.index)

    r, p = stats.pearsonr(x[idx].values, y[idx].values)
    r, p = float(r), float(p)

    if r < -0.5:
        interp = "Correlação negativa forte: Selic alta → IBOVESPA tende a cair"
    elif r < -0.3:
        interp = "Correlação negativa moderada: Selic alta pressiona a bolsa"
    elif r > 0.5:
        interp = "Correlação positiva forte (incomum no Brasil)"
    elif r > 0.3:
        interp = "Correlação positiva moderada"
    else:
        interp = "Correlação fraca ou nula no período analisado"

    return {
        "r":             round(r, 4),
        "r2":            round(r ** 2, 4),
        "p_value":       round(p, 6),
        "significativo": p < 0.05,
        "interpretacao": interp,
        "n":             len(idx),
    }


def pearson_matrix(df: pd.DataFrame, colunas: list) -> pd.DataFrame:
    cols = [c for c in colunas if c in df.columns]
    return df[cols].corr(method="pearson").round(3)


def anova_retorno_por_regime_selic(merged: pd.DataFrame) -> dict:
    """
    Compara o retorno mensal do IBOVESPA entre tres regimes de Selic:
    baixa, moderada e alta.
    """
    df = merged[["selic_anual", "ibov_retorno_mensal"]].dropna().copy()
    if df.empty:
        return _anova_vazia()

    df["regime_selic"] = pd.cut(
        df["selic_anual"],
        bins=[0.0, 0.09, 0.12, 1.0],
        labels=["Baixa", "Moderada", "Alta"],
        include_lowest=True,
    )
    grupos = {
        regime: grupo["ibov_retorno_mensal"].values
        for regime, grupo in df.groupby("regime_selic", observed=False)
        if len(grupo) >= 2
    }
    resumo = (
        df.groupby("regime_selic", observed=False)["ibov_retorno_mensal"]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
        .rename(columns={
            "regime_selic": "Regime Selic",
            "count": "n",
            "mean": "media",
            "median": "mediana",
            "std": "desvio_padrao",
        })
    )

    if len(grupos) < 2:
        return {
            **_anova_vazia(),
            "resumo": resumo,
            "interpretacao": "Não há grupos suficientes para comparar os regimes de Selic.",
        }

    f_stat, p_val = stats.f_oneway(*grupos.values())
    medias = {regime: float(np.mean(valores)) for regime, valores in grupos.items()}
    melhor_regime = max(medias, key=medias.get)

    return {
        "f_stat": round(float(f_stat), 4),
        "p_value": round(float(p_val), 6),
        "significativo": p_val < 0.05,
        "grupos_validos": len(grupos),
        "melhor_regime": melhor_regime,
        "resumo": resumo.round(4),
        "interpretacao": (
            f"A ANOVA indica diferenca estatisticamente significativa entre regimes de Selic; "
            f"o maior retorno medio apareceu no regime {melhor_regime}."
            if p_val < 0.05 else
            "A ANOVA nao encontrou evidencia estatistica suficiente de diferenca entre os regimes de Selic."
        ),
    }


# ── Estatísticas Descritivas ──────────────────────────────────────────────────

def stats_descritivas(series: pd.Series) -> dict:
    s = series.dropna()
    return {
        "n":          int(len(s)),
        "media":      round(float(s.mean()), 4),
        "mediana":    round(float(s.median()), 4),
        "desvio_pad": round(float(s.std()), 4),
        "minimo":     round(float(s.min()), 4),
        "maximo":     round(float(s.max()), 4),
        "assimetria": round(float(s.skew()), 4),
        "curtose":    round(float(s.kurtosis()), 4),
    }


# ── Quartis ───────────────────────────────────────────────────────────────────

def analise_quartis(series: pd.Series) -> pd.DataFrame:
    q = series.quantile([0.0, 0.25, 0.5, 0.75, 1.0])
    iqr = float(q[0.75] - q[0.25])
    return pd.DataFrame({
        "Estatistica": ["Minimo", "Q1 (25%)", "Mediana", "Q3 (75%)", "Maximo", "IQR"],
        "Valor":       [
            round(q[0.00], 4),
            round(q[0.25], 4),
            round(q[0.50], 4),
            round(q[0.75], 4),
            round(q[1.00], 4),
            round(iqr,     4),
        ],
    })


# ── Outliers ──────────────────────────────────────────────────────────────────

def outliers_iqr(series: pd.Series) -> pd.Series:
    """Regra de Tukey: outlier se x < Q1 - 1,5xIQR ou x > Q3 + 1,5xIQR."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return (series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)


def outliers_zscore(series: pd.Series, limiar: float = 2.5) -> pd.Series:
    clean = series.dropna()
    z     = np.abs(stats.zscore(clean))
    mask  = pd.Series(False, index=series.index)
    mask.loc[clean.index] = z > limiar
    return mask


def resumo_outliers(merged: pd.DataFrame) -> pd.DataFrame:
    linhas = []
    for col, nome in [
        ("ibov_retorno_mensal", "Retorno Mensal IBOVESPA (%)"),
        ("selic_anual",         "Selic Anual"),
    ]:
        if col not in merged.columns:
            continue
        s   = merged[col].dropna()
        iqr = outliers_iqr(s)
        zsc = outliers_zscore(s)
        for dt in s[iqr | zsc].index:
            linhas.append({
                "Data":    dt.strftime("%m/%Y"),
                "Metrica": nome,
                "Valor":   round(s[dt], 4),
                "IQR":     "Sim" if iqr[dt] else "Nao",
                "Z-Score": "Sim" if zsc[dt] else "Nao",
            })
    return pd.DataFrame(linhas)


# ── Tendência Linear ──────────────────────────────────────────────────────────

def tendencia_ibovespa(ibov_metrics: pd.DataFrame) -> dict:
    """
    Regressão linear simples sobre o nível do IBOVESPA ao longo do tempo.
    Indica se a tendência é de alta ou queda no período analisado.
    """
    s = ibov_metrics["ibovespa"].dropna()
    x = np.arange(len(s))
    slope, intercept, r, p, _ = stats.linregress(x, s.values)
    return {
        "inclinacao":    round(float(slope), 2),
        "r2":            round(float(r ** 2), 4),
        "p_value":       round(float(p), 6),
        "tendencia":     "Alta" if slope > 0 else "Queda",
        "significativa": p < 0.05,
    }


def _anova_vazia() -> dict:
    return {
        "f_stat": float("nan"),
        "p_value": float("nan"),
        "significativo": False,
        "grupos_validos": 0,
        "melhor_regime": "indefinido",
        "resumo": pd.DataFrame(),
        "interpretacao": "Dados insuficientes para executar a ANOVA.",
    }
