"""
Fase 4 (ML) — Modelo de Recomendação de Investimento

Modelo:
  Ridge Regression com features da Selic para prever o
  retorno esperado do IBOVESPA no mês seguinte.

  Features:  selic_anual, selic_lag1, selic_delta, ibov_lag1
  Target:    ibov_retorno_mensal (%)

Também embute regras de negócio derivadas da análise histórica (1999-2024)
para gerar uma recomendação entre três estratégias:
  Poupança | Renda Fixa (CDB/Tesouro) | Renda Variável (IBOVESPA)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error


# ── Treinamento ───────────────────────────────────────────────────────────────

FEATURES = ["selic_anual", "selic_lag1", "selic_delta", "ibov_lag1"]
TARGET   = "ibov_retorno_mensal"


def treinar_modelo(ml_features: pd.DataFrame) -> dict:
    """
    Treina Ridge Regression com split temporal 80/20 (sem data leakage).
    Retorna modelo, métricas, predições e DataFrame de teste.
    """
    df = ml_features[[*FEATURES, TARGET]].dropna()

    if len(df) < 8:
        return _modelo_fallback(df)

    split = max(int(len(df) * 0.80), len(df) - 6)
    X_tr = df[FEATURES].values[:split]
    y_tr = df[TARGET].values[:split]
    X_te = df[FEATURES].values[split:]
    y_te = df[TARGET].values[split:]

    modelo = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge",  Ridge(alpha=1.0)),
    ])
    modelo.fit(X_tr, y_tr)

    if len(X_te) > 0:
        preds = modelo.predict(X_te)
        r2  = round(float(r2_score(y_te, preds)), 4)
        mae = round(float(mean_absolute_error(y_te, preds)), 4)
    else:
        preds = np.array([])
        r2, mae = float("nan"), float("nan")

    test_df = df.iloc[split:].copy()
    if len(preds) > 0:
        test_df["pred"] = preds
        test_df["erro"] = test_df[TARGET] - test_df["pred"]

    coef = dict(zip(FEATURES, modelo.named_steps["ridge"].coef_))

    return {
        "modelo":       modelo,
        "r2":           r2,
        "mae":          mae,
        "coeficientes": {k: round(float(v), 4) for k, v in coef.items()},
        "df_treino":    df.iloc[:split],
        "df_teste":     test_df,
        "n_treino":     split,
        "n_teste":      len(df) - split,
        "features":     FEATURES,
    }


def _modelo_fallback(df: pd.DataFrame) -> dict:
    media = float(df[TARGET].mean()) if not df.empty else 0.8
    return {
        "modelo":       None,
        "r2":           float("nan"),
        "mae":          float("nan"),
        "coeficientes": {},
        "df_treino":    df,
        "df_teste":     pd.DataFrame(),
        "n_treino":     len(df),
        "n_teste":      0,
        "_media":       media,
        "features":     FEATURES,
    }


# ── Predição Pontual ──────────────────────────────────────────────────────────

def prever_retorno_ibov(resultado_ml: dict, selic_atual: float,
                        ibov_lag1: float = 0.8) -> float:
    if resultado_ml.get("modelo") is None:
        return resultado_ml.get("_media", 0.8)

    selic_lag1  = resultado_ml["df_treino"]["selic_anual"].iloc[-1] \
                  if not resultado_ml["df_treino"].empty else selic_atual
    selic_delta = selic_atual - selic_lag1

    X = np.array([[selic_atual, selic_lag1, selic_delta, ibov_lag1]])
    return float(resultado_ml["modelo"].predict(X)[0])


# ── Simulador de Investimento ─────────────────────────────────────────────────

def simular_investimento(
    selic_anual:      float,
    valor:            float,
    periodo_meses:    int,
    tolerancia:       str,
    ibov_pred_mensal=None,
) -> dict:
    """
    Simula o crescimento de capital nas três estratégias.

    Poupança:   70% da Selic quando Selic >= 8,5% a.a.; c/c 0,5% a.m.
    Renda Fixa: 97% do CDI (liquido IR para prazo >= 12 meses)
    Renda Var:  Retorno IBOVESPA ajustado por tolerancia ao risco
    """
    if selic_anual >= 0.085:
        tx_poupanca = ((1 + selic_anual) ** (1 / 12) - 1) * 0.70
    else:
        tx_poupanca = 0.005

    tx_rf = ((1 + selic_anual) ** (1 / 12) - 1) * 0.97

    if ibov_pred_mensal is None:
        ibov_pred_mensal_decimal = _ibov_esperado_mensal(selic_anual)
    else:
        ibov_pred_mensal_decimal = ibov_pred_mensal / 100

    ajuste  = {"baixa": 0.7, "media": 1.0, "alta": 1.3}.get(tolerancia, 1.0)
    tx_ibov = ibov_pred_mensal_decimal * ajuste

    def _montante(tx: float) -> float:
        return valor * (1 + tx) ** periodo_meses

    m_poup = _montante(tx_poupanca)
    m_rf   = _montante(tx_rf)
    m_ibov = _montante(tx_ibov)

    retornos = {
        "Poupanca":       round(m_poup - valor, 2),
        "Renda Fixa":     round(m_rf   - valor, 2),
        "Renda Variavel": round(m_ibov - valor, 2),
    }
    montantes = {
        "Poupanca":       round(m_poup, 2),
        "Renda Fixa":     round(m_rf,   2),
        "Renda Variavel": round(m_ibov, 2),
    }
    taxas_mensais = {
        "Poupanca":       round(tx_poupanca * 100, 4),
        "Renda Fixa":     round(tx_rf       * 100, 4),
        "Renda Variavel": round(tx_ibov     * 100, 4),
    }

    if tolerancia == "baixa":
        rec = "Renda Fixa" if m_rf >= m_poup else "Poupanca"
    elif tolerancia == "media":
        rec = max(retornos, key=retornos.get)
        if rec == "Renda Variavel" and selic_anual >= 0.13:
            rec = "Renda Fixa"
    else:
        rec = max(retornos, key=retornos.get)

    confianca = _calcular_confianca(selic_anual, tolerancia, rec)

    return {
        "retornos":      retornos,
        "montantes":     montantes,
        "taxas_mensais": taxas_mensais,
        "recomendacao":  rec,
        "confianca":     confianca,
        "justificativa": _justificar(selic_anual, rec, tolerancia, periodo_meses),
        "serie_temporal": _serie_crescimento(valor, tx_poupanca, tx_rf, tx_ibov, periodo_meses),
    }


def _ibov_esperado_mensal(selic_anual: float) -> float:
    if selic_anual >= 0.13:
        return 0.006
    elif selic_anual >= 0.10:
        return 0.009
    else:
        return 0.012


def _calcular_confianca(selic: float, tolerancia: str, rec: str) -> str:
    if rec in ("Poupanca", "Renda Fixa"):
        return "Alta (90%)" if tolerancia == "baixa" else "Moderada (75%)"
    if selic >= 0.13:
        return "Baixa (45%)"
    return "Moderada (65%)" if tolerancia == "alta" else "Baixa (50%)"


def _justificar(selic: float, rec: str, tolerancia: str, periodo: int) -> str:
    sp = round(selic * 100, 2)
    if rec == "Poupanca":
        return (
            f"Com Selic em {sp}% a.a. e perfil conservador, a Poupanca garante "
            "liquidez diaria e rendimento isento de IR — ideal para reserva de emergencia."
        )
    elif rec == "Renda Fixa":
        return (
            f"Com Selic em {sp}% a.a., o Tesouro Selic ou CDB pos-fixado remunera "
            f"proximo de {sp}% a.a. com baixo risco e liquidez em {periodo} meses. "
            "O custo de oportunidade de assumir risco da bolsa e elevado nesse cenario."
        )
    else:
        return (
            f"Mesmo com Selic em {sp}% a.a., o IBOVESPA historicamente supera a renda fixa "
            f"em horizontes acima de 24 meses para perfil {tolerancia}. "
            "Diversificacao parcial em renda variavel pode aumentar o retorno ajustado ao risco."
        )


def _serie_crescimento(
    valor: float,
    tx_poup: float,
    tx_rf: float,
    tx_ibov: float,
    meses: int,
) -> pd.DataFrame:
    t = np.arange(meses + 1)
    return pd.DataFrame({
        "mes":            t,
        "Poupanca":       valor * (1 + tx_poup) ** t,
        "Renda Fixa":     valor * (1 + tx_rf)   ** t,
        "Renda Variavel": valor * (1 + tx_ibov) ** t,
    })
