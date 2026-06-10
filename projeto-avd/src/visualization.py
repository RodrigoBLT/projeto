"""
Fase 4 — Visualização Gestalt

Princípios aplicados em cada gráfico:
  Proximidade   → KPIs e gráficos relacionados agrupados no mesmo cartão
  Similaridade  → Cor fixa por estratégia em todos os gráficos
  Continuidade  → Séries temporais com linhas contínuas (sem marcadores isolados)
  Figura/Fundo  → Fundo escuro (#0e1117), elementos de destaque em cores vivas
  Pregnância    → Layouts enxutos, hierarquia tipográfica clara, sem ruído visual
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

CORES = {
    "ibovespa":   "#2196F3",
    "selic":      "#FF9800",
    "poupanca":   "#66BB6A",
    "renda_fixa": "#42A5F5",
    "renda_var":  "#AB47BC",
    "positivo":   "#00e676",
    "negativo":   "#ff5252",
    "neutro":     "#64b5f6",
    "grade":      "#1e2533",
    "fundo":      "#0e1117",
    "texto":      "#e0e0e0",
}

_BASE = dict(
    paper_bgcolor=CORES["fundo"],
    plot_bgcolor=CORES["fundo"],
    font=dict(color=CORES["texto"], family="Inter, sans-serif"),
    margin=dict(l=44, r=20, t=50, b=40),
    legend=dict(bgcolor="rgba(30,35,50,0.85)", bordercolor="#2d3748", borderwidth=1),
)

_AXIS = dict(gridcolor=CORES["grade"], zeroline=False)


def _titulo(texto: str) -> dict:
    return dict(text=texto, font=dict(size=15, color="#90caf9"), x=0)


# ── Gráfico 1: IBOVESPA × Selic ──────────────────────────────────────────────

def grafico_selic_ibov(ibov_metrics: pd.DataFrame, selic: pd.DataFrame) -> go.Figure:
    """
    Gestalt: Continuidade + Similaridade
    Eixo duplo: IBOVESPA (esq.) × Selic anual em % (dir.).
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=ibov_metrics.index, y=ibov_metrics["ibovespa"],
        fill="tozeroy", fillcolor="rgba(33,150,243,0.07)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=ibov_metrics.index, y=ibov_metrics["ibovespa"],
        name="IBOVESPA", mode="lines",
        line=dict(color=CORES["ibovespa"], width=2),
        hovertemplate="<b>IBOVESPA</b><br>%{x|%d/%m/%Y}<br>%{y:,.0f} pts<extra></extra>",
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=selic.index, y=selic["selic_anual"] * 100,
        name="Selic (% a.a.)", mode="lines",
        line=dict(color=CORES["selic"], width=2.5, dash="dot"),
        hovertemplate="<b>Selic</b><br>%{x|%m/%Y}<br>%{y:.2f}% a.a.<extra></extra>",
    ), secondary_y=True)

    fig.update_layout(
        **_BASE, height=400,
        title=_titulo("IBOVESPA x Taxa Selic — Últimos 12 Meses"),
        hovermode="x unified",
        xaxis=dict(**_AXIS),
        yaxis=dict(**_AXIS, title="IBOVESPA (pontos)"),
        yaxis2=dict(**_AXIS, title="Selic (% a.a.)", showgrid=False),
    )
    return fig


# ── Gráfico 2: Distribuição de Retornos ──────────────────────────────────────

def grafico_distribuicao_retornos(ibov_metrics: pd.DataFrame) -> go.Figure:
    """Gestalt: Figura/Fundo — Histograma + Box Plot lado a lado."""
    retornos = ibov_metrics["retorno_diario"].dropna() * 100

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Histograma de Retornos Diários (%)", "Box Plot — Dispersão"],
        column_widths=[0.65, 0.35],
    )

    fig.add_trace(go.Histogram(
        x=retornos, nbinsx=45,
        marker_color=CORES["ibovespa"], opacity=0.82,
        name="Frequência",
        hovertemplate="Retorno: %{x:.2f}%<br>Freq: %{y}<extra></extra>",
    ), row=1, col=1)

    media = retornos.mean()
    fig.add_vline(x=media, line_dash="dash", line_color=CORES["selic"],
                  annotation_text=f" u = {media:.3f}%",
                  annotation_font_color=CORES["selic"], row=1, col=1)
    fig.add_vline(x=0, line_dash="dot", line_color="#555", row=1, col=1)

    fig.add_trace(go.Box(
        y=retornos, marker_color=CORES["ibovespa"],
        boxmean="sd", name="Retorno (%)", hoverinfo="y",
    ), row=1, col=2)

    fig.update_layout(**_BASE, height=360,
                      title=_titulo("Distribuição de Retornos Diários — IBOVESPA"),
                      showlegend=False)
    fig.update_xaxes(**_AXIS)
    fig.update_yaxes(**_AXIS)
    return fig


# ── Gráfico 3: Comparativo de Cenários ───────────────────────────────────────

def grafico_comparativo_cenarios(sim: dict, valor: float) -> go.Figure:
    """
    Gestalt: Proximidade + Similaridade
    Barras por estratégia com destaque para a recomendada.
    """
    estrategias = list(sim["montantes"].keys())
    montantes   = list(sim["montantes"].values())
    retornos    = list(sim["retornos"].values())
    cores       = [CORES["poupanca"], CORES["renda_fixa"], CORES["renda_var"]]

    fig = go.Figure(go.Bar(
        x=estrategias, y=montantes,
        marker_color=cores,
        text=[f"R$ {m:,.0f}\n+R$ {r:,.0f}" for m, r in zip(montantes, retornos)],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Montante: R$ %{y:,.2f}<extra></extra>",
    ))

    fig.add_hline(y=valor, line_dash="dot", line_color="#555",
                  annotation_text=f"Capital inicial: R$ {valor:,.0f}",
                  annotation_position="bottom right",
                  annotation_font_color="#8892a4")

    rec = sim.get("recomendacao", "")
    if rec in estrategias:
        idx = estrategias.index(rec)
        fig.add_annotation(
            x=idx, y=montantes[idx],
            text="Recomendado",
            showarrow=True, arrowhead=2,
            arrowcolor=cores[idx],
            font=dict(color=cores[idx], size=12),
            yshift=30,
        )

    fig.update_layout(
        **_BASE, height=400,
        title=_titulo("Comparativo de Cenários — Montante Final"),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(**_AXIS, title="Montante Final (R$)"),
        showlegend=False,
    )
    return fig


# ── Gráfico 4: Heatmap de Correlação ─────────────────────────────────────────

def grafico_heatmap_correlacao(merged: pd.DataFrame) -> go.Figure:
    """Gestalt: Pregnância — Matriz de Pearson com anotações."""
    mapa_nomes = {
        "selic_anual":         "Selic a.a.",
        "ibov_retorno_mensal": "IBOV Ret. Mensal %",
        "selic_delta":         "Delta Selic",
        "ibov_lag1":           "IBOV Lag 1 mes",
        "poupanca_mensal":     "Poupanca Mensal %",
        "rf_mensal":           "Renda Fixa Mensal %",
    }
    cols   = [c for c in mapa_nomes if c in merged.columns]
    labels = [mapa_nomes[c] for c in cols]
    corr   = merged[cols].corr(method="pearson").round(2)

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=labels, y=labels,
        colorscale="RdBu_r",
        zmin=-1, zmax=1,
        text=corr.values,
        texttemplate="%{text:.2f}",
        hovertemplate="<b>%{y} x %{x}</b><br>r = %{z:.3f}<extra></extra>",
        colorbar=dict(title="r", tickfont=dict(color=CORES["texto"])),
    ))

    fig.update_layout(
        **_BASE, height=400,
        title=_titulo("Heatmap de Correlação de Pearson"),
        xaxis=dict(tickfont=dict(size=11), tickangle=-30),
        yaxis=dict(tickfont=dict(size=11)),
    )
    return fig


# ── Gráfico 5: IBOVESPA + Médias Móveis ──────────────────────────────────────

def grafico_medias_moveis(ibov_metrics: pd.DataFrame) -> go.Figure:
    """Gestalt: Continuidade — IBOVESPA com MA20 e MA50."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ibov_metrics.index, y=ibov_metrics["ibovespa"],
        name="IBOVESPA", mode="lines",
        line=dict(color=CORES["ibovespa"], width=1.5), opacity=0.85,
    ))
    fig.add_trace(go.Scatter(
        x=ibov_metrics.index, y=ibov_metrics["ma20"],
        name="MA20", mode="lines",
        line=dict(color=CORES["positivo"], width=1.8, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=ibov_metrics.index, y=ibov_metrics["ma50"],
        name="MA50", mode="lines",
        line=dict(color=CORES["selic"], width=1.8, dash="dot"),
    ))
    fig.update_layout(
        **_BASE, height=360,
        title=_titulo("IBOVESPA com Médias Móveis — MA20 e MA50"),
        xaxis=dict(**_AXIS),
        yaxis=dict(**_AXIS, title="Pontos"),
        hovermode="x unified",
    )
    return fig


# ── Gráfico 6: Série de Crescimento ──────────────────────────────────────────

def grafico_serie_crescimento(serie: pd.DataFrame) -> go.Figure:
    """Gestalt: Continuidade + Similaridade — evolução mensal do capital."""
    fig = go.Figure()
    mapa_cores = {
        "Poupanca":       CORES["poupanca"],
        "Renda Fixa":     CORES["renda_fixa"],
        "Renda Variavel": CORES["renda_var"],
    }
    for col, cor in mapa_cores.items():
        if col not in serie.columns:
            continue
        fig.add_trace(go.Scatter(
            x=serie["mes"], y=serie[col],
            name=col, mode="lines",
            line=dict(color=cor, width=2),
            hovertemplate=f"<b>{col}</b><br>Mes %{{x}}<br>R$ %{{y:,.2f}}<extra></extra>",
        ))
    fig.update_layout(
        **_BASE, height=340,
        title=_titulo("Evolução do Capital ao Longo do Tempo"),
        xaxis=dict(**_AXIS, title="Meses"),
        yaxis=dict(**_AXIS, title="Montante (R$)"),
        hovermode="x unified",
    )
    return fig


# ── Gráfico 7: Scatter Selic × Retorno IBOVESPA ──────────────────────────────

def grafico_scatter_selic_ibov(merged: pd.DataFrame) -> go.Figure:
    """
    Gestalt: Figura/Fundo — Dispersão com linha de tendência OLS.
    Mostra visualmente a correlação entre Selic e retorno do IBOVESPA.
    """
    x = merged["selic_anual"].dropna() * 100
    y = merged["ibov_retorno_mensal"].dropna()
    idx = x.index.intersection(y.index)
    xv, yv = x[idx].values, y[idx].values

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xv, y=yv, mode="markers",
        marker=dict(color=CORES["ibovespa"], size=9,
                    line=dict(color="#fff", width=0.5), opacity=0.85),
        hovertemplate="Selic: %{x:.2f}%<br>IBOV Ret.: %{y:.2f}%<extra></extra>",
        name="Observacoes",
    ))

    if len(xv) >= 4:
        m, b = np.polyfit(xv, yv, 1)
        xs = np.linspace(xv.min(), xv.max(), 60)
        fig.add_trace(go.Scatter(
            x=xs, y=m * xs + b,
            mode="lines",
            line=dict(color=CORES["selic"], width=2, dash="dash"),
            name=f"Tendencia (b={m:.2f})",
            hoverinfo="skip",
        ))

    fig.add_hline(y=0, line_dash="dot", line_color="#444")
    fig.update_layout(
        **_BASE, height=340,
        title=_titulo("Selic (% a.a.) x Retorno Mensal do IBOVESPA (%)"),
        xaxis=dict(**_AXIS, title="Selic (% a.a.)"),
        yaxis=dict(**_AXIS, title="Retorno Mensal IBOV (%)"),
    )
    return fig
