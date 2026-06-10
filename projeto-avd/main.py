"""
Fase 5 — Dashboard Interativo (Streamlit)

Storytelling: "Como a taxa Selic impacta sua decisão de investimento?"
Narrativa: Poupança × Renda Fixa × Renda Variável (IBOVESPA)

Abas:
  Contexto        — Apresentação, Selic atual, IBOVESPA × Selic
  Simulador       — Simulação personalizada de investimento
  Analise         — Análise estatística completa (Fase 3)
  ML Recomendacao — Modelo preditivo e recomendação inteligente
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(__file__))
from src import data_extraction, etl, analysis, ml_model, visualization as viz

# ── Configuração da Página ────────────────────────────────────────────────────

st.set_page_config(
    page_title="AVD — Decisão de Investimento",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS Personalizado ─────────────────────────────────────────────────────────

st.markdown("""
<style>
  .stApp { background-color: #0e1117; }
  .kpi { background: linear-gradient(135deg,#1a1f2e,#252b3b);
         border: 1px solid #2d3748; border-radius: 12px;
         padding: 18px 22px; text-align: center; }
  .kpi-label { font-size: 11px; color: #8892a4; text-transform: uppercase;
               letter-spacing: 1.2px; margin-bottom: 4px; }
  .kpi-value { font-size: 30px; font-weight: 700; margin: 2px 0; }
  .kpi-sub   { font-size: 12px; color: #8892a4; }
  .pos  { color: #00e676; }
  .neg  { color: #ff5252; }
  .neu  { color: #64b5f6; }
  .warn { color: #FF9800; }
  .sec { font-size: 14px; font-weight: 600; color: #90caf9;
         border-left: 3px solid #1565c0; padding-left: 10px;
         margin: 16px 0 10px; }
  .rec-card { background: linear-gradient(135deg,#0d2137,#0a1628);
              border: 2px solid #1565c0; border-radius: 14px;
              padding: 22px 26px; margin-top: 12px; }
  .rec-title { font-size: 20px; font-weight: 700; color: #90caf9; }
  .rec-text  { font-size: 14px; color: #c8d6e5; margin-top: 8px; line-height: 1.6; }
  .rec-conf  { display: inline-block; background: #0d47a1; color: #90caf9;
               padding: 3px 12px; border-radius: 20px; font-size: 12px; margin-top: 10px; }
  .badge-pos { background:#1b5e20; color:#a5d6a7; padding:3px 10px;
               border-radius:20px; font-size:12px; }
  .badge-neg { background:#b71c1c; color:#ef9a9a; padding:3px 10px;
               border-radius:20px; font-size:12px; }
  .badge-neu { background:#1a237e; color:#90caf9; padding:3px 10px;
               border-radius:20px; font-size:12px; }
  [data-testid="stSidebar"] { background-color: #131720; border-right: 1px solid #1e2533; }
  .stTabs [data-baseweb="tab"] { font-size: 14px; font-weight: 500; }
  .stTabs [aria-selected="true"] { color:#64b5f6 !important; border-bottom-color:#1565c0 !important; }
  .narrative { background: #131720; border-left: 4px solid #1565c0;
               border-radius: 0 8px 8px 0; padding: 16px 20px;
               font-size: 14px; color: #c8d6e5; line-height: 1.7; margin: 12px 0; }
</style>
""", unsafe_allow_html=True)


# ── Carregamento de Dados ─────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def carregar_dados() -> dict:
    ibov  = data_extraction.fetch_ibovespa("2y")
    selic = data_extraction.fetch_selic(26)
    resultado = etl.run(ibov, selic)
    return {
        "ibov_metrics": resultado["ibov_metrics"],
        "merged":       resultado["merged"],
        "ml_features":  resultado["ml_features"],
    }


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 💹 AVD Investimentos")
    st.markdown("*Selic x IBOVESPA — Análise Comparativa*")
    st.divider()

    if st.button("🔄 Atualizar Dados", use_container_width=True, type="primary"):
        st.cache_data.clear()
        st.rerun()

    st.divider()

    with st.spinner("Carregando pipeline..."):
        dados = carregar_dados()

    ibov_m = dados["ibov_metrics"]
    merged = dados["merged"]
    ml_f   = dados["ml_features"]

    selic_atual = float(merged["selic_anual"].iloc[-1]) if not merged.empty else 0.1375
    # Garante que o valor esteja em % a.a. (1–25), não em decimal (0.01–0.25)
    selic_pct_default = max(1.0, min(25.0, round(selic_atual * 100, 2)))

    st.markdown("### Parâmetros Globais")
    selic_override = st.number_input(
        "Selic atual (% a.a.)", min_value=1.0, max_value=25.0,
        value=selic_pct_default, step=0.25,
        help="Ajuste para testar cenários hipotéticos",
    ) / 100

    st.divider()
    st.markdown("**Sobre**")
    st.caption(
        "Projeto AVD — Python: análise comparativa de estratégias de "
        "alocação de capital com dados reais do IBOVESPA e da Selic (BCB)."
    )
    st.caption("Fases: Extração → ETL → Análise → Gestalt → Streamlit")


# ── Header ────────────────────────────────────────────────────────────────────

st.markdown(
    "<h1 style='color:#e0e0e0; margin-bottom:0'>💹 Decisão de Investimento</h1>"
    "<p style='color:#8892a4; font-size:15px'>"
    "Como a taxa Selic impacta a escolha entre Poupança, Renda Fixa e Renda Variável?</p>",
    unsafe_allow_html=True,
)
st.divider()


# ── KPIs Globais ──────────────────────────────────────────────────────────────

def _kpi(col, label, value, sub, cls="neu"):
    col.markdown(
        f"<div class='kpi'>"
        f"<div class='kpi-label'>{label}</div>"
        f"<div class='kpi-value {cls}'>{value}</div>"
        f"<div class='kpi-sub'>{sub}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


if not merged.empty and not ibov_m.empty:
    ibov_atual   = float(ibov_m["ibovespa"].iloc[-1])
    ibov_inicio  = float(ibov_m["ibovespa"].iloc[0])
    retorno_12m  = (ibov_atual / ibov_inicio - 1) * 100
    vol_atual    = float(ibov_m["volatilidade"].dropna().iloc[-1]) if "volatilidade" in ibov_m.columns else 0.0
    poup_aa      = ((1 + selic_override * 0.70) ** 12 - 1) * 100 if selic_override >= 0.085 else 6.17

    k1, k2, k3, k4, k5 = st.columns(5)
    _kpi(k1, "Selic Atual",         f"{selic_override*100:.2f}%",    "a.a.",                              "warn")
    _kpi(k2, "IBOVESPA",            f"{ibov_atual:,.0f}",             "pontos",                            "neu")
    _kpi(k3, "Retorno 12m (IBOV)",  f"{retorno_12m:+.1f}%",          "base no 1o dia do período",         "pos" if retorno_12m >= 0 else "neg")
    _kpi(k4, "Volatilidade",        f"{vol_atual:.1f}%",              "anualizada — janela 21d",           "neu")
    _kpi(k5, "Poupança a.a.",       f"{poup_aa:.2f}%",               "rendimento atual",                  "pos")
    st.markdown("")

st.divider()


# ── Abas ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "📖  Contexto",
    "💰  Simulador",
    "📊  Análise Estatística",
    "🤖  ML Recomendação",
])


# ═══════════════════════════════════════════════════════
#  TAB 1 — CONTEXTO
# ═══════════════════════════════════════════════════════

with tab1:
    st.markdown("""
    <div class='narrative'>
    <strong style='color:#90caf9; font-size:16px'>O Dilema do Investidor Brasileiro</strong><br><br>
    Imagine que você tem <strong>R$ 10.000</strong> para investir e precisa decidir entre três caminhos:
    deixar na <strong>Poupança</strong>, aplicar em <strong>Renda Fixa</strong> (CDB ou Tesouro Direto)
    ou arriscar na <strong>Renda Variável</strong> via IBOVESPA.<br><br>
    A resposta certa depende, em grande parte, de uma única variável macroeconômica:
    <strong>a taxa Selic</strong>. Quando os juros sobem, a renda fixa fica mais atrativa e a
    bolsa de valores sofre. Quando caem, o mercado de capitais ganha impulso.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")
    col_selic, col_ibov = st.columns([1, 2])

    with col_selic:
        st.markdown("<div class='sec'>Taxa Selic Atual</div>", unsafe_allow_html=True)
        cor_selic = "neg" if selic_override >= 0.12 else ("warn" if selic_override >= 0.09 else "pos")
        st.markdown(
            f"<div class='kpi' style='padding:28px;'>"
            f"<div class='kpi-label'>Meta da Taxa Básica de Juros</div>"
            f"<div class='kpi-value {cor_selic}' style='font-size:52px'>{selic_override*100:.2f}%</div>"
            f"<div class='kpi-sub'>ao ano — definida pelo Copom (BCB)</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        selic_mensal = ((1 + selic_override) ** (1 / 12) - 1) * 100
        st.markdown(
            f"<div style='margin-top:12px; padding:12px 16px; background:#1a1f2e;"
            f"border-radius:8px; border:1px solid #2d3748;'>"
            f"<span style='color:#8892a4; font-size:11px; text-transform:uppercase;'>Equivalente mensal</span><br>"
            f"<span style='color:#64b5f6; font-size:22px; font-weight:700;'>{selic_mensal:.4f}% a.m.</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown("")
        if selic_override >= 0.13:
            st.warning("Selic **alta**: renda fixa muito atrativa. Bolsa pressionada.")
        elif selic_override >= 0.09:
            st.info("Selic **moderada**: equilíbrio entre risco e retorno.")
        else:
            st.success("Selic **baixa**: estímulo ao mercado de capitais.")

    with col_ibov:
        st.markdown("<div class='sec'>IBOVESPA x Selic — Últimos 12 Meses</div>", unsafe_allow_html=True)
        if not ibov_m.empty and not merged.empty:
            _cutoff   = ibov_m.index[-1] - pd.Timedelta(days=365)
            ibov_12m  = ibov_m[ibov_m.index >= _cutoff]
            selic_12m = merged[merged.index >= _cutoff][["selic_anual"]]
            st.plotly_chart(viz.grafico_selic_ibov(ibov_12m, selic_12m), width='stretch', key='chart_selic_ibov')
        else:
            st.warning("Dados não disponíveis. Clique em 'Atualizar Dados'.")

    st.divider()
    st.markdown("<div class='sec'>Como a Selic Afeta Cada Estratégia</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class='kpi' style='text-align:left; padding:20px;'>
        <div style='font-size:28px; margin-bottom:8px;'>🏦</div>
        <div style='color:#66BB6A; font-weight:700; font-size:15px;'>Poupança</div>
        <div style='color:#8892a4; font-size:12px; margin:6px 0;'>MENOR RISCO</div>
        <div style='color:#c8d6e5; font-size:13px; line-height:1.6;'>
        Rende <strong>70% da Selic</strong> quando a meta ≥ 8,5% a.a.
        Isento de IR. Liquidez diária. FGC até R$ 250 mil.
        </div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class='kpi' style='text-align:left; padding:20px;'>
        <div style='font-size:28px; margin-bottom:8px;'>📄</div>
        <div style='color:#42A5F5; font-weight:700; font-size:15px;'>Renda Fixa</div>
        <div style='color:#8892a4; font-size:12px; margin:6px 0;'>BAIXO RISCO</div>
        <div style='color:#c8d6e5; font-size:13px; line-height:1.6;'>
        CDB ou Tesouro Selic rendem <strong>~100% do CDI</strong>.
        IR regressivo: 22,5% até 6 meses → 15% após 2 anos.
        </div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class='kpi' style='text-align:left; padding:20px;'>
        <div style='font-size:28px; margin-bottom:8px;'>📈</div>
        <div style='color:#AB47BC; font-weight:700; font-size:15px;'>Renda Variável</div>
        <div style='color:#8892a4; font-size:12px; margin:6px 0;'>ALTO RISCO / ALTO POTENCIAL</div>
        <div style='color:#c8d6e5; font-size:13px; line-height:1.6;'>
        IBOVESPA reflete as maiores empresas do Brasil.
        Pode superar a renda fixa no longo prazo, com alta volatilidade.
        </div></div>""", unsafe_allow_html=True)

    if not merged.empty:
        st.markdown("")
        st.markdown("<div class='sec'>Dispersão: Selic x Retorno Mensal do IBOVESPA</div>", unsafe_allow_html=True)
        col_sc, col_ci = st.columns([3, 2])
        with col_sc:
            st.plotly_chart(viz.grafico_scatter_selic_ibov(merged), width='stretch', key='chart_scatter_t1')
        with col_ci:
            corr = analysis.pearson_selic_ibov(merged)
            sinal_c = "badge-neg" if corr["r"] < 0 else "badge-pos"
            sig_c   = "Significativo (p < 0,05)" if corr["significativo"] else "Nao significativo"
            st.markdown(
                f"<div class='kpi' style='text-align:left; padding:20px; margin-top:40px;'>"
                f"<div class='kpi-label'>Correlação de Pearson</div>"
                f"<div class='kpi-value {'neg' if corr['r']<0 else 'pos'}' style='font-size:44px;'>{corr['r']}</div>"
                f"<div style='margin:8px 0;'><span class='{sinal_c}'>{sig_c}</span></div>"
                f"<div style='color:#8892a4; font-size:12px;'>R² = {corr['r2']} | p = {corr['p_value']}</div>"
                f"<div style='color:#c8d6e5; font-size:13px; margin-top:10px; border-top:1px solid #2d3748; padding-top:8px;'>"
                f"{corr['interpretacao']}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════
#  TAB 2 — SIMULADOR
# ═══════════════════════════════════════════════════════

with tab2:
    st.markdown("""
    <div class='narrative'>
    <strong style='color:#90caf9'>Simulador de Investimento Personalizado</strong><br><br>
    Informe o valor que deseja investir, o prazo e sua tolerância ao risco.
    O simulador calculará o montante final esperado para cada estratégia
    com base na Selic atual e no histórico do IBOVESPA.
    </div>
    """, unsafe_allow_html=True)

    c_left, c_right = st.columns([1, 1.4])
    with c_left:
        st.markdown("<div class='sec'>Parâmetros da Simulação</div>", unsafe_allow_html=True)
        valor = st.number_input(
            "Valor a investir (R$)", min_value=100.0, max_value=10_000_000.0,
            value=10_000.0, step=500.0, format="%.2f",
        )
        periodo = st.slider("Período de investimento (meses)", min_value=1, max_value=120, value=12)
        tolerancia = st.radio(
            "Tolerância ao risco", options=["baixa", "media", "alta"],
            index=1, horizontal=True,
        )
        st.markdown("")
        simular = st.button("▶ Simular", type="primary", use_container_width=True)

    with c_right:
        if simular or "sim_result" in st.session_state:
            if simular:
                with st.spinner("Calculando cenários..."):
                    sim = ml_model.simular_investimento(
                        selic_anual=selic_override,
                        valor=valor,
                        periodo_meses=periodo,
                        tolerancia=tolerancia,
                    )
                    st.session_state["sim_result"] = sim
                    st.session_state["sim_valor"]  = valor

            sim       = st.session_state["sim_result"]
            valor_sim = st.session_state.get("sim_valor", valor)

            st.markdown("<div class='sec'>Resultado da Simulação</div>", unsafe_allow_html=True)
            ka, kb, kc = st.columns(3)
            for col_r, (nome, montante) in zip([ka, kb, kc], sim["montantes"].items()):
                ret = sim["retornos"][nome]
                tx  = sim["taxas_mensais"][nome]
                dest = " style='border-color:#1565c0;border-width:2px;'" if nome == sim["recomendacao"] else ""
                col_r.markdown(
                    f"<div class='kpi'{dest}>"
                    f"<div class='kpi-label'>{nome}</div>"
                    f"<div class='kpi-value pos'>R$ {montante:,.0f}</div>"
                    f"<div class='kpi-sub'>+R$ {ret:,.0f} | {tx:.4f}% a.m.</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    if "sim_result" in st.session_state:
        sim       = st.session_state["sim_result"]
        valor_sim = st.session_state.get("sim_valor", 10000.0)
        st.markdown("")
        col_b, col_s = st.columns(2)
        with col_b:
            st.plotly_chart(viz.grafico_comparativo_cenarios(sim, valor_sim), width='stretch', key='chart_sim_bar')
        with col_s:
            st.plotly_chart(viz.grafico_serie_crescimento(sim["serie_temporal"]), width='stretch', key='chart_sim_serie')

        rec   = sim["recomendacao"]
        cores = {"Poupanca": "#66BB6A", "Renda Fixa": "#42A5F5", "Renda Variavel": "#AB47BC"}
        cor   = cores.get(rec, "#64b5f6")
        st.markdown(
            f"<div class='rec-card'>"
            f"<div class='rec-title' style='color:{cor}'>Recomendação: {rec}</div>"
            f"<div class='rec-text'>{sim['justificativa']}</div>"
            f"<div class='rec-conf'>Confiança: {sim['confianca']}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        with st.expander("Ver tabela de taxas"):
            df_det = pd.DataFrame({
                "Estrategia":       list(sim["taxas_mensais"].keys()),
                "Taxa Mensal %":    list(sim["taxas_mensais"].values()),
                "Retorno Total R$": [sim["retornos"][k] for k in sim["taxas_mensais"]],
                "Montante Final R$":[sim["montantes"][k] for k in sim["taxas_mensais"]],
            }).set_index("Estrategia")
            st.dataframe(df_det.style.format({
                "Taxa Mensal %": "{:.4f}",
                "Retorno Total R$": "R$ {:,.2f}",
                "Montante Final R$": "R$ {:,.2f}",
            }), use_container_width=True)


# ═══════════════════════════════════════════════════════
#  TAB 3 — ANÁLISE ESTATÍSTICA
# ═══════════════════════════════════════════════════════

with tab3:
    if merged.empty or ibov_m.empty:
        st.warning("Dados não disponíveis. Clique em 'Atualizar Dados' na sidebar.")
    else:
        st.markdown("<div class='sec'>Correlação de Pearson — Selic x IBOVESPA</div>", unsafe_allow_html=True)
        corr = analysis.pearson_selic_ibov(merged)
        col_cr, col_sc = st.columns([1, 1.8])
        with col_cr:
            sn = "neg" if corr["r"] < -0.3 else ("pos" if corr["r"] > 0.3 else "neu")
            sc = "badge-pos" if corr["significativo"] else "badge-neg"
            sg = "Significativo" if corr["significativo"] else "Nao significativo"
            st.markdown(
                f"<div class='kpi' style='text-align:left; padding:22px;'>"
                f"<div class='kpi-label'>r de Pearson</div>"
                f"<div class='kpi-value {sn}' style='font-size:44px'>{corr['r']}</div>"
                f"<div style='margin:8px 0'><span class='{sc}'>{sg}</span></div>"
                f"<div style='color:#8892a4; font-size:12px;'>R² = {corr['r2']} | p = {corr['p_value']} | n = {corr['n']}</div>"
                f"<div style='color:#c8d6e5; font-size:13px; margin-top:10px; border-top:1px solid #2d3748; padding-top:8px;'>"
                f"{corr['interpretacao']}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with col_sc:
            st.plotly_chart(viz.grafico_scatter_selic_ibov(merged), width='stretch', key='chart_scatter_t3')

        st.divider()
        st.markdown("<div class='sec'>Estatísticas Descritivas — Retornos Diários (%)</div>", unsafe_allow_html=True)
        desc = analysis.stats_descritivas(ibov_m["retorno_diario"] * 100)
        cols_d = st.columns(len(desc))
        for col_d, (chave, val) in zip(cols_d, desc.items()):
            cls = "pos" if isinstance(val, float) and val > 0 else ("neg" if isinstance(val, float) and val < 0 else "neu")
            col_d.markdown(
                f"<div class='kpi'><div class='kpi-label'>{chave}</div>"
                f"<div class='kpi-value {cls}' style='font-size:18px'>{val}</div></div>",
                unsafe_allow_html=True,
            )

        st.markdown("")
        col_h, col_q = st.columns([1.4, 1])
        with col_h:
            st.plotly_chart(viz.grafico_distribuicao_retornos(ibov_m), width='stretch', key='chart_dist')
        with col_q:
            st.markdown("<div class='sec'>Quartis — Retorno Mensal IBOVESPA (%)</div>", unsafe_allow_html=True)
            q_df = analysis.analise_quartis(merged["ibov_retorno_mensal"])
            st.dataframe(q_df.style.format({"Valor": "{:.4f}"}), use_container_width=True, height=260)
            outliers = analysis.resumo_outliers(merged)
            if not outliers.empty:
                st.markdown("<div class='sec'>Outliers (IQR + Z-Score)</div>", unsafe_allow_html=True)
                st.dataframe(outliers, use_container_width=True)
            else:
                st.success("Nenhum outlier detectado no período analisado.")

        st.divider()
        col_hm, col_ma = st.columns(2)
        with col_hm:
            st.markdown("<div class='sec'>Heatmap de Correlação</div>", unsafe_allow_html=True)
            st.plotly_chart(viz.grafico_heatmap_correlacao(merged), width='stretch', key='chart_heatmap')
        with col_ma:
            st.markdown("<div class='sec'>IBOVESPA — Médias Móveis MA20 e MA50</div>", unsafe_allow_html=True)
            _c = ibov_m.index[-1] - pd.Timedelta(days=365)
            st.plotly_chart(viz.grafico_medias_moveis(ibov_m[ibov_m.index >= _c]), width='stretch', key='chart_ma')

        tend = analysis.tendencia_ibovespa(ibov_m)
        tc = "badge-pos" if tend["tendencia"] == "Alta" else "badge-neg"
        sg = "Significativa" if tend["significativa"] else "Nao significativa"
        st.markdown(
            f"<span class='{tc}'>Tendência: {tend['tendencia']}</span> | "
            f"Inclinação = {tend['inclinacao']:.2f} pts/dia | R² = {tend['r2']} | {sg}",
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════
#  TAB 4 — ML RECOMENDAÇÃO
# ═══════════════════════════════════════════════════════

with tab4:
    st.markdown("<div class='sec'>Modelo de Regressão: Selic → Retorno Esperado do IBOVESPA</div>",
                unsafe_allow_html=True)
    st.markdown("""
    <div class='narrative'>
    O modelo de <strong>Ridge Regression</strong> usa a taxa Selic atual e seus valores
    defasados (lags de 1 e 2 meses) como features para prever o retorno mensal esperado
    do IBOVESPA. O split temporal 80/20 garante ausência de <em>data leakage</em>.<br><br>
    <strong>Features:</strong> selic_anual, selic_lag1, selic_delta, ibov_lag1
    &nbsp;|&nbsp; <strong>Target:</strong> ibov_retorno_mensal (%)
    </div>
    """, unsafe_allow_html=True)

    if ml_f.empty:
        st.warning("Dados de features ML não disponíveis. Clique em 'Atualizar Dados'.")
    else:
        treinar = st.button("🚀 Treinar Modelo", type="primary")

        if treinar:
            with st.spinner("Treinando Ridge Regression..."):
                resultado = ml_model.treinar_modelo(ml_f)
                st.session_state["ml_resultado"] = resultado

        if "ml_resultado" in st.session_state:
            res = st.session_state["ml_resultado"]

            mk1, mk2, mk3, mk4 = st.columns(4)
            _kpi(mk1, "R² no Teste",  f"{res['r2']:.4f}"  if not np.isnan(res['r2'])  else "N/A", "qualidade do ajuste", "pos" if not np.isnan(res['r2']) and res['r2'] > 0.2 else "warn")
            _kpi(mk2, "MAE no Teste", f"{res['mae']:.4f}" if not np.isnan(res['mae']) else "N/A", "pp de retorno mensal", "neu")
            _kpi(mk3, "Obs. Treino",  str(res["n_treino"]), "Split 80/20 temporal", "neu")
            _kpi(mk4, "Obs. Teste",   str(res["n_teste"]),  "sem data leakage", "neu")

            st.markdown("")

            if res["coeficientes"]:
                col_cf, col_pp = st.columns(2)
                with col_cf:
                    st.markdown("<div class='sec'>Coeficientes do Modelo</div>", unsafe_allow_html=True)
                    coef_df = pd.DataFrame.from_dict(res["coeficientes"], orient="index", columns=["Coeficiente"]).sort_values("Coeficiente")
                    st.dataframe(coef_df.style.format("{:.4f}").bar(subset=["Coeficiente"], color=["#b71c1c","#1b5e20"]), use_container_width=True)
                    st.caption("Coeficiente negativo para selic_anual confirma: Selic alta → retorno IBOVESPA menor.")

                with col_pp:
                    st.markdown("<div class='sec'>Predição vs Realidade (Conjunto de Teste)</div>", unsafe_allow_html=True)
                    if not res["df_teste"].empty and "pred" in res["df_teste"].columns:
                        te = res["df_teste"]
                        fig_p = go.Figure()
                        fig_p.add_trace(go.Scatter(x=te.index, y=te["ibov_retorno_mensal"], name="Real",
                            mode="lines+markers", line=dict(color="#2196F3", width=2)))
                        fig_p.add_trace(go.Scatter(x=te.index, y=te["pred"], name="Previsto",
                            mode="lines+markers", line=dict(color="#FF9800", width=2, dash="dash")))
                        fig_p.update_layout(
                            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                            font=dict(color="#e0e0e0"), height=280,
                            margin=dict(l=40,r=20,t=30,b=40),
                            xaxis=dict(gridcolor="#1e2533"),
                            yaxis=dict(gridcolor="#1e2533", title="Retorno Mensal (%)"),
                            hovermode="x unified",
                            legend=dict(bgcolor="rgba(30,35,50,0.8)"),
                        )
                        st.plotly_chart(fig_p, width='stretch', key='chart_ml_pred')
                    else:
                        st.info("Dados de teste insuficientes.")

            st.divider()
            st.markdown("<div class='sec'>Recomendação Inteligente com ML</div>", unsafe_allow_html=True)

            ci1, ci2, ci3 = st.columns(3)
            with ci1:
                valor_ml = st.number_input("Valor (R$)", key="ml_v", min_value=100.0, max_value=10_000_000.0, value=10_000.0, step=500.0, format="%.2f")
            with ci2:
                periodo_ml = st.slider("Período (meses)", key="ml_p", min_value=1, max_value=120, value=24)
            with ci3:
                risco_ml = st.radio("Tolerância ao risco", key="ml_r", options=["baixa","media","alta"], index=1, horizontal=True)

            if st.button("🔍 Gerar Recomendação ML", type="primary"):
                ibov_lag1_ml = float(ml_f["ibov_lag1"].dropna().iloc[-1]) if not ml_f.empty and "ibov_lag1" in ml_f.columns else 0.8
                pred_ret = ml_model.prever_retorno_ibov(res, selic_override, ibov_lag1_ml)
                sim_ml = ml_model.simular_investimento(
                    selic_anual=selic_override, valor=valor_ml,
                    periodo_meses=periodo_ml, tolerancia=risco_ml,
                    ibov_pred_mensal=pred_ret,
                )
                st.session_state["sim_ml"]  = sim_ml
                st.session_state["pred_ml"] = pred_ret

            if "sim_ml" in st.session_state:
                sim_ml = st.session_state["sim_ml"]
                pred   = st.session_state.get("pred_ml", 0.0)

                st.markdown(
                    f"<div style='background:#1a1f2e; border-radius:8px; padding:10px 16px; display:inline-block;'>"
                    f"<span style='color:#8892a4;font-size:12px;'>Retorno mensal previsto pelo modelo:</span>&nbsp;"
                    f"<span style='color:#FF9800; font-weight:700; font-size:18px;'>{pred:.4f}%</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                st.markdown("")

                cr1, cr2, cr3 = st.columns(3)
                for col_r, (nome, montante) in zip([cr1,cr2,cr3], sim_ml["montantes"].items()):
                    ret = sim_ml["retornos"][nome]
                    dest = " style='border-color:#1565c0;border-width:2px;'" if nome == sim_ml["recomendacao"] else ""
                    col_r.markdown(
                        f"<div class='kpi'{dest}><div class='kpi-label'>{nome}</div>"
                        f"<div class='kpi-value pos'>R$ {montante:,.0f}</div>"
                        f"<div class='kpi-sub'>+R$ {ret:,.0f}</div></div>",
                        unsafe_allow_html=True,
                    )

                st.markdown("")
                rec_ml = sim_ml["recomendacao"]
                cr_map = {"Poupanca":"#66BB6A","Renda Fixa":"#42A5F5","Renda Variavel":"#AB47BC"}
                cor_r  = cr_map.get(rec_ml, "#64b5f6")
                st.markdown(
                    f"<div class='rec-card'>"
                    f"<div class='rec-title' style='color:{cor_r}'>Recomendação ML: {rec_ml}</div>"
                    f"<div class='rec-text'>{sim_ml['justificativa']}</div>"
                    f"<div class='rec-conf'>Confiança: {sim_ml['confianca']}</div>"
                    f"<div style='color:#8892a4; font-size:12px; margin-top:8px;'>"
                    f"Selic: {selic_override*100:.2f}% a.a. | Ret. IBOV previsto: {pred:.4f}% a.m. | Perfil: {risco_ml}"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )
                st.markdown("")
                cbm, csm = st.columns(2)
                with cbm:
                    st.plotly_chart(viz.grafico_comparativo_cenarios(sim_ml, valor_ml), width='stretch', key='chart_ml_bar')
                with csm:
                    st.plotly_chart(viz.grafico_serie_crescimento(sim_ml["serie_temporal"]), width='stretch', key='chart_ml_serie')
