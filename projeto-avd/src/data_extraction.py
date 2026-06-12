"""
Fase 1 — Extração de Dados

Fontes:
  - Yahoo Finance (yfinance): histórico do IBOVESPA (^BVSP) — últimos 2 anos
  - Banco Central do Brasil (SGS/API): taxa Selic meta histórica (série 11)
  - Portais de economia (HTML): scraping de manchetes de mercado

Retorna DataFrames diretamente, sem persistência em disco.
"""

from datetime import datetime, timedelta
from urllib.parse import urljoin

import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup

_BCB_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.11/dados"
_HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


# ── IBOVESPA ──────────────────────────────────────────────────────────────────

def fetch_ibovespa(period: str = "2y") -> pd.DataFrame:
    """
    Baixa fechamentos ajustados do IBOVESPA via yfinance.
    O ticker ^BVSP representa o índice no Yahoo Finance.
    """
    raw = yf.Ticker("^BVSP").history(period=period, auto_adjust=True)
    df = raw[["Close", "Volume"]].rename(columns={"Close": "ibovespa", "Volume": "volume"})
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "data"
    return df.dropna(subset=["ibovespa"])


# ── NEWS SCRAPING ────────────────────────────────────────────────────────────

def fetch_economic_news(limit: int = 12) -> pd.DataFrame:
    """
    Faz scraping de manchetes em portais de economia usando HTML tags.
    Extrai titulo, URL, fonte e tag HTML do bloco identificado.
    """
    fontes = [
        ("InfoMoney", "https://www.infomoney.com.br/mercados/"),
        ("CNN Brasil", "https://www.cnnbrasil.com.br/economia/"),
    ]
    registros = []

    for fonte, url in fontes:
        try:
            resposta = requests.get(url, headers=_HTTP_HEADERS, timeout=20)
            resposta.raise_for_status()
            registros.extend(_parse_news_html(resposta.text, fonte, url))
        except Exception as exc:
            print(f"  AVISO: scraping de {fonte} indisponivel ({exc}).")

    df = pd.DataFrame(registros)
    if df.empty:
        fallback = _economic_news_fallback(limit)
        fallback.attrs["fonte"] = "fallback"
        return fallback

    df = df.drop_duplicates(subset=["titulo", "url"]).head(limit).copy()
    df["coletado_em"] = pd.Timestamp.utcnow().tz_localize(None)
    df.attrs["fonte"] = "scraping"
    return df


def _parse_news_html(html: str, fonte: str, base_url: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    itens = []
    vistos = set()

    for heading in soup.find_all(["h1", "h2", "h3", "h4"]):
        titulo = " ".join(heading.stripped_strings)
        if len(titulo) < 25:
            continue

        anchor = None
        if heading.get("href"):
            anchor = heading
        elif heading.find("a", href=True):
            anchor = heading.find("a", href=True)
        else:
            anchor = heading.find_parent("a", href=True)

        if anchor is None:
            continue

        href = anchor.get("href", "").strip()
        if not href or href.startswith("#") or href.startswith("javascript:"):
            continue

        url = urljoin(base_url, href)
        chave = (titulo.lower(), url)
        if chave in vistos or any(token in url.lower() for token in ("/autor/", "/tag/", "/wp-content/")):
            continue

        vistos.add(chave)
        itens.append({
            "fonte": fonte,
            "titulo": titulo,
            "url": url,
            "tag_html": heading.name,
        })

        if len(itens) >= 10:
            break

    return itens


# ── SELIC ─────────────────────────────────────────────────────────────────────

def fetch_selic(meses: int = 26) -> pd.DataFrame:
    """
    Busca a Meta da Taxa SELIC histórica via API SGS do Banco Central.
    Série 11 → taxa diária (ex.: 0.0516 = 0,0516% a.d.).
    Reagrupa para frequência mensal e anualiza em 252 dias úteis.
    """
    fim    = datetime.now()
    inicio = fim - timedelta(days=meses * 31)
    params = {
        "dataInicial": inicio.strftime("%d/%m/%Y"),
        "dataFinal":   fim.strftime("%d/%m/%Y"),
        "formato":     "json",
    }
    try:
        resp = requests.get(_BCB_URL, params=params, timeout=20)
        resp.raise_for_status()
        raw = resp.json()
        df  = pd.DataFrame(raw)
        df["data"]  = pd.to_datetime(df["data"], format="%d/%m/%Y")
        df["valor"] = pd.to_numeric(df["valor"], errors="coerce") / 100
        df = df.dropna().set_index("data").rename(columns={"valor": "selic_diaria"})
        mensal = _agregar_selic_mensal(df)
        mensal.attrs["fonte"] = "api"
        return mensal
    except Exception as exc:
        print(f"  AVISO: API BCB indisponivel ({exc}). Usando dados historicos sinteticos.")
        sintetica = _selic_sintetica(meses)
        sintetica.attrs["fonte"] = "sintetica"
        return sintetica


def _agregar_selic_mensal(df_diaria: pd.DataFrame) -> pd.DataFrame:
    """
    Último valor do mês → meta Selic vigente.
    BCB série 11 pode retornar taxa diária (% a.d., ex: 0.0516) ou
    anualizada (% a.a., ex: 13.75). Detecta o formato pelo valor médio:
    valores < 0.01 após /100 são diários e precisam ser anualizados.
    """
    last = df_diaria["selic_diaria"].resample("ME").last()
    if last.mean() < 0.01:
        # Taxa diária decimal (ex: 0.000516) → anualizar em 252 dias úteis
        selic_anual = (1 + last) ** 252 - 1
    else:
        # Taxa já anualizada em decimal (ex: 0.1375)
        selic_anual = last
    selic_mensal = (1 + selic_anual) ** (1 / 12) - 1
    return pd.DataFrame({
        "selic_anual":  selic_anual,
        "selic_mensal": selic_mensal,
    }).dropna()


def _selic_sintetica(meses: int) -> pd.DataFrame:
    """
    Série histórica aproximada da Selic (2023-2026) para fallback
    quando a API do BCB não responde.
    """
    historico = [
        ("2023-01", 0.1375), ("2023-02", 0.1375), ("2023-03", 0.1375),
        ("2023-04", 0.1375), ("2023-05", 0.1375), ("2023-06", 0.1375),
        ("2023-07", 0.1375), ("2023-08", 0.1325), ("2023-09", 0.1275),
        ("2023-10", 0.1225), ("2023-11", 0.1175), ("2023-12", 0.1175),
        ("2024-01", 0.1125), ("2024-02", 0.1075), ("2024-03", 0.1075),
        ("2024-04", 0.1075), ("2024-05", 0.1075), ("2024-06", 0.1065),
        ("2024-07", 0.1065), ("2024-08", 0.1065), ("2024-09", 0.1075),
        ("2024-10", 0.1115), ("2024-11", 0.1165), ("2024-12", 0.1215),
        ("2025-01", 0.1315), ("2025-02", 0.1365),
    ]
    datas   = pd.to_datetime([h[0] for h in historico], format="%Y-%m") + pd.offsets.MonthEnd(0)
    valores = [h[1] for h in historico]
    df = pd.DataFrame({"selic_anual": valores}, index=datas)
    df["selic_mensal"] = df["selic_anual"].apply(lambda r: (1 + r) ** (1 / 12) - 1)
    return df.tail(meses)


def _economic_news_fallback(limit: int) -> pd.DataFrame:
    exemplos = [
        {
            "fonte": "Fallback",
            "titulo": "Mercado acompanha decisao sobre juros e impacto no custo do credito",
            "url": "https://exemplo.local/noticia-1",
            "tag_html": "h2",
        },
        {
            "fonte": "Fallback",
            "titulo": "Bolsa brasileira sobe com expectativa de inflacao mais controlada",
            "url": "https://exemplo.local/noticia-2",
            "tag_html": "h3",
        },
        {
            "fonte": "Fallback",
            "titulo": "Investidores reavaliam renda fixa apos mudanca nas projecoes da Selic",
            "url": "https://exemplo.local/noticia-3",
            "tag_html": "h2",
        },
        {
            "fonte": "Fallback",
            "titulo": "Dolar recua e melhora humor para ativos de risco no Brasil",
            "url": "https://exemplo.local/noticia-4",
            "tag_html": "h3",
        },
    ]
    df = pd.DataFrame(exemplos[:limit])
    df["coletado_em"] = pd.Timestamp.utcnow().tz_localize(None)
    return df
