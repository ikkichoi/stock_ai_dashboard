import os
import io
import re
import json
import time
import urllib.request
from datetime import datetime
import pandas as pd
import streamlit as st
import yfinance as yf
from dotenv import load_dotenv
from google import genai
from google.genai import types
from apscheduler.schedulers.blocking import BlockingScheduler
from database import init_db, insert_analysis, save_universe, load_universe

load_dotenv()


def _load_gemini_key() -> str:
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return str(st.secrets["GEMINI_API_KEY"]).strip().strip('"').strip("'")
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY", "").strip().strip('"').strip("'")


GEMINI_API_KEY = _load_gemini_key()
API_SLEEP_SEC = float(os.getenv("API_SLEEP_SEC", "4.2"))

SYSTEM_ANALYST_PROMPT = """
당신은 2026년 기관투자자 기준의 수석 주식 리서치 애널리스트입니다.
제공된 실제 재무 팩트(Owner's Earnings = FCF - SBC, GAAP OPM, EV/EBITDA, 순부채/EBITDA, 매출성장률 등)와 실시간 웹 검색(Google Search)을 바탕으로 아래 7단계 리서치 원칙에 따라 엄격히 평가하십시오.
특히 액면분할(Stock Split)·병합이 반영된 2026년 현재 실제 거래 주가를 반드시 기준으로 삼아 목표주가를 산출하십시오.

[7단계 리서치 분석 원칙]
1. 사업 & 수익구조: 핵심 매출원, 가격결정력, 반복성 점검. GAAP 영업이익률(OPM)과 비GAAP 조정 EBITDA를 절대 혼용하지 않음.
2. TAM & 실제 침투율: 시장 규모 단순 추정이 아닌 실제 점유율·단가(ARPU)·수익화 개선 여부를 구분하고, 미검증 신사업은 옵션으로만 취급.
3. 재무 건전성 & Owner's Earnings(OE):
   - OE 대리값 = 보고 FCF에서 주식보상비용(SBC)을 경제적 희석비용으로 차감한 값.
   - 순현금/순부채, 고객예수금·우선청구권 등 자본구조와 레버리지(순부채/EBITDA)를 반영.
4. 경영진 & 자본배분: 창업자/경영진 실행력, 의결권 집중, 자사주 매입·M&A 시너지 검증 여부 평가.
5. 업종별 최적 가치평가 (OE DCF vs 대체 멀티플):
   - 현금흐름이 안정적인 플랫폼/소프트웨어/소비재 기업: Owner's Earnings 기반 5년 DCF (성장률 / 할인율 / 영구성장률 g 명시) 우선 적용.
   - 자본집약·고레버리지·CAPEX 변동 큰 기업(통신·케이블, 인프라, 유틸리티 등): OE 정상화 한계를 명시하고 EV/조정 EBITDA 및 레버리지 배수로 대체 평가.
6. 매매 시점 & 안전마진 분리:
   - 단기 주가 급등락만으로 매수 신호를 만들지 않고, '사업 매력도' · '가격 안전마진(Base 괴리율)' · '다음 분기 관찰 지표'를 구분해 판정.
7. 엄격한 0~100점 스코어링 & Bear/Base/Bull 목표주가:
   - 반드시 제공된 액면분할 반영 현재 실제 주가(actual_close_price)를 기준으로 bear_target <= base_target <= bull_target 순서를 준수하십시오. 과거 분할 전 주가 스케일을 절대 사용하지 마십시오.

반드시 아래 JSON 스키마로만 응답하십시오:
{
  "actual_close_price": 액면분할이 반영된 2026년 현재 실제 주가 숫자(소수점 2자리),
  "market_cap": "시가총액 문자열 (예: $250.40B, 제공값이 N/A면 웹 검색으로 실제 시가총액 기입)",
  "per": 후행 또는 선행 PER 숫자(예: 28.5, 없으면 null),
  "ev_ebitda": EV/EBITDA 배수 숫자(예: 21.4, 없으면 null),
  "fcf_yield": FCF Yield 백분율 숫자(예: 3.45, 없으면 null),
  "oe_yield": Owner's Earnings(FCF-SBC) Yield 백분율 숫자(예: 3.10, 없으면 null),
  "gaap_opm": GAAP 영업이익률 백분율 숫자(예: 27.8, 없으면 null),
  "net_debt_ebitda": 순부채/EBITDA 배수 숫자(순현금이면 음수 예: -0.45, 없으면 null),
  "score": 1~100 사이 정수,
  "opinion": "강력매수" | "매수" | "중립" | "매도",
  "valuation_metric": "주 평가법 및 핵심 배수 (예: OE DCF [성장 15%/할인 9.5%/g 3.0%] 또는 EV/EBITDA 5.25x)",
  "dcf_assumptions": "시나리오 가정 요약 (예: Bear 10%/10.5%/2.5% | Base 20%/9.5%/3.0% | Bull 25%/9.0%/3.5% 또는 EV/EBITDA 4.5x/5.25x/5.75x)",
  "normalization_note": "액면분할 반영 여부, SBC 차감, 순부채/순현금, GICS 섹터 분류 등 회계·정상화 주의사항 1줄 요약",
  "bear_target": 보수적 목표주가 숫자(소수점 2자리),
  "base_target": 기준 목표주가 숫자(소수점 2자리),
  "bull_target": 낙관적 목표주가 숫자(소수점 2자리),
  "summary": "핵심 투자 결론 한 줄 요약 (55자 이내)",
  "reasoning": "[사업·해자] / [재무·OE·부채] / [밸류에이션·관찰지표] 3가지 관점에서 구체적 수치를 인용해 4~5문장으로 작성"
}
"""


def _get_candidate_models(client) -> list[str]:
    """무료 등급에서 할당량이 넉넉한 3.x Flash / Flash-Lite 계열만 반환 (2.5 이하 구형 및 Pro 제외)."""
    preferred = [
        "gemini-3.8-flash",
        "gemini-3.8-flash-lite",
        "gemini-3.5-flash",
        "gemini-3.0-flash",
    ]
    available = []
    try:
        for m in client.models.list():
            name = m.name.replace("models/", "")
            if (
                "generateContent" in (m.supported_actions or [])
                and "flash" in name.lower()
                and not any(
                    x in name.lower()
                    for x in ["image", "audio", "tts", "live", "embedding", "vision", "1.0", "1.5", "2.0", "2.5"]
                )
            ):
                available.append(name)
    except Exception:
        pass

    ordered = [p for p in preferred if p in available]
    for m in sorted(available, reverse=True):
        if m not in ordered:
            ordered.append(m)

    return ordered if ordered else ["gemini-3.8-flash", "gemini-3.8-flash-lite"]


def sync_sp500_universe() -> pd.DataFrame:
    """Wikipedia 크롤링 우선 시도 후 실패 시 GitHub 공식 S&P 500 CSV 백업 소스로 100% 동기화."""
    init_db()
    universe_df = pd.DataFrame()

    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url, storage_options={"User-Agent": "Mozilla/5.0"})
        raw_df = tables[0]
        universe_df = pd.DataFrame({
            "ticker": raw_df["Symbol"].astype(str).str.strip().str.replace(".", "-", regex=False),
            "company": raw_df["Security"].astype(str).str.strip(),
            "sector": raw_df["GICS Sector"].astype(str).str.strip(),
        })
    except Exception as e1:
        print(f"[Wikipedia 동기화 우회 -> 백업 CSV 사용] {e1}", flush=True)
        backup_url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
        raw_df = pd.read_csv(backup_url)
        universe_df = pd.DataFrame({
            "ticker": raw_df["Symbol"].astype(str).str.strip().str.replace(".", "-", regex=False),
            "company": raw_df["Security"].astype(str).str.strip(),
            "sector": raw_df["GICS Sector"].astype(str).str.strip(),
        })

    extra = pd.DataFrame([
        {"ticker": "RDDT", "company": "Reddit", "sector": "Communication Services"},
        {"ticker": "000660.KS", "company": "SK하이닉스", "sector": "Semiconductors (KR)"},
        {"ticker": "005930.KS", "company": "삼성전자", "sector": "Semiconductors (KR)"},
    ])
    universe_df = pd.concat([extra, universe_df], ignore_index=True).drop_duplicates(subset=["ticker"])
    universe_df["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    save_universe(universe_df)
    return universe_df


def get_target_universe() -> pd.DataFrame:
    init_db()
    df = load_universe()
    if df.empty or len(df) < 400:
        try:
            df = sync_sp500_universe()
        except Exception as e:
            print(f"[유니버스 동기화 경고] {e}", flush=True)
    return df


def _fetch_batch_closes(tickers: list[str]) -> dict:
    """100종목 단위 배치 다운로드로 최근 종가 및 5일 등락률 수집."""
    price_map = {}
    chunk_size = 100
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        try:
            data = yf.download(chunk, period="5d", group_by="ticker", progress=False, threads=True)
            if data.empty:
                continue
            for t in chunk:
                try:
                    closes = data["Close"].dropna() if len(chunk) == 1 else data[t]["Close"].dropna()
                    if not closes.empty:
                        curr = float(closes.iloc[-1])
                        prev = float(closes.iloc[0]) if len(closes) > 1 else curr
                        chg_5d = ((curr - prev) / prev * 100) if prev else 0.0
                        price_map[t] = {"close": round(curr, 2), "chg_5d": round(chg_5d, 2)}
                except Exception:
                    continue
        except Exception as e:
            print(f"[배치 시세 오류] {e}", flush=True)
    return price_map


def _format_money(val) -> str:
    if val is None or not isinstance(val, (int, float)) or pd.isna(val):
        return "N/A"
    sign = "-" if val < 0 else ""
    abs_v = abs(val)
    if abs_v >= 1e12:
        return f"{sign}${abs_v / 1e12:.2f}T"
    if abs_v >= 1e9:
        return f"{sign}${abs_v / 1e9:.2f}B"
    if abs_v >= 1e6:
        return f"{sign}${abs_v / 1e6:.0f}M"
    return f"{sign}${abs_v:,.0f}"


def _fetch_finviz_fundamentals(ticker: str) -> dict:
    """yfinance 차단 시 Finviz 스냅샷 테이블에서 시가총액, PER, GAAP OPM, 매출성장률 백업 파싱."""
    res = {}
    if "." in ticker:
        return res
    try:
        from curl_cffi import requests as cffi_requests
        s = cffi_requests.Session(impersonate="chrome")
        r = s.get(f"https://finviz.com/quote.ashx?t={ticker}", timeout=8)
        if r.status_code != 200:
            return res
        html = r.text

        def _grab(label: str) -> str | None:
            m = re.search(rf">{re.escape(label)}</td>\s*<td[^>]*>(?:<[^>]+>)*([^<]+)", html)
            return m.group(1).strip() if m else None

        mcap = _grab("Market Cap")
        if mcap and mcap != "-":
            res["market_cap"] = f"${mcap}" if not mcap.startswith("$") else mcap

        pe = _grab("P/E")
        if pe and pe != "-":
            try:
                res["per"] = round(float(pe.replace(",", "")), 2)
            except Exception:
                pass

        opm = _grab("Oper. Margin")
        if opm and opm != "-":
            try:
                res["gaap_opm"] = round(float(opm.replace("%", "").replace(",", "")), 1)
            except Exception:
                pass

        sales_yy = _grab("Sales Y/Y") or _grab("Sales Q/Q")
        if sales_yy and sales_yy != "-":
            try:
                res["rev_growth_pct"] = round(float(sales_yy.replace("%", "").replace(",", "")), 1)
            except Exception:
                pass

        price_s = _grab("Price")
        if price_s and price_s != "-":
            try:
                res["current_price"] = round(float(price_s.replace(",", "")), 2)
            except Exception:
                pass
    except Exception:
        pass
    return res


def _fetch_fundamentals(ticker: str) -> dict:
    """curl_cffi + yfinance 우선 시도 후 누락 항목은 Finviz 스냅샷에서 자동 보완."""
    base = {
        "current_price": 0.0,
        "market_cap": "N/A",
        "per": None,
        "ev_ebitda": None,
        "fcf_str": "N/A",
        "sbc_str": "미확인",
        "oe_str": "N/A",
        "fcf_yield": None,
        "oe_yield": None,
        "gaap_opm": None,
        "rev_growth_pct": None,
        "net_debt_str": "N/A",
        "net_debt_ebitda": None,
    }

    try:
        from curl_cffi import requests as cffi_requests
        session = cffi_requests.Session(impersonate="chrome")
        tk = yf.Ticker(ticker, session=session)
    except Exception:
        tk = yf.Ticker(ticker)

    try:
        info = tk.info or {}

        mcap_raw = info.get("marketCap")
        shares_raw = info.get("sharesOutstanding")

        price_raw = (
            info.get("currentPrice")
            or info.get("regularMarketPrice")
            or info.get("previousClose")
        )
        if (
            not isinstance(price_raw, (int, float))
            and isinstance(mcap_raw, (int, float))
            and isinstance(shares_raw, (int, float))
            and shares_raw > 0
        ):
            price_raw = mcap_raw / shares_raw

        if isinstance(price_raw, (int, float)) and price_raw > 0:
            base["current_price"] = round(float(price_raw), 2)

        if mcap_raw:
            base["market_cap"] = _format_money(mcap_raw)

        per_raw = info.get("trailingPE") or info.get("forwardPE")
        if isinstance(per_raw, (int, float)):
            base["per"] = round(float(per_raw), 2)

        ev_ebitda_raw = info.get("enterpriseToEbitda")
        if isinstance(ev_ebitda_raw, (int, float)):
            base["ev_ebitda"] = round(float(ev_ebitda_raw), 2)

        fcf_raw = info.get("freeCashflow")
        op_margin_raw = info.get("operatingMargins")
        if isinstance(op_margin_raw, (int, float)):
            base["gaap_opm"] = round(float(op_margin_raw) * 100, 1)

        rev_growth = info.get("revenueGrowth")
        if isinstance(rev_growth, (int, float)):
            base["rev_growth_pct"] = round(float(rev_growth) * 100, 1)

        total_debt = info.get("totalDebt") or 0
        total_cash = info.get("totalCash") or 0
        ebitda_raw = info.get("ebitda")

        sbc_raw = 0.0
        try:
            cf = tk.cashflow
            if cf is not None and not cf.empty:
                for label in ["Stock Based Compensation", "StockBasedCompensation"]:
                    if label in cf.index:
                        val = cf.loc[label].dropna()
                        if not val.empty:
                            sbc_raw = float(val.iloc[0])
                        break
        except Exception:
            sbc_raw = 0.0

        if sbc_raw:
            base["sbc_str"] = _format_money(sbc_raw)

        if isinstance(fcf_raw, (int, float)):
            base["fcf_str"] = _format_money(fcf_raw)
            oe_raw = fcf_raw - sbc_raw
            base["oe_str"] = _format_money(oe_raw)
            if isinstance(mcap_raw, (int, float)) and mcap_raw > 0:
                base["fcf_yield"] = round((fcf_raw / mcap_raw) * 100, 2)
                base["oe_yield"] = round((oe_raw / mcap_raw) * 100, 2)

        if total_debt or total_cash:
            net_debt = total_debt - total_cash
            base["net_debt_str"] = _format_money(net_debt)
            if isinstance(ebitda_raw, (int, float)) and ebitda_raw > 0:
                base["net_debt_ebitda"] = round(net_debt / ebitda_raw, 2)
    except Exception:
        pass

    # yfinance에서 핵심 재무가 N/A로 빠졌으면 Finviz 백업으로 보충
    if base["market_cap"] == "N/A" or base["gaap_opm"] is None or base["current_price"] <= 0:
        fv = _fetch_finviz_fundamentals(ticker)
        if base["current_price"] <= 0 and fv.get("current_price", 0) > 0:
            base["current_price"] = fv["current_price"]
        if base["market_cap"] == "N/A" and fv.get("market_cap"):
            base["market_cap"] = fv["market_cap"]
        if base["per"] is None and fv.get("per") is not None:
            base["per"] = fv["per"]
        if base["gaap_opm"] is None and fv.get("gaap_opm") is not None:
            base["gaap_opm"] = fv["gaap_opm"]
        if base["rev_growth_pct"] is None and fv.get("rev_growth_pct") is not None:
            base["rev_growth_pct"] = fv["rev_growth_pct"]

    return base


def _fetch_fallback_price(ticker: str) -> tuple[float, float]:
    """curl_cffi 브라우저 위장으로 Yahoo v8 Chart -> Finviz -> Google Finance 순으로 실제 종가 조회."""
    try:
        from curl_cffi import requests as cffi_requests
        s = cffi_requests.Session(impersonate="chrome")

        # 1차: Yahoo v8 Chart (curl_cffi Chrome TLS 위장)
        try:
            r = s.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?range=5d&interval=1d", timeout=8)
            if r.status_code == 200:
                res = r.json()["chart"]["result"][0]
                meta_price = res["meta"].get("regularMarketPrice")
                closes = [c for c in res["indicators"]["quote"][0]["close"] if c is not None]
                curr = float(meta_price) if meta_price else (float(closes[-1]) if closes else 0.0)
                prev = float(closes[0]) if len(closes) > 1 else curr
                chg = ((curr - prev) / prev * 100) if prev else 0.0
                if curr > 0:
                    return round(curr, 2), round(chg, 2)
        except Exception:
            pass

        # 2차: Finviz 스크리너 HTML
        if "." not in ticker:
            try:
                r = s.get(f"https://finviz.com/quote.ashx?t={ticker}", timeout=8)
                if r.status_code == 200:
                    m_price = re.search(r'class="quote-price_wrapper_price"[^>]*>([0-9,.]+)<', r.text)
                    if not m_price:
                        m_price = re.search(r'"price"\s*:\s*([0-9.]+)', r.text)
                    if m_price:
                        curr = float(m_price.group(1).replace(",", ""))
                        if curr > 0:
                            return round(curr, 2), 0.0
            except Exception:
                pass

        # 3차: Google Finance HTML
        if "." not in ticker:
            for exch in ["NASDAQ", "NYSE"]:
                try:
                    r = s.get(f"https://www.google.com/finance/quote/{ticker}:{exch}", timeout=8)
                    if r.status_code == 200:
                        m_price = re.search(r'data-last-price="([0-9.]+)"', r.text)
                        if m_price:
                            curr = float(m_price.group(1))
                            if curr > 0:
                                return round(curr, 2), 0.0
                except Exception:
                    continue
    except Exception:
        pass

    return 0.0, 0.0


def _to_float_or_none(val):
    if val is None or isinstance(val, bool):
        return None
    if isinstance(val, (int, float)):
        return round(float(val), 2)
    if isinstance(val, str):
        cleaned = val.replace("%", "").replace("x", "").replace("$", "").replace(",", "").strip()
        try:
            return round(float(cleaned), 2)
        except Exception:
            return None
    return None


def _extract_json(text: str) -> dict:
    cleaned = text.strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start:end + 1]
    return json.loads(cleaned)


def run_ai_pipeline_once(
    limit: int = 20,
    target_sector: str | None = None,
    specific_tickers: list[str] | None = None,
    progress_callback=None,
) -> int:
    """Owner's Earnings(FCF-SBC) & 7단계 기관 리서치 프레임워크 기반 파이프라인."""
    init_db()
    api_key = _load_gemini_key()
    if not api_key:
        raise ValueError("GEMINI_API_KEY가 설정되지 않았습니다. Streamlit Settings > Secrets 또는 .env 파일을 확인하세요.")

    universe = get_target_universe()
    if specific_tickers:
        universe = universe[universe["ticker"].isin(specific_tickers)]
    elif target_sector and target_sector != "전체 (All Sectors)":
        universe = universe[universe["sector"] == target_sector]

    target_df = universe.head(limit).copy()
    if target_df.empty:
        return 0

    tickers = target_df["ticker"].tolist()
    if progress_callback:
        progress_callback(0, len(tickers), f"📡 {len(tickers)}개 종목 시장 시세 일괄 수집 중...")

    price_map = _fetch_batch_closes(tickers)
    client = genai.Client(api_key=api_key)
    candidate_models = _get_candidate_models(client)

    saved_count = 0
    total = len(target_df)

    for idx, (_, row) in enumerate(target_df.iterrows(), start=1):
        ticker = row["ticker"]
        company = row["company"]
        sector = row["sector"]

        if progress_callback:
            progress_callback(idx - 1, total, f"🤖 [{idx}/{total}] {company} ({ticker}) OE·재무 추출 및 DCF 분석 중...")

        p_info = price_map.get(ticker, {"close": 0.0, "chg_5d": 0.0})
        close_price = p_info["close"]
        chg_5d = p_info["chg_5d"]
        fund = _fetch_fundamentals(ticker)

        if fund.get("current_price", 0.0) > 0:
            close_price = fund["current_price"]

        if close_price <= 0:
            fb_price, fb_chg = _fetch_fallback_price(ticker)
            if fb_price > 0:
                close_price = fb_price
                chg_5d = fb_chg

        user_prompt = f"""
분석 대상 기업: {company} (티커: {ticker})
S&P GICS 공식 섹터 분류: {sector}
최근 종가 (액면분할 반영 실제 시장가): ${close_price} (최근 5거래일 등락률: {chg_5d:+.2f}%)
※ 중요: 현재 실제 주가는 ${close_price}입니다. N/A로 표시된 재무 지표(시가총액, GAAP OPM, FCF Yield, OE Yield, EV/EBITDA, 순부채/EBITDA)는 반드시 Google Search로 최신 실적·재무제표를 조회해 JSON 필드에 채워 넣으십시오.

[1차 수집된 재무 · 현금흐름 · Owner's Earnings 팩트]
- 시가총액: {fund['market_cap']}
- 보고 FCF (잉여현금흐름): {fund['fcf_str']} (FCF Yield: {f"{fund['fcf_yield']}%" if fund['fcf_yield'] is not None else 'N/A'})
- 주식보상비용 (SBC): {fund['sbc_str']}
- Owner's Earnings 대리값 (FCF - SBC): {fund['oe_str']} (OE Yield: {f"{fund['oe_yield']}%" if fund['oe_yield'] is not None else 'N/A'})
- GAAP 영업이익률 (OPM): {f"{fund['gaap_opm']}%" if fund['gaap_opm'] is not None else 'N/A'}
- 매출 성장률 (YoY): {f"{fund['rev_growth_pct']}%" if fund['rev_growth_pct'] is not None else 'N/A'}
- PER: {fund['per'] if fund['per'] is not None else 'N/A'} | EV/EBITDA: {fund['ev_ebitda'] if fund['ev_ebitda'] is not None else 'N/A'}
- 순부채 (음수면 순현금): {fund['net_debt_str']} (순부채/EBITDA: {f"{fund['net_debt_ebitda']}x" if fund['net_debt_ebitda'] is not None else 'N/A'})

위 팩트와 웹 검색 결과를 바탕으로 누락된 재무 지표 보완, OE DCF 또는 업종 대체 멀티플 선택, Bear/Base/Bull 목표주가와 7단계 리서치 근거를 JSON으로 출력하세요.
"""
        try:
            res = None
            last_err = None
            for m_try in candidate_models:
                try:
                    res = client.models.generate_content(
                        model=m_try,
                        contents=user_prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=SYSTEM_ANALYST_PROMPT,
                            tools=[{"google_search": {}}],
                            temperature=0.2,
                        ),
                    )
                    if res and res.text:
                        break
                except Exception as api_e:
                    last_err = api_e
                    if close_price > 0:
                        try:
                            res = client.models.generate_content(
                                model=m_try,
                                contents=user_prompt,
                                config=types.GenerateContentConfig(
                                    system_instruction=SYSTEM_ANALYST_PROMPT,
                                    response_mime_type="application/json",
                                    temperature=0.2,
                                ),
                            )
                            if res and res.text:
                                break
                        except Exception as e2:
                            last_err = e2
                    time.sleep(1.5)
                    continue

            if not res or not res.text:
                raise last_err or RuntimeError("사용 가능한 Flash 모델 할당량이 일시 소진되었거나 Google Search가 차단되었습니다.")

            parsed = _extract_json(res.text)

            if close_price <= 0:
                searched_price = parsed.get("actual_close_price") or parsed.get("close_price")
                if isinstance(searched_price, (int, float)) and searched_price > 0:
                    close_price = round(float(searched_price), 2)

            if close_price <= 0:
                raise RuntimeError(f"{company} ({ticker})의 실제 주가 조회에 실패했습니다(0.00). 잠시 후 다시 시도해 주세요.")

            # API/Finviz에서 N/A로 빠진 재무 지표는 AI 웹 검색 결과로 자동 보충
            final_mcap = fund["market_cap"]
            if (not final_mcap or final_mcap == "N/A") and parsed.get("market_cap"):
                m_str = str(parsed["market_cap"]).strip()
                if m_str and m_str != "N/A":
                    final_mcap = m_str

            final_per = fund["per"] if fund["per"] is not None else _to_float_or_none(parsed.get("per"))
            final_ev_ebitda = fund["ev_ebitda"] if fund["ev_ebitda"] is not None else _to_float_or_none(parsed.get("ev_ebitda"))
            final_fcf_yield = fund["fcf_yield"] if fund["fcf_yield"] is not None else _to_float_or_none(parsed.get("fcf_yield"))
            final_oe_yield = fund["oe_yield"] if fund["oe_yield"] is not None else _to_float_or_none(parsed.get("oe_yield"))
            final_gaap_opm = fund["gaap_opm"] if fund["gaap_opm"] is not None else _to_float_or_none(parsed.get("gaap_opm"))
            final_net_debt_ebitda = (
                fund["net_debt_ebitda"]
                if fund["net_debt_ebitda"] is not None
                else _to_float_or_none(parsed.get("net_debt_ebitda"))
            )

            raw_score = parsed.get("score")
            score_val = int(raw_score) if isinstance(raw_score, (int, float)) and raw_score > 0 else 65

            bear_t = float(parsed.get("bear_target") or round(close_price * 0.85, 2))
            base_t = float(parsed.get("base_target") or round(close_price * 1.12, 2))
            bull_t = float(parsed.get("bull_target") or round(close_price * 1.35, 2))

            if close_price > 0 and base_t > close_price * 4:
                ratio = round(base_t / close_price)
                if ratio >= 5:
                    bear_t = round(bear_t / ratio, 2)
                    base_t = round(base_t / ratio, 2)
                    bull_t = round(bull_t / ratio, 2)

            upside_pct = round(((base_t - close_price) / close_price * 100), 2)

            insert_analysis({
                "company": company,
                "ticker": ticker,
                "sector": sector,
                "close_price": close_price,
                "score": score_val,
                "opinion": parsed.get("opinion", "중립"),
                "valuation_metric": parsed.get("valuation_metric", ""),
                "summary": parsed.get("summary", ""),
                "reasoning": parsed.get("reasoning", ""),
                "market_cap": final_mcap,
                "per": final_per,
                "ev_ebitda": final_ev_ebitda,
                "fcf_yield": final_fcf_yield,
                "oe_yield": final_oe_yield,
                "gaap_opm": final_gaap_opm,
                "net_debt_ebitda": final_net_debt_ebitda,
                "dcf_assumptions": parsed.get("dcf_assumptions", ""),
                "normalization_note": parsed.get("normalization_note", ""),
                "bear_target": round(bear_t, 2),
                "base_target": round(base_t, 2),
                "bull_target": round(bull_t, 2),
                "upside_pct": upside_pct,
                "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })
            saved_count += 1
        except Exception as e:
            err_msg = f"[{company} ({ticker}) 분석 실패] {type(e).__name__}: {e}"
            print(err_msg, flush=True)
            raise RuntimeError(err_msg)

        if idx < total:
            time.sleep(API_SLEEP_SEC)

    if progress_callback:
        progress_callback(total, total, f"✅ 완료! 총 {saved_count}개 종목 OE 리서치 DB 적재 완료")

    return saved_count


if __name__ == "__main__":
    print("=== 500+ 전 종목 Owner's Earnings & 7단계 AI 리서치 파이프라인 시작 ===")
    run_ai_pipeline_once(limit=550)
    scheduler = BlockingScheduler(timezone="Asia/Seoul")
    scheduler.add_job(lambda: run_ai_pipeline_once(limit=550), "cron", hour=16, minute=30)
    scheduler.start()