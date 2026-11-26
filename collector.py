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

# SEC EDGAR 공식 API는 User-Agent에 앱 이름과 연락처 이메일 표기를 요구함
SEC_HEADERS = {
    "User-Agent": "InstitutionalResearchApp/1.0 (contact@research-app.local)",
    "Accept": "application/json",
}

_CIK_CACHE: dict[str, str] = {}

SYSTEM_ANALYST_PROMPT = """
당신은 2026년 기관투자자 기준의 수석 주식 리서치 애널리스트입니다.
제공된 실제 SEC 공시 기반 재무 팩트(Owner's Earnings = FCF - SBC, GAAP OPM, EV/EBITDA, 순부채/EBITDA, 매출성장률 등)와 실시간 웹 검색(Google Search)을 바탕으로 아래 7단계 리서치 원칙에 따라 엄격히 평가하십시오.
절대 재무 수치를 임의로 지어내지 말고, 제공된 팩트와 실제 주가를 기준으로 목표주가를 산출하십시오.

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
   - 반드시 제공된 액면분할 반영 현재 실제 주가(actual_close_price)를 기준으로 bear_target <= base_target <= bull_target 순서를 준수하십시오.

반드시 아래 JSON 스키마로만 응답하십시오:
{
  "actual_close_price": 액면분할이 반영된 2026년 현재 실제 주가 숫자(소수점 2자리),
  "score": 1~100 사이 정수,
  "opinion": "강력매수" | "매수" | "중립" | "매도",
  "valuation_metric": "주 평가법 및 핵심 배수 (예: OE DCF [성장 15%/할인 9.5%/g 3.0%] 또는 EV/EBITDA 5.25x)",
  "dcf_assumptions": "시나리오 가정 요약 (예: Bear 10%/10.5%/2.5% | Base 20%/9.5%/3.0% | Bull 25%/9.0%/3.5% 또는 EV/EBITDA 4.5x/5.25x/5.75x)",
  "normalization_note": "SEC 공시 기준 GAAP OPM, SBC 차감, 순부채/순현금, GICS 섹터 분류 등 회계·정상화 주의사항 1줄 요약",
  "bear_target": 보수적 목표주가 숫자(소수점 2자리),
  "base_target": 기준 목표주가 숫자(소수점 2자리),
  "bull_target": 낙관적 목표주가 숫자(소수점 2자리),
  "summary": "핵심 투자 결론 한 줄 요약 (55자 이내)",
  "reasoning": "[사업·해자] / [재무·OE·부채] / [밸류에이션·관찰지표] 3가지 관점에서 제공된 실제 수치를 인용해 4~5문장으로 작성"
}
"""


def _get_candidate_models(client) -> list[str]:
    """무료 등급에서 할당량이 넉넉한 3.x Flash / Flash-Lite 계열만 반환."""
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


def _get_sec_cik(ticker: str) -> str | None:
    """SEC 공식 ticker-to-CIK 매핑 테이블 조회."""
    clean_t = ticker.upper().replace("-", ".")
    if clean_t in _CIK_CACHE:
        return _CIK_CACHE[clean_t]
    try:
        req = urllib.request.Request("https://www.sec.gov/files/company_tickers.json", headers=SEC_HEADERS)
        with urllib.request.urlopen(req, timeout=10) as resp:
            mapping = json.loads(resp.read().decode("utf-8"))
            for item in mapping.values():
                t_sym = str(item.get("ticker", "")).upper()
                cik_str = str(item.get("cik_str", "")).zfill(10)
                _CIK_CACHE[t_sym] = cik_str
        return _CIK_CACHE.get(clean_t)
    except Exception:
        return None


def _extract_latest_sec_fact(facts_us_gaap: dict, tags: list[str], annual_only: bool = True) -> float | None:
    """SEC XBRL companyfacts에서 최신 연간(10-K) 또는 최신 시점(10-Q/10-K) 수치를 정확히 추출."""
    for tag in tags:
        node = facts_us_gaap.get(tag)
        if not node:
            continue
        units = node.get("units", {})
        unit_list = units.get("USD") or units.get("shares")
        if not unit_list:
            continue

        # 10-K(연간) 우선, 시점(Instant) 항목은 10-K/10-Q 최신값 사용
        filtered = []
        for entry in unit_list:
            form = entry.get("form", "")
            val = entry.get("val")
            end = entry.get("end", "")
            start = entry.get("start", "")
            if not isinstance(val, (int, float)) or not end:
                continue
            if annual_only:
                if form == "10-K":
                    # 연간 기간(약 350~370일)인지 확인하여 분기값 혼입 방지
                    if start:
                        try:
                            days = (datetime.strptime(end, "%Y-%m-%d") - datetime.strptime(start, "%Y-%m-%d")).days
                            if days < 330:
                                continue
                        except Exception:
                            pass
                    filtered.append((end, entry.get("filed", ""), float(val)))
            else:
                if form in ("10-K", "10-Q"):
                    filtered.append((end, entry.get("filed", ""), float(val)))

        if filtered:
            filtered.sort(key=lambda x: (x[0], x[1]), reverse=True)
            return filtered[0][2]
    return None


def _fetch_sec_edgar_fundamentals(ticker: str, current_price: float) -> dict:
    """미국 SEC EDGAR 공식 10-K/10-Q XBRL 공시 데이터로 GAAP OPM, FCF, SBC, OE, 순부채 직접 계산."""
    result = {}
    if "." in ticker:
        return result

    cik = _get_sec_cik(ticker)
    if not cik:
        return result

    try:
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        req = urllib.request.Request(url, headers=SEC_HEADERS)
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        gaap = data.get("facts", {}).get("us-gaap", {})
        dei = data.get("facts", {}).get("dei", {})

        rev = _extract_latest_sec_fact(gaap, [
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "Revenues",
            "SalesRevenueNet",
        ], annual_only=True)

        op_inc = _extract_latest_sec_fact(gaap, [
            "OperatingIncomeLoss",
        ], annual_only=True)

        ocf = _extract_latest_sec_fact(gaap, [
            "NetCashProvidedByUsedInOperatingActivities",
        ], annual_only=True)

        capex = _extract_latest_sec_fact(gaap, [
            "PaymentsToAcquirePropertyPlantAndEquipment",
            "PaymentsToAcquireProductiveAssets",
        ], annual_only=True)

        sbc = _extract_latest_sec_fact(gaap, [
            "ShareBasedCompensation",
            "AllocatedShareBasedCompensationExpense",
        ], annual_only=True) or 0.0

        da = _extract_latest_sec_fact(gaap, [
            "DepreciationDepletionAndAmortization",
            "DepreciationAndAmortization",
        ], annual_only=True) or 0.0

        cash = _extract_latest_sec_fact(gaap, [
            "CashAndCashEquivalentsAtCarryingValue",
            "CashCashEquivalentsAndShortTermInvestments",
        ], annual_only=False) or 0.0

        lt_debt = _extract_latest_sec_fact(gaap, [
            "LongTermDebtNoncurrent",
            "LongTermDebt",
        ], annual_only=False) or 0.0

        st_debt = _extract_latest_sec_fact(gaap, [
            "LongTermDebtCurrent",
            "DebtCurrent",
            "ShortTermBorrowings",
        ], annual_only=False) or 0.0

        shares = _extract_latest_sec_fact(dei, [
            "EntityCommonStockSharesOutstanding",
        ], annual_only=False) or _extract_latest_sec_fact(gaap, [
            "CommonStockSharesOutstanding",
            "WeightedAverageNumberOfDilutedSharesOutstanding",
        ], annual_only=False)

        net_inc = _extract_latest_sec_fact(gaap, [
            "NetIncomeLoss",
        ], annual_only=True)

        if rev and op_inc and rev > 0:
            result["gaap_opm"] = round((op_inc / rev) * 100, 1)

        mcap = None
        if shares and shares > 0 and current_price > 0:
            mcap = shares * current_price
            result["market_cap"] = _format_money(mcap)

        if mcap and net_inc and net_inc > 0:
            result["per"] = round(mcap / net_inc, 2)

        fcf = None
        if ocf is not None:
            fcf = ocf - abs(capex or 0.0)
            result["fcf_str"] = _format_money(fcf)
            result["sbc_str"] = _format_money(sbc) if sbc > 0 else "$0"
            oe = fcf - sbc
            result["oe_str"] = _format_money(oe)
            if mcap and mcap > 0:
                result["fcf_yield"] = round((fcf / mcap) * 100, 2)
                result["oe_yield"] = round((oe / mcap) * 100, 2)

        total_debt = lt_debt + st_debt
        net_debt = total_debt - cash
        result["net_debt_str"] = _format_money(net_debt)

        if op_inc is not None:
            ebitda = op_inc + da
            if ebitda > 0:
                result["net_debt_ebitda"] = round(net_debt / ebitda, 2)
                if mcap and mcap > 0:
                    ev = mcap + net_debt
                    result["ev_ebitda"] = round(ev / ebitda, 2)

    except Exception as e:
        print(f"[SEC EDGAR 파싱 참고] {ticker}: {e}", flush=True)

    return result


def _fetch_real_price(ticker: str) -> tuple[float, float]:
    """curl_cffi 브라우저 위장 + yfinance + Google Finance로 실제 거래 종가 조회."""
    # 1차: curl_cffi + Yahoo v8 Chart
    try:
        from curl_cffi import requests as cffi_requests
        s = cffi_requests.Session(impersonate="chrome")
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

    # 2차: Google Finance HTML
    if "." not in ticker:
        try:
            from curl_cffi import requests as cffi_requests
            s = cffi_requests.Session(impersonate="chrome")
            for exch in ["NASDAQ", "NYSE"]:
                r = s.get(f"https://www.google.com/finance/quote/{ticker}:{exch}", timeout=8)
                if r.status_code == 200:
                    m_price = re.search(r'data-last-price="([0-9.]+)"', r.text)
                    if m_price:
                        curr = float(m_price.group(1))
                        if curr > 0:
                            return round(curr, 2), 0.0
        except Exception:
            pass

    # 3차: yfinance 기본 다운로드
    try:
        df = yf.download(ticker, period="5d", progress=False)
        if not df.empty and "Close" in df.columns:
            closes = df["Close"].dropna()
            if not closes.empty:
                curr = float(closes.iloc[-1].item() if hasattr(closes.iloc[-1], "item") else closes.iloc[-1])
                prev = float(closes.iloc[0].item() if hasattr(closes.iloc[0], "item") else closes.iloc[0])
                chg = ((curr - prev) / prev * 100) if prev else 0.0
                if curr > 0:
                    return round(curr, 2), round(chg, 2)
    except Exception:
        pass

    return 0.0, 0.0


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
    """SEC EDGAR 공식 공시 팩트 + Owner's Earnings(FCF-SBC) & 7단계 기관 리서치 파이프라인."""
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

    client = genai.Client(api_key=api_key)
    candidate_models = _get_candidate_models(client)

    saved_count = 0
    total = len(target_df)

    for idx, (_, row) in enumerate(target_df.iterrows(), start=1):
        ticker = row["ticker"]
        company = row["company"]
        sector = row["sector"]

        if progress_callback:
            progress_callback(idx - 1, total, f"🤖 [{idx}/{total}] {company} ({ticker}) SEC 공식 공시 & 시세 추출 중...")

        close_price, chg_5d = _fetch_real_price(ticker)
        sec_fund = _fetch_sec_edgar_fundamentals(ticker, close_price)

        fund = {
            "market_cap": sec_fund.get("market_cap", "N/A"),
            "per": sec_fund.get("per"),
            "ev_ebitda": sec_fund.get("ev_ebitda"),
            "fcf_str": sec_fund.get("fcf_str", "N/A"),
            "sbc_str": sec_fund.get("sbc_str", "미확인"),
            "oe_str": sec_fund.get("oe_str", "N/A"),
            "fcf_yield": sec_fund.get("fcf_yield"),
            "oe_yield": sec_fund.get("oe_yield"),
            "gaap_opm": sec_fund.get("gaap_opm"),
            "net_debt_str": sec_fund.get("net_debt_str", "N/A"),
            "net_debt_ebitda": sec_fund.get("net_debt_ebitda"),
        }

        user_prompt = f"""
분석 대상 기업: {company} (티커: {ticker})
S&P GICS 공식 섹터 분류: {sector}
최근 종가 (액면분할 반영 실제 시장가): ${close_price} (최근 5거래일 등락률: {chg_5d:+.2f}%)

[미국 SEC EDGAR 공식 10-K/10-Q 공시 기반 실제 재무 · Owner's Earnings 팩트]
- 시가총액: {fund['market_cap']}
- 보고 FCF (영업현금흐름 - CAPEX): {fund['fcf_str']} (FCF Yield: {f"{fund['fcf_yield']}%" if fund['fcf_yield'] is not None else 'N/A'})
- 주식보상비용 (SBC): {fund['sbc_str']}
- Owner's Earnings (FCF - SBC): {fund['oe_str']} (OE Yield: {f"{fund['oe_yield']}%" if fund['oe_yield'] is not None else 'N/A'})
- GAAP 영업이익률 (OPM): {f"{fund['gaap_opm']}%" if fund['gaap_opm'] is not None else 'N/A'}
- PER: {fund['per'] if fund['per'] is not None else 'N/A'} | EV/EBITDA: {fund['ev_ebitda'] if fund['ev_ebitda'] is not None else 'N/A'}
- 순부채 (음수면 순현금): {fund['net_debt_str']} (순부채/EBITDA: {f"{fund['net_debt_ebitda']}x" if fund['net_debt_ebitda'] is not None else 'N/A'})

위 SEC 공식 공시 팩트와 웹 검색 결과를 바탕으로 OE DCF 또는 업종 대체 멀티플(EV/EBITDA 등)을 선택하여 Bear/Base/Bull 목표주가와 7단계 리서치 근거를 JSON으로 출력하세요.
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
                    # 주가를 뒤늦게 찾은 경우 SEC 발행주식수로 시가총액·Yield 재계산
                    sec_fund = _fetch_sec_edgar_fundamentals(ticker, close_price)
                    for k in ["market_cap", "per", "ev_ebitda", "fcf_yield", "oe_yield"]:
                        if sec_fund.get(k) is not None:
                            fund[k] = sec_fund[k]

            if close_price <= 0:
                raise RuntimeError(f"{company} ({ticker})의 실제 주가 조회에 실패했습니다(0.00). 잠시 후 다시 시도해 주세요.")

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
                "market_cap": fund["market_cap"],
                "per": fund["per"],
                "ev_ebitda": fund["ev_ebitda"],
                "fcf_yield": fund["fcf_yield"],
                "oe_yield": fund["oe_yield"],
                "gaap_opm": fund["gaap_opm"],
                "net_debt_ebitda": fund["net_debt_ebitda"],
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
        progress_callback(total, total, f"✅ 완료! 총 {saved_count}개 종목 SEC 공식 공시 기반 OE 리서치 적재 완료")

    return saved_count


if __name__ == "__main__":
    print("=== 500+ 전 종목 Owner's Earnings & 7단계 AI 리서치 파이프라인 시작 ===")
    run_ai_pipeline_once(limit=550)
    scheduler = BlockingScheduler(timezone="Asia/Seoul")
    scheduler.add_job(lambda: run_ai_pipeline_once(limit=550), "cron", hour=16, minute=30)
    scheduler.start()