import os
import json
import time
from datetime import datetime
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from google import genai
from google.genai import types
from apscheduler.schedulers.blocking import BlockingScheduler
from database import init_db, insert_analysis, save_universe, load_universe

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
API_SLEEP_SEC = float(os.getenv("API_SLEEP_SEC", "4.2"))

SYSTEM_ANALYST_PROMPT = """
당신은 2026년 기관투자자 기준의 수석 주식 리서치 애널리스트입니다.
제공된 실제 재무 팩트(Owner's Earnings = FCF - SBC, GAAP OPM, EV/EBITDA, 순부채/EBITDA, 매출성장률 등)를 바탕으로 아래 7단계 리서치 원칙에 따라 엄격히 평가하십시오.

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
   - bear_target <= base_target <= bull_target 순서를 반드시 준수하십시오.

반드시 아래 JSON 스키마로만 응답하십시오:
{
  "score": 0~100 사이 정수,
  "opinion": "강력매수" | "매수" | "중립" | "매도",
  "valuation_metric": "주 평가법 및 핵심 배수 (예: OE DCF [성장 15%/할인 9.5%/g 3.0%] 또는 EV/EBITDA 5.25x)",
  "dcf_assumptions": "시나리오 가정 요약 (예: Bear 10%/10.5%/2.5% | Base 20%/9.5%/3.0% | Bull 25%/9.0%/3.5% 또는 EV/EBITDA 4.5x/5.25x/5.75x)",
  "normalization_note": "SBC 차감, 순부채/순현금, 일회성 항목, GICS 섹터 분류 등 회계·정상화 주의사항 1줄 요약",
  "bear_target": 보수적 목표주가 숫자(소수점 2자리),
  "base_target": 기준 목표주가 숫자(소수점 2자리),
  "bull_target": 낙관적 목표주가 숫자(소수점 2자리),
  "summary": "핵심 투자 결론 한 줄 요약 (55자 이내)",
  "reasoning": "[사업·해자] / [재무·OE·부채] / [밸류에이션·관찰지표] 3가지 관점에서 구체적 수치를 인용해 4~5문장으로 작성"
}
"""


def _get_flash_model(client) -> str:
    preferred = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
    available = [
        m.name.replace("models/", "")
        for m in client.models.list()
        if "generateContent" in (m.supported_actions or [])
        and "flash" in m.name.lower()
        and not any(x in m.name.lower() for x in ["image", "audio", "tts", "live", "embedding"])
    ]
    for p in preferred:
        if p in available:
            return p
    return available[0] if available else "gemini-2.0-flash"


def sync_sp500_universe() -> pd.DataFrame:
    """Wikipedia에서 최신 S&P 500 전 종목(500+)을 크롤링하고 주요 추가 종목을 병합해 DB에 저장."""
    init_db()
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url, storage_options={"User-Agent": "Mozilla/5.0"})
    raw_df = tables[0]

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
            print(f"[유니버스 동기화 경고] {e}")
    return df


def _fetch_batch_closes(tickers: list[str]) -> dict:
    """100종목 단위 배치 다운로드로 500+ 종목 최근 종가 및 5일 등락률 고속 수집."""
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
            print(f"[배치 시세 오류] {e}")
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


def _fetch_fundamentals(ticker: str) -> dict:
    """실제 재무제표·현금흐름표에서 Owner's Earnings(FCF-SBC), GAAP OPM, 순부채/EBITDA 추출."""
    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}

        mcap_raw = info.get("marketCap")
        per_raw = info.get("trailingPE") or info.get("forwardPE")
        ev_ebitda_raw = info.get("enterpriseToEbitda")
        fcf_raw = info.get("freeCashflow")
        op_margin_raw = info.get("operatingMargins")
        rev_growth = info.get("revenueGrowth")
        total_debt = info.get("totalDebt") or 0
        total_cash = info.get("totalCash") or 0
        ebitda_raw = info.get("ebitda")

        # 현금흐름표에서 주식보상비용(SBC) 추출 후 Owner's Earnings(FCF - SBC) 계산
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

        oe_raw = None
        oe_yield = None
        fcf_yield = None
        if isinstance(fcf_raw, (int, float)) and isinstance(mcap_raw, (int, float)) and mcap_raw > 0:
            fcf_yield = round((fcf_raw / mcap_raw) * 100, 2)
            oe_raw = fcf_raw - sbc_raw
            oe_yield = round((oe_raw / mcap_raw) * 100, 2)

        net_debt = total_debt - total_cash
        net_debt_ebitda = None
        if isinstance(ebitda_raw, (int, float)) and ebitda_raw > 0:
            net_debt_ebitda = round(net_debt / ebitda_raw, 2)

        return {
            "market_cap": _format_money(mcap_raw),
            "per": round(float(per_raw), 2) if isinstance(per_raw, (int, float)) else None,
            "ev_ebitda": round(float(ev_ebitda_raw), 2) if isinstance(ev_ebitda_raw, (int, float)) else None,
            "fcf_str": _format_money(fcf_raw),
            "sbc_str": _format_money(sbc_raw) if sbc_raw else "미확인",
            "oe_str": _format_money(oe_raw),
            "fcf_yield": fcf_yield,
            "oe_yield": oe_yield,
            "gaap_opm": round(float(op_margin_raw) * 100, 1) if isinstance(op_margin_raw, (int, float)) else None,
            "rev_growth_pct": round(float(rev_growth) * 100, 1) if isinstance(rev_growth, (int, float)) else None,
            "net_debt_str": _format_money(net_debt),
            "net_debt_ebitda": net_debt_ebitda,
        }
    except Exception:
        return {
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
    if not GEMINI_API_KEY or not GEMINI_API_KEY.startswith("AIza"):
        raise ValueError("GEMINI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

def run_ai_pipeline_once(
    limit: int = 20,
    target_sector: str | None = None,
    specific_tickers: list[str] | None = None,
    progress_callback=None,
) -> int:
    """Owner's Earnings(FCF-SBC) & 7단계 기관 리서치 프레임워크 기반 파이프라인."""
    init_db()
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY가 설정되지 않았습니다.")

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
    client = genai.Client(api_key=GEMINI_API_KEY)
    model_name = _get_flash_model(client)

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

        user_prompt = f"""
분석 대상 기업: {company} (티커: {ticker})
S&P GICS 공식 섹터 분류: {sector}
최근 종가: {close_price} (최근 5거래일 등락률: {chg_5d:+.2f}%)

[실제 재무 · 현금흐름 · Owner's Earnings 팩트]
- 시가총액: {fund['market_cap']}
- 보고 FCF (잉여현금흐름): {fund['fcf_str']} (FCF Yield: {f"{fund['fcf_yield']}%" if fund['fcf_yield'] is not None else 'N/A'})
- 주식보상비용 (SBC): {fund['sbc_str']}
- Owner's Earnings 대리값 (FCF - SBC): {fund['oe_str']} (OE Yield: {f"{fund['oe_yield']}%" if fund['oe_yield'] is not None else 'N/A'})
- GAAP 영업이익률 (OPM): {f"{fund['gaap_opm']}%" if fund['gaap_opm'] is not None else 'N/A'}
- 매출 성장률 (YoY): {f"{fund['rev_growth_pct']}%" if fund['rev_growth_pct'] is not None else 'N/A'}
- PER: {fund['per'] if fund['per'] is not None else 'N/A'} | EV/EBITDA: {fund['ev_ebitda'] if fund['ev_ebitda'] is not None else 'N/A'}
- 순부채 (음수면 순현금): {fund['net_debt_str']} (순부채/EBITDA: {f"{fund['net_debt_ebitda']}x" if fund['net_debt_ebitda'] is not None else 'N/A'})

위 팩트를 바탕으로 OE DCF 또는 업종 대체 멀티플(EV/EBITDA 등)을 선택하여 Bear/Base/Bull 목표주가와 7단계 리서치 근거를 JSON으로 출력하세요.
"""
        try:
            res = client.models.generate_content(
                model=model_name,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_ANALYST_PROMPT,
                    response_mime_type="application/json",
                    temperature=0.2,
                ),
            )
            parsed = json.loads(res.text)
            bear_t = float(parsed.get("bear_target") or round(close_price * 0.85, 2))
            base_t = float(parsed.get("base_target") or round(close_price * 1.10, 2))
            bull_t = float(parsed.get("bull_target") or round(close_price * 1.30, 2))
            upside_pct = round(((base_t - close_price) / close_price * 100), 2) if close_price > 0 else 0.0

            insert_analysis({
                "company": company,
                "ticker": ticker,
                "sector": sector,
                "close_price": close_price,
                "score": int(parsed.get("score", 50)),
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
            print(f"[{company} 분석 실패] {e}")

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