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

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
API_SLEEP_SEC = float(os.getenv("API_SLEEP_SEC", "4.2"))

SYSTEM_ANALYST_PROMPT = """
당신은 글로벌 기관투자자(바이사이드/셀사이드) 수준의 수석 주식 리서치 애널리스트입니다.
제공된 실제 재무 지표(PER, EV/EBITDA, FCF Yield, 시가총액, 매출성장률, 부채비율)와 최근 주가 흐름을 바탕으로 종목을 냉철하게 평가하고 적정주가 시나리오를 산출하십시오.

[핵심 분석 및 스코어링 철학]
1. 업종별 최적 밸류에이션 멀티플 자동 선택:
   - 자본집약·레버리지 기업(통신/케이블, 미디어/인프라, 유틸리티 등): EV/EBITDA 및 부채 차감 후 잉여현금흐름(FCF Yield) 최우선 적용.
   - 플랫폼·테크·광고/엔터 기업: 주식보상비용(SBC) 차감 후 실질 Owner's Earnings, FCF Yield, ROE 및 네트워크 해자 지속성 평가.
   - 일반 소비재·제조·금융: PER, PBR, 영업이익률 및 자본배분(자사주 매입·배당) 효율성 평가.
2. 엄격한 0~100점 스코어링 기준 (무분별한 고점 금지):
   - 80~100 (강력매수 / Strong Buy): 구조적 해자 + 압도적 현금창출력 + 멀티플 현저한 저평가 + Base 상승여력 25% 이상.
   - 60~79 (매수 / Buy): 견고한 펀더멘털과 양호한 밸류에이션, 합리적 상승여력(10~25%).
   - 40~59 (중립 / Hold): 적정가치 부근이거나 성장 둔화·부채·경쟁 리스크가 상쇄.
   - 0~39 (매도 / Sell): 고평가 멀티플, 현금흐름 악화, 구조적 해자 훼손 또는 과도한 레버리지.
3. 적정주가 시나리오 (Bear / Base / Bull):
   - 현재 주가(close_price)와 제공된 멀티플·성장률을 기준으로 12개월 목표주가를 보수적(bear_target), 기준(base_target), 낙관적(bull_target) 3단계 숫자로 산출하십시오.
   - bear_target <= base_target <= bull_target 논리적 순서를 반드시 지키십시오.

반드시 아래 JSON 스키마로만 응답하십시오:
{
  "score": 0~100 사이 정수,
  "opinion": "강력매수" | "매수" | "중립" | "매도",
  "valuation_metric": "적용 핵심 멀티플 (예: EV/EBITDA 7.2x & FCF Yield 9.1%)",
  "bear_target": 보수적 목표주가 숫자(소수점 2자리),
  "base_target": 기준 목표주가 숫자(소수점 2자리),
  "bull_target": 낙관적 목표주가 숫자(소수점 2자리),
  "summary": "핵심 투자 포인트 한 줄 요약 (50자 이내)",
  "reasoning": "1) 업종·멀티플 선정 근거 2) 재무 팩트 기반 긍정 요인(해자·현금흐름·자본배분) 3) 핵심 리스크 요인을 구체적 수치와 함께 3~4문장으로 서술"
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
    """Wikipedia에서 최신 S&P 500 전 종목(500+)을 크롤링하고 한국 대표주를 병합해 DB에 저장."""
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


def _format_mcap(mcap_val) -> str:
    if not mcap_val or not isinstance(mcap_val, (int, float)):
        return "N/A"
    if mcap_val >= 1e12:
        return f"${mcap_val / 1e12:.2f}T"
    if mcap_val >= 1e9:
        return f"${mcap_val / 1e9:.1f}B"
    if mcap_val >= 1e6:
        return f"${mcap_val / 1e6:.0f}M"
    return str(mcap_val)


def _fetch_fundamentals(ticker: str) -> dict:
    """개별 종목의 실제 밸류에이션 멀티플 및 현금흐름·성장률 재무 팩트 추출."""
    try:
        info = yf.Ticker(ticker).info or {}
        mcap_raw = info.get("marketCap")
        per_raw = info.get("trailingPE") or info.get("forwardPE")
        ev_ebitda_raw = info.get("enterpriseToEbitda")
        fcf_raw = info.get("freeCashflow")
        rev_growth = info.get("revenueGrowth")
        debt_to_eq = info.get("debtToEquity")

        fcf_yield = None
        if isinstance(fcf_raw, (int, float)) and isinstance(mcap_raw, (int, float)) and mcap_raw > 0:
            fcf_yield = round((fcf_raw / mcap_raw) * 100, 2)

        return {
            "market_cap": _format_mcap(mcap_raw),
            "per": round(float(per_raw), 2) if isinstance(per_raw, (int, float)) else None,
            "ev_ebitda": round(float(ev_ebitda_raw), 2) if isinstance(ev_ebitda_raw, (int, float)) else None,
            "fcf_yield": fcf_yield,
            "rev_growth_pct": round(float(rev_growth) * 100, 1) if isinstance(rev_growth, (int, float)) else None,
            "debt_to_equity": round(float(debt_to_eq), 1) if isinstance(debt_to_eq, (int, float)) else None,
        }
    except Exception:
        return {
            "market_cap": "N/A",
            "per": None,
            "ev_ebitda": None,
            "fcf_yield": None,
            "rev_growth_pct": None,
            "debt_to_equity": None,
        }


def run_ai_pipeline_once(
    limit: int = 20,
    target_sector: str | None = None,
    specific_tickers: list[str] | None = None,
    progress_callback=None,
) -> int:
    """재무 팩트 기반 AI 애널리스트 분석 및 Bear/Base/Bull 목표주가 산출 파이프라인."""
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
            progress_callback(idx - 1, total, f"🤖 [{idx}/{total}] {company} ({ticker}) 재무 추출 및 AI 분석 중...")

        p_info = price_map.get(ticker, {"close": 0.0, "chg_5d": 0.0})
        close_price = p_info["close"]
        chg_5d = p_info["chg_5d"]
        fund = _fetch_fundamentals(ticker)

        user_prompt = f"""
분석 대상 기업: {company} (티커: {ticker})
소속 섹터: {sector}
최근 종가: {close_price} (최근 5거래일 등락률: {chg_5d:+.2f}%)

[실제 재무 및 밸류에이션 팩트]
- 시가총액: {fund['market_cap']}
- PER: {fund['per'] if fund['per'] is not None else '데이터 없음'}
- EV/EBITDA: {fund['ev_ebitda'] if fund['ev_ebitda'] is not None else '데이터 없음'}
- FCF Yield (잉여현금흐름 수익률): {f"{fund['fcf_yield']}%" if fund['fcf_yield'] is not None else '데이터 없음'}
- 매출 성장률(YoY): {f"{fund['rev_growth_pct']}%" if fund['rev_growth_pct'] is not None else '데이터 없음'}
- 부채비율(Debt/Equity): {f"{fund['debt_to_equity']}%" if fund['debt_to_equity'] is not None else '데이터 없음'}

위 재무 팩트와 업종 특성을 반영해 AI 점수, 투자의견, Bear/Base/Bull 목표주가, 핵심 근거를 JSON으로 출력하세요.
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
        progress_callback(total, total, f"✅ 완료! 총 {saved_count}개 종목 리서치 DB 적재 완료")

    return saved_count


if __name__ == "__main__":
    print("=== 500+ 전 종목 재무 기반 AI 리서치 파이프라인 시작 ===")
    run_ai_pipeline_once(limit=550)
    scheduler = BlockingScheduler(timezone="Asia/Seoul")
    scheduler.add_job(lambda: run_ai_pipeline_once(limit=550), "cron", hour=16, minute=30)
    scheduler.start()