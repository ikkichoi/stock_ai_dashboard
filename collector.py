# -*- coding: utf-8 -*-
import os
import time
import json
import pandas as pd
import yfinance as yf
from google import genai
from google.genai import types
from apscheduler.schedulers.blocking import BlockingScheduler
from dotenv import load_dotenv
from database import init_db, save_analysis, save_universe, load_universe

load_dotenv()
init_db()

# 한 번에 분석할 기본 종목 수 (대시보드 버튼 테스트 시 20~50개 권장, 야간 배치 시 550으로 전체 실행)
DEFAULT_BATCH_LIMIT = int(os.getenv("MAX_STOCKS_PER_RUN", "550"))
API_SLEEP_SEC = float(os.getenv("API_SLEEP_SEC", "4.2"))  # 무료 티어(15 RPM) 방어용 기본 4.2초

# 사용자 투자 철학 및 분석 프레임워크 (범용 프롬프트 + Communication 섹터 리서치 감각 반영)
SYSTEM_ANALYST_PROMPT = """
너는 기관급 주식 리서치 애널리스트다. 주어진 종목의 업종·비즈니스 모델·최근 주가 흐름을 바탕으로 정량·정성 평가를 수행하고, 반드시 지정된 JSON 형식으로만 응답하라.

[1. 업종별 핵심 밸류에이션 멀티플 선택 원칙]
모든 기업에 같은 지표를 쓰지 말고 비즈니스 모델과 회계 구조에 맞는 핵심 멀티플(valuation_metric)을 먼저 확정하라:
- Asset-Light 독점·구독형 플랫폼 및 성숙기 저부채 소비재: PER, FCF Yield
- 통신·케이블·인프라·중후장대·M&A 빈번 기업(감가상각 및 이자 왜곡 제거): EV/EBITDA, Net Debt/EBITDA
- AI·클라우드·네트워크 대규모 증설기 및 수주형 기업: Owner's Earnings(순이익+감가상각-유지CAPEX-SBC), FCF 전환율
- 적자/초기 고성장 SaaS·바이오·플랫폼: PSR, EV/GMV
- 은행·보험·증권 등 전통 금융업: PBR-ROE, CET1/NIM

[2. 애널리스트 핵심 평가 관점 (투자 철학 반영)]
- 자본배분과 주주환원: 단순 회계상 순이익보다 주식보상(SBC)을 차감한 실질 Owner's Earnings와 자사주 매입·소각의 복리 효과를 높이 평가하라. 반면 부채 레버리지가 과도해 경기 침체/고금리 시 주주가치를 훼손할 위험이 크면 감점하라.
- 광고·플랫폼·미디어 해자: OS·규제 변화에도 작동하는 1st-party 데이터 기반 타겟팅 효율(ROAS), 인플레이션 가격 전가력(저가 광고형 요금제의 마진 레버리지 등), B2C 대비 B2B 확장성과 해자 지속성을 점검하라.
- 통신·인프라 방어력: 망 고도화(CAPEX 피크) 이후 FCF 급반등 여부, 유무선 결합(Lock-in)을 통한 무선(5G FWA) 침투 방어력, 정부 규제에 따른 가격 인상 제약을 입체적으로 반영하라.
- 사실과 추정의 구분: 확인되지 않은 최신 실적을 지어내지 말고, 확인된 구조적 펀더멘털과 불확실성(리스크)을 명확히 구분해 점수에 반영하라.

[3. 점수(score) 산출 기준 (-100 ~ +100 정수)]
- +60 ~ +100 (강력매수/매수우세): 확실한 해자 + 매력적 밸류에이션/주주환원 + 펀더멘털 개선 촉매
- +20 ~ +59 (완만한 매수우세): 사업 경쟁력은 우수하나 일부 성장 둔화·밸류에이션 부담 공존
- -19 ~ +19 (관망): 호재와 악재(부채, 경쟁 심화, 규제 등)가 팽팽히 맞섬
- -60 ~ -20 (매도우세): 구조적 해자 약화, 마진 압박, 높은 레버리지 리스크
- -100 ~ -61 (강력매도): 심각한 재무 위험 또는 비즈니스 모델 훼손

[출력 JSON 스키마]
{
  "score": -100에서 100 사이의 정수,
  "opinion": "매수우세" 또는 "관망" 또는 "매도우세",
  "valuation_metric": "이 기업에 가장 적합한 핵심 밸류에이션 지표 1~2개 (예: EV/EBITDA & FCF Yield)",
  "summary": "핵심 투자 논지와 결론을 담은 1~2문장 요약",
  "reasoning": "왜 이 점수가 나왔는지 구체적 산출 근거 (1) 업종 및 핵심 멀티플 선정 이유, (2) 긍정 요인(해자·자본배분·성장동력), (3) 감점/리스크 요인(부채·경쟁·규제·밸류에이션 부담)을 조목조목 한국어로 설명"
}
"""

def sync_sp500_universe() -> pd.DataFrame:
    """Wikipedia에서 S&P 500 전 종목(500개 이상) 티커·기업명·섹터 자동 수집 후 DB 저장"""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url, storage_options={"User-Agent": "Mozilla/5.0"})
    raw_df = tables[0]
    universe_df = pd.DataFrame({
        "ticker": raw_df["Symbol"].astype(str).str.replace(".", "-", regex=False).str.strip(),
        "company": raw_df["Security"].astype(str).str.strip(),
        "sector": raw_df["GICS Sector"].astype(str).str.strip(),
        "sub_industry": raw_df["GICS Sub-Industry"].astype(str).str.strip(),
    })
    # 국내 대표주도 함께 포함하고 싶다면 추가
    extra = pd.DataFrame([
        {"ticker": "000660.KS", "company": "SK하이닉스", "sector": "Information Technology", "sub_industry": "Semiconductors"},
        {"ticker": "005930.KS", "company": "삼성전자", "sector": "Information Technology", "sub_industry": "Semiconductors"},
    ])
    universe_df = pd.concat([extra, universe_df], ignore_index=True).drop_duplicates(subset=["ticker"])
    save_universe(universe_df)
    return universe_df

def get_target_universe() -> pd.DataFrame:
    df = load_universe()
    if len(df) < 500:
        try:
            df = sync_sp500_universe()
        except Exception as e:
            print(f"[경고] S&P 500 웹 동기화 실패, 기본 종목으로 대체: {e}")
            if df.empty:
                df = pd.DataFrame([
                    {"ticker": "CHTR", "company": "Charter Communications", "sector": "Communication Services", "sub_industry": "Cable & Satellite"},
                    {"ticker": "NFLX", "company": "Netflix", "sector": "Communication Services", "sub_industry": "Movies & Entertainment"},
                    {"ticker": "OMC", "company": "Omnicom Group", "sector": "Communication Services", "sub_industry": "Advertising"},
                    {"ticker": "RDDT", "company": "Reddit", "sector": "Communication Services", "sub_industry": "Interactive Media"},
                    {"ticker": "APP", "company": "AppLovin", "sector": "Information Technology", "sub_industry": "Application Software"},
                    {"ticker": "ABNB", "company": "Airbnb", "sector": "Consumer Discretionary", "sub_industry": "Hotels, Resorts & Cruise Lines"},
                    {"ticker": "NVDA", "company": "NVIDIA", "sector": "Information Technology", "sub_industry": "Semiconductors"},
                ])
    return df

def _fetch_batch_closes(tickers: list[str]) -> dict[str, tuple[float, float]]:
    """yfinance 배치 다운로드로 500개 종목 최근 종가 및 5일 수익률(%) 고속 수집"""
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
                    sub = data[t] if len(chunk) > 1 else data
                    closes = sub["Close"].dropna()
                    if len(closes) >= 1:
                        latest = float(closes.iloc[-1])
                        prev = float(closes.iloc[0])
                        chg_5d = ((latest - prev) / prev * 100.0) if prev else 0.0
                        price_map[t] = (latest, chg_5d)
                except Exception:
                    continue
        except Exception as e:
            print(f"[시세 배치 경고] {i}~{i+chunk_size}: {e}")
    return price_map

def _get_candidate_models(client) -> list[str]:
    try:
        models = [
            m.name for m in client.models.list()
            if "generateContent" in (m.supported_actions or [])
            and "flash" in m.name.lower()
            and not any(x in m.name.lower() for x in ["image", "audio", "tts", "live", "embedding"])
        ]
        return models if models else ["gemini-2.5-flash", "gemini-2.0-flash"]
    except Exception:
        return ["gemini-2.5-flash", "gemini-2.0-flash"]

def run_ai_pipeline_once(
    limit: int = DEFAULT_BATCH_LIMIT,
    target_sector: str | None = None,
    specific_tickers: list[str] | None = None,
    progress_callback=None
):
    """500+ 종목 일괄 시세 수집 -> Gemini AI 심층 분석(점수+근거) -> SQLite DB 적재"""
    universe_df = get_target_universe()

    if specific_tickers:
        universe_df = universe_df[universe_df["ticker"].isin(specific_tickers)]
    elif target_sector and target_sector != "전체 (All Sectors)":
        universe_df = universe_df[universe_df["sector"] == target_sector]

    if limit and limit > 0:
        universe_df = universe_df.head(limit)

    tickers = universe_df["ticker"].tolist()
    if not tickers:
        return 0

    price_map = _fetch_batch_closes(tickers)

    ai_key = os.getenv("GEMINI_API_KEY", "").strip()
    client = genai.Client(api_key=ai_key) if ai_key else None
    candidate_models = _get_candidate_models(client) if client else []

    total = len(universe_df)
    processed = 0

    for idx, row in enumerate(universe_df.itertuples(), start=1):
        ticker = row.ticker
        company = row.company
        sector = row.sector
        sub_industry = row.sub_industry

        if ticker not in price_map:
            continue
        latest_close, chg_5d = price_map[ticker]

        if progress_callback:
            progress_callback(idx, total, f"[{idx}/{total}] {company} ({ticker}) 분석 중...")

        if client:
            user_prompt = (
                f"분석 대상: {company} ({ticker})\n"
                f"GICS 섹터 / 세부산업: {sector} / {sub_industry}\n"
                f"최근 종가: {latest_close:,.2f} (최근 5거래일 등락률: {chg_5d:+.2f}%)\n"
                f"위 기업에 대해 업종 맞춤형 밸류에이션 멀티플을 적용하고, 점수(score)와 구체적인 점수 산출 근거(reasoning)를 JSON으로 작성하라."
            )
            res = None
            for model_name in candidate_models:
                try:
                    res = client.models.generate_content(
                        model=model_name,
                        contents=user_prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=SYSTEM_ANALYST_PROMPT,
                            response_mime_type="application/json",
                            temperature=0.2
                        )
                    )
                    break
                except Exception:
                    continue

            if res is not None:
                try:
                    result = json.loads(res.text)
                except Exception:
                    result = {
                        "score": 0,
                        "opinion": "관망",
                        "valuation_metric": "PER / EV/EBITDA",
                        "summary": f"{company} 응답 파싱 보류",
                        "reasoning": "모델 응답 JSON 파싱 실패로 기본값 기록"
                    }
            else:
                result = {
                    "score": 0,
                    "opinion": "관망",
                    "valuation_metric": "N/A",
                    "summary": f"{company} API 호출 한도 초과 또는 일시 오류",
                    "reasoning": "Gemini API 호출 실패(할당량 제한 등)로 다음 배치 재시도 예정"
                }

            if API_SLEEP_SEC > 0 and idx < total:
                time.sleep(API_SLEEP_SEC)
        else:
            result = {
                "score": 65,
                "opinion": "매수우세",
                "valuation_metric": "EV/EBITDA & FCF Yield",
                "summary": f"{company} ({sector}) 최근 종가 {latest_close:,.2f} 기준 견조한 흐름",
                "reasoning": "API 키 미설정 상태의 테스트 기본값입니다. GEMINI_API_KEY 설정 시 실제 AI 분석 근거가 기록됩니다."
            }

        save_analysis(
            company=company,
            ticker=ticker,
            sector=sector,
            close_price=latest_close,
            score=result.get("score", 0),
            opinion=result.get("opinion", "관망"),
            summary=result.get("summary", ""),
            reasoning=result.get("reasoning", ""),
            valuation_metric=result.get("valuation_metric", "")
        )
        processed += 1
        print(f"[{processed}/{total}] {company}({ticker}) | 종가:{latest_close:.2f} | 점수:{result.get('score')} | {result.get('opinion')}")

    return processed

if __name__ == "__main__":
    # 야간/장마감 후 전 종목(500+) 무인 학습 실행
    run_ai_pipeline_once(limit=550)
    scheduler = BlockingScheduler(timezone="Asia/Seoul")
    scheduler.add_job(lambda: run_ai_pipeline_once(limit=550), "cron", hour=16, minute=30)
    print("⏰ 500+ 종목 AI 무인 수집·분석 스케줄러 가동 중 (매일 16:30 자동 실행)...")
    scheduler.start()