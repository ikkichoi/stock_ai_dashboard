# -*- coding: utf-8 -*-
import os
import json
import yfinance as yf
from google import genai
from google.genai import types
from apscheduler.schedulers.blocking import BlockingScheduler
from dotenv import load_dotenv
from database import init_db, save_analysis

load_dotenv()
init_db()

TARGET_STOCKS = {
    "SK하이닉스": "000660.KS",
    "삼성전자": "005930.KS",
    "엔비디아": "NVDA"
}

def run_ai_pipeline_once():
    """전 종목 최신 주가 조회 -> Gemini AI 분석 -> SQLite DB 영구 저장"""
    ai_key = os.getenv("GEMINI_API_KEY", "").strip()
    client = genai.Client(api_key=ai_key) if ai_key else None

    for company, ticker in TARGET_STOCKS.items():
        # 1. yfinance 최근 5일 시세 조회
        df = yf.download(ticker, period="5d", progress=False)
        if df.empty:
            continue
        if hasattr(df.columns, "levels"):
            df.columns = df.columns.get_level_values(0)
        latest_close = float(df["Close"].iloc[-1])

        # 2. Gemini AI 분석 (API 키 미설정 시 테스트용 기본값 방어)
        if client:
            prompt = f"종목: {company} ({ticker}), 최근 종가: {latest_close}. 투자 심리와 핵심 요약을 JSON({{'score': -100~100 정수, 'opinion': '매수우세/관망/매도우세', 'summary': '한줄요약'}})으로만 답하세요."
            res = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            result = json.loads(res.text)
        else:
            result = {"score": 72, "opinion": "매수우세", "summary": f"{company} 최근 종가 {latest_close:,.0f} 기준 견조한 흐름 유지"}

        # 3. SQLite DB에 자동 적재
        save_analysis(company, ticker, latest_close, result["score"], result["opinion"], result["summary"])
        print(f"[자동수집 완료] {company} | 종가: {latest_close} | 점수: {result['score']}")

if __name__ == "__main__":
    # 프로그램 시작 즉시 1회 실행 후, 매일 장 마감 시간(16:00) 자동 반복
    run_ai_pipeline_once()
    scheduler = BlockingScheduler(timezone="Asia/Seoul")
    scheduler.add_job(run_ai_pipeline_once, "cron", hour=16, minute=0)
    print("⏰ 백엔드 무인 수집 스케줄러 가동 중 (매일 16:00 자동 실행)...")
    scheduler.start()