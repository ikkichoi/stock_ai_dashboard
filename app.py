# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from database import init_db, load_history
from collector import TARGET_STOCKS, run_ai_pipeline_once

init_db()
st.set_page_config(page_title="AI 주식 통합 파이프라인 대시보드", page_icon="🤖", layout="wide")

st.title("🤖 AI 자동 수집 & 실시간 분석 대시보드")
st.caption("백엔드 수집기(collector.py)가 SQLite DB에 적재한 분석 리포트를 실시간 시각화합니다.")

with st.sidebar:
    st.header("⚙️ 제어 패널")
    selected_company = st.selectbox("종목 선택", list(TARGET_STOCKS.keys()))
    if st.button("🔄 지금 즉시 전 종목 AI 분석 실행", type="primary", width="stretch"):
        with st.spinner("실시간 시세 조회 및 Gemini AI 분석을 DB에 적재하는 중..."):
            run_ai_pipeline_once()
        st.success("✅ 최신 분석 결과가 DB에 적재되었습니다!")
        st.rerun()

# DB에서 해당 종목 누적 이력 조회
history_df = load_history(selected_company)

if history_df.empty:
    st.warning("아직 DB에 수집된 데이터가 없습니다. 왼쪽 사이드바의 [지금 즉시 전 종목 AI 분석 실행] 버튼을 눌러보세요!")
else:
    # database.py의 컬럼명이 영문/한글 어떤 형태든 통일되도록 자동 매핑
    col_map = {
        "id": "번호",
        "company": "종목명",
        "ticker": "티커",
        "close_price": "당시종가",
        "price": "당시종가",
        "score": "감성점수",
        "sentiment_score": "감성점수",
        "opinion": "투자의견",
        "summary": "요약",
        "created_at": "분석일시",
        "timestamp": "분석일시",
    }
    history_df = history_df.rename(columns={k: v for k, v in col_map.items() if k in history_df.columns})

    latest = history_df.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("최근 분석 종가", f"{float(latest['당시종가']):,.0f}" if "당시종가" in latest else "-")
    c2.metric("최신 AI 감성점수", f"{latest['감성점수']} 점" if "감성점수" in latest else "-")
    c3.metric("투자의견", str(latest["투자의견"]) if "투자의견" in latest else "-")
    c4.metric("마지막 업데이트", str(latest["분석일시"])[:16] if "분석일시" in latest else "-")

    st.divider()
    # 시계열 감성점수 추이 차트 (과거 -> 최신순 정렬)
    sort_col = "번호" if "번호" in history_df.columns else history_df.columns[0]
    chart_df = history_df.sort_values(sort_col)
    if "분석일시" in chart_df.columns and "감성점수" in chart_df.columns:
        fig = go.Figure(go.Scatter(
            x=chart_df["분석일시"],
            y=chart_df["감성점수"],
            mode="lines+markers",
            name="AI 감성점수",
            line=dict(color="#2ca02c", width=3)
        ))
        fig.update_layout(title=f"{selected_company} 날짜별 AI 감성점수 변화 추이", yaxis_range=[-100, 100], height=350)
        st.plotly_chart(fig, width="stretch")

    st.subheader("📜 DB 누적 분석 히스토리")
    st.dataframe(history_df, width="stretch", hide_index=True)