# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from database import init_db, load_history, load_latest_snapshot
from collector import get_target_universe, sync_sp500_universe, run_ai_pipeline_once

init_db()
st.set_page_config(page_title="AI 주식 500+ 통합 리서치 대시보드", page_icon="🤖", layout="wide")

st.title("🤖 AI 500+ 종목 통합 리서치 & 자동 분석 대시보드")
st.caption("S&P 500 전 종목 유니버스 기반 업종별 맞춤형 밸류에이션 및 AI 감성·펀더멘털 스코어링")

universe_df = get_target_universe()

with st.sidebar:
    st.header("⚙️ 500+ 파이프라인 제어")
    st.info(f"현재 등록된 분석 유니버스: **{len(universe_df)}개 종목**")

    if st.button("🌐 S&P 500 유니버스 동기화 (500+)", width="stretch"):
        with st.spinner("Wikipedia에서 S&P 500 전 종목 리스트를 가져오는 중..."):
            universe_df = sync_sp500_universe()
        st.success(f"✅ 총 {len(universe_df)}개 종목 동기화 완료!")
        st.rerun()

    st.divider()
    sectors = ["전체 (All Sectors)"] + sorted(universe_df["sector"].dropna().unique().tolist())
    selected_sector = st.selectbox("1️⃣ 분석/필터 섹터 선택", sectors)

    if selected_sector != "전체 (All Sectors)":
        filtered_universe = universe_df[universe_df["sector"] == selected_sector]
    else:
        filtered_universe = universe_df

    batch_limit = st.slider(
        "2️⃣ 1회 일괄 분석 종목 수",
        min_value=1,
        max_value=max(1, len(filtered_universe)),
        value=min(10, len(filtered_universe)),
        help="웹 화면에서는 타임아웃 방지를 위해 5~20개 단위로 실행하고, 500개 전체는 터미널(python collector.py)로 실행하는 것을 권장합니다."
    )

    if st.button("🚀 선택 섹터 일괄 AI 분석 실행", type="primary", width="stretch"):
        prog_bar = st.progress(0)
        status_txt = st.empty()

        def update_prog(curr, tot, msg):
            prog_bar.progress(curr / tot)
            status_txt.text(msg)

        count = run_ai_pipeline_once(
            limit=batch_limit,
            target_sector=selected_sector,
            progress_callback=update_prog
        )
        st.success(f"✅ {count}개 종목 AI 분석 및 근거 적재 완료!")
        st.rerun()

    st.divider()
    company_options = filtered_universe["company"].tolist() if not filtered_universe.empty else universe_df["company"].tolist()
    selected_company = st.selectbox("3️⃣ 상세 리포트 조회 종목", company_options)

    if st.button("🎯 이 종목만 즉시 단건 분석", width="stretch"):
        target_ticker = universe_df.loc[universe_df["company"] == selected_company, "ticker"].iloc[0]
        with st.spinner(f"{selected_company} ({target_ticker}) 심층 분석 중..."):
            run_ai_pipeline_once(limit=1, specific_tickers=[target_ticker])
        st.success("✅ 단건 분석 완료!")
        st.rerun()

tab1, tab2 = st.tabs(["📊 종목별 AI 심층 분석 & 점수 근거", "🏆 500+ 전 종목 AI 스코어 랭킹"])

with tab1:
    history_df = load_history(selected_company)

    if history_df.empty:
        st.warning(f"아직 **{selected_company}**의 DB 분석 데이터가 없습니다. 왼쪽 사이드바에서 **[🎯 이 종목만 즉시 단건 분석]** 또는 **[🚀 선택 섹터 일괄 AI 분석 실행]**을 눌러보세요!")
    else:
        latest = history_df.iloc[0]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("최근 분석 종가", f"{float(latest['close_price']):,.2f}")
        c2.metric("최신 AI 스코어", f"{latest['score']} 점")
        c3.metric("투자의견", str(latest["opinion"]))
        c4.metric("핵심 밸류에이션 지표", str(latest.get("valuation_metric") or "-"))
        c5.metric("마지막 업데이트", str(latest["created_at"])[:16])

        # 핵심 요약 및 점수 산출 근거 상세 박스
        st.subheader("🧠 AI 점수 산출 근거 및 리서치 코멘트")
        st.info(f"**[한줄 핵심 요약]** {latest.get('summary', '')}")
        with st.container(border=True):
            st.markdown(f"**📌 왜 {latest['score']}점({latest['opinion']})이 나왔는가?**")
            reasoning_text = latest.get("reasoning") or "상세 근거가 기록되지 않은 이전 버전 데이터입니다. 재분석을 실행하면 상세 근거가 표시됩니다."
            st.write(reasoning_text)

        st.divider()
        chart_df = history_df.sort_values("id")
        fig = go.Figure(go.Scatter(
            x=chart_df["created_at"],
            y=chart_df["score"],
            mode="lines+markers",
            name="AI 스코어",
            line=dict(color="#2ca02c", width=3)
        ))
        fig.update_layout(title=f"{selected_company} 날짜별 AI 스코어 변화 추이", yaxis_range=[-100, 100], height=320)
        st.plotly_chart(fig, width="stretch")

        st.subheader("📜 DB 누적 분석 히스토리")
        display_cols = [c for c in ["id", "created_at", "company", "ticker", "sector", "close_price", "score", "opinion", "valuation_metric", "summary", "reasoning"] if c in history_df.columns]
        st.dataframe(history_df[display_cols], width="stretch", hide_index=True)

with tab2:
    st.subheader("🏆 DB 적재 전 종목 최신 AI 스코어 순위")
    snap_df = load_latest_snapshot()
    if snap_df.empty:
        st.info("아직 분석된 종목이 없습니다. 사이드바에서 일괄 분석을 실행해 보세요.")
    else:
        st.caption(f"현재 총 **{len(snap_df)}개** 종목의 최신 AI 분석 리포트가 DB에 적재되어 있습니다.")
        show_cols = [c for c in ["score", "opinion", "company", "ticker", "sector", "close_price", "valuation_metric", "summary", "reasoning", "created_at"] if c in snap_df.columns]
        st.dataframe(snap_df[show_cols], width="stretch", hide_index=True)