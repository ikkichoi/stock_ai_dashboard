import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from database import init_db, load_history, load_latest_snapshot
from collector import get_target_universe, sync_sp500_universe, run_ai_pipeline_once

init_db()
st.set_page_config(page_title="AI 주식 500+ Owner's Earnings 리서치", page_icon="🤖", layout="wide")

universe_df = get_target_universe()

st.title("🤖 AI 주식 500+ Owner's Earnings 펀더멘털 리서치")
st.caption(
    f"현재 커버리지: 총 **{len(universe_df)}개 종목** (S&P 500 + 주요 플랫폼/반도체) | "
    "SBC 차감 Owner's Earnings DCF · GAAP OPM 분리 · Bear/Base/Bull 시나리오 자동 산출"
)

# ================= 사이드바 파이프라인 제어 =================
with st.sidebar:
    st.header("⚙️ 500+ 리서치 엔진 제어")

    if st.button("🌐 S&P 500 유니버스 동기화 (500+)", width="stretch"):
        with st.spinner("S&P 500 구성종목(500+) 동기화 중..."):
            try:
                universe_df = sync_sp500_universe()
                st.success(f"총 {len(universe_df)}개 종목 동기화 완료!")
                st.rerun()
            except Exception as e:
                st.exception(e)
                
    st.divider()
    st.subheader("🚀 배치 AI 리서치 실행")
    sectors = ["전체 (All Sectors)"] + sorted(universe_df["sector"].dropna().unique().tolist())
    selected_sector = st.selectbox("대상 섹터 필터 (GICS 기준)", sectors)
    batch_limit = st.slider("1회 실행 종목 수 (API 안전 배치)", min_value=1, max_value=100, value=10)

    if st.button("▶️ 선택 섹터 AI 분석 실행", type="primary", width="stretch"):
        prog_bar = st.progress(0)
        status_txt = st.empty()

        def update_prog(curr, tot, msg):
            prog_bar.progress(curr / tot if tot > 0 else 1.0)
            status_txt.text(msg)

        try:
            count = run_ai_pipeline_once(
                limit=batch_limit,
                target_sector=selected_sector,
                progress_callback=update_prog,
            )
            if count == 0:
                st.warning("분석 대상 종목을 찾지 못했습니다. 먼저 'S&P 500 유니버스 동기화'를 실행하세요.")
            else:
                st.success(f"{count}개 종목 리서치 완료!")
                st.rerun()
        except Exception as e:
            st.error(f"실행 오류: {e}")

    st.divider()
    st.subheader("🔍 개별 종목 심층 조회 & 즉시 분석")
    company_options = universe_df["company"].tolist()
    default_idx = company_options.index("Netflix") if "Netflix" in company_options else 0
    selected_company = st.selectbox("종목 선택 (검색 가능)", company_options, index=default_idx)

    target_row = universe_df[universe_df["company"] == selected_company].iloc[0]
    target_ticker = target_row["ticker"]
    target_sector_label = target_row["sector"]

    if st.button(f"⚡ '{selected_company}' 즉시 심층 분석", width="stretch"):
        st.write(f"1️⃣ 버튼 클릭됨: {selected_company} ({target_ticker})")
        try:
            cnt = run_ai_pipeline_once(limit=1, specific_tickers=[target_ticker])
            st.write(f"2️⃣ 실행 완료 반환값: {cnt}")
            if cnt == 0:
                st.error("⚠️ 반환값이 0입니다. 유니버스 매칭 실패 또는 API 호출 중단!")
            else:
                st.success("✅ DB 저장 성공! 새로고침합니다.")
                st.rerun()
        except Exception as e:
            st.exception(e)  # 에러 전체 스택을 화면에 빨간 박스로 강제 출력

# ================= 메인 탭 구성 =================
tab1, tab2 = st.tabs(["📊 종목별 OE DCF · 재무 정상화 리포트", "🏆 500+ 저평가 종목 발굴 & 랭킹"])

with tab1:
    history_df = load_history(selected_company)

    if history_df.empty:
        st.info(
            f"💡 **[{selected_company} ({target_ticker}) | GICS: {target_sector_label}]**의 분석 기록이 아직 없습니다.\n\n"
            f"왼쪽 사이드바에서 **'⚡ {selected_company} 즉시 심층 분석'** 버튼을 누르시면 SBC 차감 Owner's Earnings와 목표주가가 즉시 산출됩니다."
        )
    else:
        latest = history_df.iloc[0]
        st.subheader(f"📌 {selected_company} ({latest['ticker']}) — GICS: {latest['sector']}")

        # 1단: 핵심 투자 지표
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("최근 종가", f"{latest['close_price']:,.2f}")
        c2.metric("AI 투자 스코어", f"{latest['score']} / 100")
        c3.metric("투자의견", latest["opinion"])
        upside = latest.get("upside_pct")
        c4.metric("Base 안전마진(괴리율)", f"{upside:+.1f}%" if pd.notna(upside) else "N/A")
        c5.metric("업데이트 시각", str(latest["updated_at"])[:16])

        # 2단: GAAP · Owner's Earnings · 레버리지 팩트
        st.markdown("#### 📐 재무 팩트 & Owner's Earnings (SBC 차감)")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("시가총액", latest.get("market_cap") or "N/A")
        m2.metric("GAAP OPM", f"{latest['gaap_opm']:.1f}%" if pd.notna(latest.get("gaap_opm")) else "N/A")
        m3.metric("FCF Yield", f"{latest['fcf_yield']:.1f}%" if pd.notna(latest.get("fcf_yield")) else "N/A")
        m4.metric("OE Yield (FCF-SBC)", f"{latest['oe_yield']:.1f}%" if pd.notna(latest.get("oe_yield")) else "N/A")
        m5.metric("EV/EBITDA", f"{latest['ev_ebitda']:.1f}x" if pd.notna(latest.get("ev_ebitda")) else "N/A")
        m6.metric("순부채/EBITDA", f"{latest['net_debt_ebitda']:.2f}x" if pd.notna(latest.get("net_debt_ebitda")) else "N/A")

        # 3단: Bear / Base / Bull 목표주가 시나리오
        st.markdown("#### 🎯 12개월 적정주가 시나리오 (Bear / Base / Bull)")
        t1, t2, t3 = st.columns(3)
        t1.metric("🐻 Bear 시나리오", f"{latest['bear_target']:,.2f}" if pd.notna(latest.get("bear_target")) else "N/A")
        t2.metric("🎯 Base 기본가치", f"{latest['base_target']:,.2f}" if pd.notna(latest.get("base_target")) else "N/A")
        t3.metric("🐂 Bull 시나리오", f"{latest['bull_target']:,.2f}" if pd.notna(latest.get("bull_target")) else "N/A")

        # 4단: AI 애널리스트 DCF 가정 & 회계 정상화 주의사항 & 7단계 근거
        with st.container(border=True):
            st.markdown(f"**⚖️ 주 평가법 & 멀티플:** `{latest.get('valuation_metric', 'N/A')}`")
            if latest.get("dcf_assumptions"):
                st.markdown(f"**📊 시나리오 가정 (성장/할인율/g 또는 배수):** `{latest['dcf_assumptions']}`")
            if latest.get("normalization_note"):
                st.warning(f"**⚠️ 회계·자본구조 정상화 주의:** {latest['normalization_note']}")
            st.markdown(f"**💡 핵심 결론:** {latest['summary']}")
            st.markdown("**🧠 7단계 기관 리서치 산출 근거:**")
            st.write(latest.get("reasoning") or "상세 근거가 기록되지 않았습니다.")

        # 5단: 누적 시계열 차트
        chart_df = history_df.sort_values("id")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=chart_df["updated_at"], y=chart_df["close_price"], name="종가", mode="lines+markers")
        )
        fig.add_trace(
            go.Scatter(
                x=chart_df["updated_at"],
                y=chart_df["score"],
                name="AI 스코어",
                yaxis="y2",
                mode="lines+markers",
                line=dict(dash="dot"),
            )
        )
        fig.update_layout(
            title="누적 주가 vs AI 스코어 추이",
            yaxis=dict(title="주가"),
            yaxis2=dict(title="AI 스코어", overlaying="y", side="right", range=[0, 100]),
            hovermode="x unified",
        )
        st.plotly_chart(fig, width="stretch")

with tab2:
    st.subheader("🏆 500+ 유니버스 최신 AI 애널리스트 스코어 & 안전마진 랭킹")
    snap_df = load_latest_snapshot()

    if snap_df.empty:
        st.warning("아직 학습·분석된 종목 데이터가 없습니다. 왼쪽 사이드바에서 AI 분석을 실행해 주세요.")
    else:
        fc1, fc2, fc3 = st.columns([2, 2, 1])
        min_score = fc1.slider("최소 AI 스코어 필터", 0, 100, 0)
        min_upside = fc2.slider("최소 Base 안전마진(%) 필터", -50, 100, -50)

        filtered_df = snap_df[
            (snap_df["score"] >= min_score)
            & (snap_df["upside_pct"].fillna(-999) >= min_upside)
        ].copy()

        csv_bytes = filtered_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        fc3.markdown("<br>", unsafe_allow_html=True)
        fc3.download_button(
            label="📥 CSV 리포트 다운로드",
            data=csv_bytes,
            file_name="ai_sp500_oe_research_report.csv",
            mime="text/csv",
            width="stretch",
        )

        st.write(f"필터링된 종목 수: **{len(filtered_df)}개** / 전체 학습 완료: **{len(snap_df)}개**")
        show_cols = [
            "company", "ticker", "sector", "market_cap", "close_price",
            "base_target", "upside_pct", "score", "opinion",
            "gaap_opm", "fcf_yield", "oe_yield", "ev_ebitda", "net_debt_ebitda",
            "valuation_metric", "normalization_note",
        ]
        existing_show_cols = [c for c in show_cols if c in filtered_df.columns]
        st.dataframe(filtered_df[existing_show_cols], width="stretch", hide_index=True)