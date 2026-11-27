import os
import requests
from datetime import datetime, timezone
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


def send_telegram_message(message: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ TELEGRAM_BOT_TOKEN 또는 TELEGRAM_CHAT_ID가 설정되지 않았습니다.")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    try:
        resp = requests.post(url, json=payload, timeout=15)
        resp.raise_for_status()
        print("✅ 텔레그램 리포트 전송 완료!")
        return True
    except Exception as e:
        print(f"❌ 텔레그램 전송 실패: {e}")
        return False


def notify_top_undervalued_stocks(min_score: int = 80, min_upside: float = 15.0):
    if not DATABASE_URL:
        print("❌ DATABASE_URL이 없습니다.")
        return

    engine = create_engine(DATABASE_URL)
    query = text("""
        SELECT ticker, company, sector, close_price, base_target, upside_pct, score, opinion, gaap_opm, fcf_yield
        FROM stock_research
        WHERE score >= :min_score
          AND upside_pct >= :min_upside
        ORDER BY score DESC, upside_pct DESC
        LIMIT 5
    """)

    with engine.connect() as conn:
        rows = conn.execute(query, {"min_score": min_score, "min_upside": min_upside}).fetchall()
        total_count = conn.execute(text("SELECT COUNT(*) FROM stock_research")).scalar()

    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    if not rows:
        msg = (
            f"📊 <b>[S&P 500 OE 리서치 일일 브리핑 - {today_str}]</b>\n\n"
            f"• 누적 분석 완료: <b>{total_count}개</b> 종목\n"
            f"• 기준(스코어 {min_score}점↑ & 안전마진 {min_upside}%↑) 충족 종목이 아직 없습니다."
        )
        send_telegram_message(msg)
        return

    lines = [
        f"🚀 <b>[S&P 500 저평가 1티어 리포트 - {today_str}]</b>",
        f"누적 DB: <b>{total_count}개</b> | 필터: 스코어 {min_score}↑ · 안전마진 {min_upside}%↑\n",
    ]

    for idx, r in enumerate(rows, 1):
        lines.append(
            f"<b>{idx}. {r.company} ({r.ticker})</b> [{r.opinion}]\n"
            f"   • 현재가: <code>${r.close_price:.2f}</code> → 목표가: <code>${r.base_target:.2f}</code> (<b>+{r.upside_pct:.1f}%</b>)\n"
            f"   • AI 스코어: <b>{r.score}점</b> | OPM: {r.gaap_opm:.1f}% | FCF수익률: {r.fcf_yield:.1f}%\n"
        )

    send_telegram_message("\n".join(lines))


if __name__ == "__main__":
    notify_top_undervalued_stocks()