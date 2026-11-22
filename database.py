import os
import pandas as pd
import psycopg2
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

def get_db_url():
    """로컬(.env) 또는 Streamlit Cloud(st.secrets)에서 DATABASE_URL을 가져옵니다."""
    db_url = os.getenv("DATABASE_URL", "").strip()
    if not db_url:
        try:
            db_url = st.secrets.get("DATABASE_URL", "").strip()
        except Exception:
            pass
    return db_url

def get_connection():
    """Supabase PostgreSQL DB 연결 객체를 반환합니다."""
    db_url = get_db_url()
    if not db_url:
        raise ValueError("DATABASE_URL이 설정되지 않았습니다. .env 또는 Streamlit Secrets를 확인하세요.")
    return psycopg2.connect(db_url)

def init_db():
    """클라우드 PostgreSQL에 analysis_history 테이블이 없으면 생성합니다."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS analysis_history (
            id SERIAL PRIMARY KEY,
            company TEXT NOT NULL,
            ticker TEXT NOT NULL,
            close_price DOUBLE PRECISION NOT NULL,
            score INTEGER NOT NULL,
            opinion TEXT NOT NULL,
            summary TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    cur.close()
    conn.close()

def save_analysis(company, ticker, close_price, score, opinion, summary):
    """AI 분석 결과 1건을 클라우드 DB에 저장합니다."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO analysis_history (company, ticker, close_price, score, opinion, summary)
        VALUES (%s, %s, %s, %s, %s, %s);
    """, (company, ticker, float(close_price), int(score), opinion, summary))
    conn.commit()
    cur.close()
    conn.close()

def load_all_history():
    """클라우드 DB에 저장된 모든 분석 이력을 최신순으로 불러옵니다."""
    conn = get_connection()
    query = """
        SELECT id, company, ticker, close_price, score, opinion, summary, created_at
        FROM analysis_history
        ORDER BY created_at DESC;
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def load_history(company=None):
    """app.py 호환 함수: 특정 종목(company) 또는 전체 분석 이력을 반환합니다."""
    df = load_all_history()
    if company and not df.empty:
        return df[df["company"] == company]
    return df