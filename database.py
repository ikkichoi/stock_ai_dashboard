# -*- coding: utf-8 -*-
import os
import sqlite3
from datetime import datetime
import pandas as pd

DB_PATH = os.getenv("DB_PATH", "stock_analysis.db")

def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    conn = get_conn()
    cur = conn.cursor()
    # 1. 분석 히스토리 테이블
    cur.execute("""
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company TEXT NOT NULL,
            ticker TEXT NOT NULL,
            sector TEXT DEFAULT 'Unknown',
            close_price REAL,
            score INTEGER,
            opinion TEXT,
            summary TEXT,
            reasoning TEXT,
            valuation_metric TEXT,
            created_at TEXT
        )
    """)
    # 기존 DB 테이블 마이그레이션 (컬럼 누락 시 자동 추가)
    cur.execute("PRAGMA table_info(analysis_history)")
    existing_cols = {row[1] for row in cur.fetchall()}
    for col, col_type in [
        ("sector", "TEXT DEFAULT 'Unknown'"),
        ("reasoning", "TEXT DEFAULT ''"),
        ("valuation_metric", "TEXT DEFAULT ''")
    ]:
        if col not in existing_cols:
            cur.execute(f"ALTER TABLE analysis_history ADD COLUMN {col} {col_type}")

    # 2. 500+ 종목 유니버스 마스터 테이블
    cur.execute("""
        CREATE TABLE IF NOT EXISTS stock_universe (
            ticker TEXT PRIMARY KEY,
            company TEXT NOT NULL,
            sector TEXT,
            sub_industry TEXT,
            updated_at TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_universe(df: pd.DataFrame):
    conn = get_conn()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = [
        (row["ticker"], row["company"], row.get("sector", "Unknown"), row.get("sub_industry", ""), now)
        for _, row in df.iterrows()
    ]
    conn.executemany("""
        INSERT OR REPLACE INTO stock_universe (ticker, company, sector, sub_industry, updated_at)
        VALUES (?, ?, ?, ?, ?)
    """, rows)
    conn.commit()
    conn.close()

def load_universe() -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql_query("SELECT ticker, company, sector, sub_industry FROM stock_universe ORDER BY sector, company", conn)
    conn.close()
    return df

def save_analysis(company, ticker, sector, close_price, score, opinion, summary, reasoning="", valuation_metric=""):
    conn = get_conn()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute("""
        INSERT INTO analysis_history
        (company, ticker, sector, close_price, score, opinion, summary, reasoning, valuation_metric, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (company, ticker, sector, close_price, int(score), opinion, summary, reasoning, valuation_metric, now))
    conn.commit()
    conn.close()

def load_history(company: str) -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT * FROM analysis_history WHERE company = ? ORDER BY id DESC",
        conn,
        params=(company,)
    )
    conn.close()
    return df

def load_latest_snapshot() -> pd.DataFrame:
    """전 종목 최신 분석 1건씩 요약 조회 (스크리닝용)"""
    conn = get_conn()
    df = pd.read_sql_query("""
        SELECT a.*
        FROM analysis_history a
        INNER JOIN (
            SELECT ticker, MAX(id) AS max_id
            FROM analysis_history
            GROUP BY ticker
        ) b ON a.id = b.max_id
        ORDER BY a.score DESC
    """, conn)
    conn.close()
    return df