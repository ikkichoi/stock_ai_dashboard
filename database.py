# -*- coding: utf-8 -*-
import sqlite3
from datetime import datetime
import pandas as pd

DB_PATH = "stock_ai.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company TEXT NOT NULL,
            ticker TEXT NOT NULL,
            close_price REAL NOT NULL,
            score INTEGER NOT NULL,
            opinion TEXT NOT NULL,
            summary TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def save_analysis(company: str, ticker: str, close_price: float, score: int, opinion: str, summary: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT INTO analysis_history (company, ticker, close_price, score, opinion, summary, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (company, ticker, close_price, score, opinion, summary, now_str))
    conn.commit()
    conn.close()

def load_history(company: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT id AS 번호, company AS 종목명, close_price AS 당시종가,
               score AS 감성점수, opinion AS 투자의견, summary AS 핵심요약, created_at AS 분석일시
        FROM analysis_history
        WHERE company = ?
        ORDER BY id DESC
    """
    df = pd.read_sql_query(query, conn, params=(company,))
    conn.close()
    return df