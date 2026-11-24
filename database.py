import os
import sqlite3
import pandas as pd

DB_PATH = os.getenv("DB_PATH", "stock_analysis.db")


def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company TEXT NOT NULL,
            ticker TEXT NOT NULL,
            sector TEXT DEFAULT 'Unknown',
            close_price REAL,
            score INTEGER,
            opinion TEXT,
            valuation_metric TEXT DEFAULT '',
            summary TEXT,
            reasoning TEXT DEFAULT '',
            market_cap TEXT DEFAULT 'N/A',
            per REAL,
            ev_ebitda REAL,
            fcf_yield REAL,
            bear_target REAL,
            base_target REAL,
            bull_target REAL,
            upside_pct REAL,
            updated_at TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS stock_universe (
            ticker TEXT PRIMARY KEY,
            company TEXT NOT NULL,
            sector TEXT DEFAULT 'Unknown',
            updated_at TEXT
        )
    """)

    # 기존 Day 2 / Day 3 DB 호환 자동 마이그레이션
    cur.execute("PRAGMA table_info(analysis_history)")
    existing_cols = {row[1] for row in cur.fetchall()}
    new_columns = [
        ("sector", "TEXT DEFAULT 'Unknown'"),
        ("reasoning", "TEXT DEFAULT ''"),
        ("valuation_metric", "TEXT DEFAULT ''"),
        ("market_cap", "TEXT DEFAULT 'N/A'"),
        ("per", "REAL"),
        ("ev_ebitda", "REAL"),
        ("fcf_yield", "REAL"),
        ("bear_target", "REAL"),
        ("base_target", "REAL"),
        ("bull_target", "REAL"),
        ("upside_pct", "REAL"),
    ]
    for col, col_type in new_columns:
        if col not in existing_cols:
            cur.execute(f"ALTER TABLE analysis_history ADD COLUMN {col} {col_type}")

    conn.commit()
    conn.close()


def save_universe(df: pd.DataFrame):
    conn = get_conn()
    cur = conn.cursor()
    rows = [
        (r["ticker"], r["company"], r["sector"], r["updated_at"])
        for _, r in df.iterrows()
    ]
    cur.executemany("""
        INSERT INTO stock_universe (ticker, company, sector, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(ticker) DO UPDATE SET
            company=excluded.company,
            sector=excluded.sector,
            updated_at=excluded.updated_at
    """, rows)
    conn.commit()
    conn.close()


def load_universe() -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT ticker, company, sector, updated_at FROM stock_universe ORDER BY company ASC",
        conn,
    )
    conn.close()
    return df


def insert_analysis(row: dict):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO analysis_history
        (company, ticker, sector, close_price, score, opinion, valuation_metric,
         summary, reasoning, market_cap, per, ev_ebitda, fcf_yield,
         bear_target, base_target, bull_target, upside_pct, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        row["company"],
        row["ticker"],
        row.get("sector", "Unknown"),
        row["close_price"],
        row["score"],
        row["opinion"],
        row.get("valuation_metric", ""),
        row["summary"],
        row.get("reasoning", ""),
        row.get("market_cap", "N/A"),
        row.get("per"),
        row.get("ev_ebitda"),
        row.get("fcf_yield"),
        row.get("bear_target"),
        row.get("base_target"),
        row.get("bull_target"),
        row.get("upside_pct"),
        row["updated_at"],
    ))
    conn.commit()
    conn.close()


def load_history(company: str) -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT * FROM analysis_history WHERE company = ? ORDER BY id DESC",
        conn,
        params=(company,),
    )
    conn.close()
    return df


def load_latest_snapshot() -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql_query("""
        SELECT a.*
        FROM analysis_history a
        INNER JOIN (
            SELECT ticker, MAX(id) AS max_id
            FROM analysis_history
            GROUP BY ticker
        ) b ON a.id = b.max_id
        ORDER BY a.score DESC, a.upside_pct DESC
    """, conn)
    conn.close()
    return df