import os
import re
import sqlite3
import pandas as pd
from openai import OpenAI
from app.modules.data_processing import get_sample_for_db

DB_PATH = "./data/credit_data.db"
TABLE_NAME = "credit_applications"

SCHEMA = """
Table: credit_applications
Columns:
  SK_ID_CURR INTEGER          -- Applicant ID
  TARGET INTEGER              -- 1=Default, 0=No Default
  AMT_INCOME_TOTAL REAL       -- Annual income
  AMT_CREDIT REAL             -- Loan amount
  AMT_ANNUITY REAL            -- Monthly annuity
  AMT_GOODS_PRICE REAL        -- Goods price
  DAYS_BIRTH INTEGER          -- Days since birth (negative)
  DAYS_EMPLOYED INTEGER       -- Days employed (negative)
  CODE_GENDER TEXT            -- M/F
  FLAG_OWN_CAR TEXT           -- Y/N
  FLAG_OWN_REALTY TEXT        -- Y/N
  NAME_CONTRACT_TYPE TEXT     -- Cash loans / Revolving loans
  NAME_INCOME_TYPE TEXT       -- Income source
  NAME_EDUCATION_TYPE TEXT    -- Education level
  NAME_FAMILY_STATUS TEXT     -- Marital status
  NAME_HOUSING_TYPE TEXT      -- Housing type
  EXT_SOURCE_1 REAL           -- External credit score 1
  EXT_SOURCE_2 REAL           -- External credit score 2
  EXT_SOURCE_3 REAL           -- External credit score 3
  REGION_RATING_CLIENT INTEGER -- Region risk rating (1-3)
  CNT_CHILDREN INTEGER        -- Number of children
  CNT_FAM_MEMBERS REAL        -- Family members
  AGE_YEARS REAL              -- Derived: age in years
  CREDIT_INCOME_RATIO REAL    -- Derived: credit/income
  ANNUITY_INCOME_RATIO REAL   -- Derived: annuity/income
"""

SYSTEM_PROMPT = f"""You are a SQL expert for a credit risk database.
Given a natural language question, generate a single valid SQLite SQL query.

{SCHEMA}

Rules:
- Return ONLY the SQL query, no explanation, no markdown, no backticks.
- Use only columns listed in the schema above.
- Always LIMIT results to 100 rows unless the question asks for aggregates.
- For aggregates (COUNT, AVG, SUM), do NOT add LIMIT.
- Use ROUND(value, 2) for decimal outputs.
- Never use DROP, DELETE, INSERT, UPDATE, or ALTER statements.
"""


def init_db():
    """Load sample data into SQLite for querying."""
    if os.path.exists(DB_PATH):
        return
    df = get_sample_for_db()
    cols = [c for c in df.columns if c in SCHEMA]
    # Keep only schema columns that exist
    keep = [
        "SK_ID_CURR", "TARGET", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
        "AMT_GOODS_PRICE", "DAYS_BIRTH", "DAYS_EMPLOYED", "CODE_GENDER",
        "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "NAME_CONTRACT_TYPE", "NAME_INCOME_TYPE",
        "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE",
        "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "REGION_RATING_CLIENT",
        "CNT_CHILDREN", "CNT_FAM_MEMBERS", "AGE_YEARS", "CREDIT_INCOME_RATIO",
        "ANNUITY_INCOME_RATIO",
    ]
    available = [c for c in keep if c in df.columns]
    conn = sqlite3.connect(DB_PATH)
    df[available].to_sql(TABLE_NAME, conn, if_exists="replace", index=False)
    conn.close()


def nl_to_sql(question: str) -> str:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        temperature=0,
        max_tokens=300,
    )
    sql = response.choices[0].message.content.strip()
    # Strip markdown if model wraps in backticks
    sql = re.sub(r"```sql|```", "", sql).strip()
    return sql


def validate_sql(sql: str) -> bool:
    forbidden = ["drop", "delete", "insert", "update", "alter", "create"]
    lower = sql.lower()
    return not any(kw in lower for kw in forbidden)


def run_query(sql: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(sql, conn)
    finally:
        conn.close()
    return df


def answer_question(question: str) -> dict:
    """Full pipeline: NL → SQL → execute → return result."""
    try:
        sql = nl_to_sql(question)
        if not validate_sql(sql):
            return {"error": "Query blocked for safety reasons.", "sql": sql}
        df = run_query(sql)
        return {"sql": sql, "data": df, "error": None}
    except Exception as e:
        return {"error": str(e), "sql": "", "data": None}
