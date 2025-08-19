
# cleaned_sales_data_final
# components/data_loader.py
import numpy as np
import pandas as pd
import streamlit as st

@st.cache_data
def load_data(path: str = "cleaned_sales_data_final.csv"):
    """
    Load và tiền xử lý cơ bản cho cleaned_sales_data.csv
    Trả về DataFrame đã được chuẩn hoá:
      - ORDERDATE: datetime
      - numeric coercion cho PRICEEACH, QUANTITYORDERED, SALES, MSRP
      - tạo YEAR_ID, MONTH_ID
      - tạo TOTAL_ORDER_VALUE, SALES_DIFF
      - giả lập COST (AVG_PRICE * 0.5 ± random noise)
    """
    # 1. Load (hỗ trợ dayfirst nếu file dùng dd/mm/YYYY)
    try:
        df = pd.read_csv(path, encoding='ISO-8859-1', parse_dates=['ORDERDATE'], dayfirst=True)
    except Exception:
        df = pd.read_csv(path, encoding='ISO-8859-1')
        df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'], dayfirst=True, errors='coerce')

    # 2. Chuẩn hoá cột
    df.columns = df.columns.str.upper().str.strip()

    # 3. Numeric coercion
    for c in ['QUANTITYORDERED', 'PRICEEACH', 'SALES', 'MSRP']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # 4. ORDERDATE -> datetime và tạo YEAR_ID / MONTH_ID
    if 'ORDERDATE' in df.columns:
        df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'], errors='coerce', dayfirst=True)
        df['YEAR_ID'] = df['ORDERDATE'].dt.year
        df['MONTH_ID'] = df['ORDERDATE'].dt.month
    # 5. Feature engineering
    if ('QUANTITYORDERED' in df.columns) and ('PRICEEACH' in df.columns):
        df['TOTAL_ORDER_VALUE'] = df['QUANTITYORDERED'] * df['PRICEEACH']
    if ('SALES' in df.columns) and ('TOTAL_ORDER_VALUE' in df.columns):
        df['SALES_DIFF'] = df['SALES'] - df['TOTAL_ORDER_VALUE']
    # 6. Giả lập cột COST
    if 'PRICEEACH' in df.columns:
        # Ví dụ: COST = 50% của PRICEEACH ± 10% random
        np.random.seed(42)
        df['COST'] = df['PRICEEACH'] * (0.5 + 0.1 * np.random.rand(len(df)))
    # 7. Tùy chọn: drop rows invalid ORDERDATE / SALES nếu muốn (comment nếu không muốn)
    # df = df.dropna(subset=['ORDERDATE', 'SALES'])
    return df

