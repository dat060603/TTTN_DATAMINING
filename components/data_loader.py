
# cleaned_sales_data_final
# components/data_loader.py
import numpy as np
import pandas as pd
import streamlit as st
from pulp import lpSum
from pulp import LpProblem, LpVariable, lpSum, LpMinimize, LpBinary, PULP_CBC_CMD

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
    # hoanganh
    # capacity = 13
    # min_delay_days = 3
    # max_extend_days = 7
    # df = df.sort_values('ORDERDATE').reset_index(drop=True)
    # # df = df.head(300)  # Thử chạy mô phỏng với 300 dòng đầu tiên
    # model = LpProblem("Simulate_Shipping", LpMinimize)
    # assign = {}
    # for i in df.index:
    #     start_date = df.loc[i, 'ORDERDATE'] + pd.Timedelta(days=min_delay_days)
    #     end_date = df.loc[i, 'ORDERDATE'] + pd.Timedelta(days=max_extend_days)
    #     for d in pd.date_range(start_date, end_date):
    #         assign[(i, d)] = LpVariable(f"assign_{i}_{d.date()}", cat=LpBinary)
    #
    # # Hàm mục tiêu: ship càng sớm càng tốt
    # model += lpSum((d - df.loc[i, 'ORDERDATE']).days * assign[(i, d)] for (i, d) in assign)
    #
    # # Ràng buộc: mỗi đơn đúng 1 ngày
    # for i in df.index:
    #     model += lpSum(assign[(ii, d)] for (ii, d) in assign if ii == i) == 1
    #
    # # Ràng buộc capacity mỗi ngày
    # all_days = sorted(set(d for (_, d) in assign))
    # for d in all_days:
    #     model += lpSum(assign[(i, dd)] for (i, dd) in assign if dd == d) <= capacity
    #
    # # Solve
    # model.solve(PULP_CBC_CMD(msg=False, timeLimit=60))
    #
    # sim_ship_dates = []
    # for i in df.index:
    #     ship_date = None
    #     start_date = df.loc[i, 'ORDERDATE'] + pd.Timedelta(days=min_delay_days)
    #     end_date = df.loc[i, 'ORDERDATE'] + pd.Timedelta(days=max_extend_days)
    #     for d in pd.date_range(start_date, end_date):
    #         if assign[(i, d)].value() == 1:
    #             ship_date = d
    #             break
    #     sim_ship_dates.append(ship_date)
    #
    # df['SIM_SHIP_DATE'] = sim_ship_dates
    #
    # # -------------------------------------------------
    # # 7. Thêm tọa độ giả định (random quanh VN)
    # # -------------------------------------------------
    df['SIM_SHIP_DATE'] = df['ORDERDATE'] + pd.to_timedelta(
        np.random.randint(1, 10, size=len(df)), unit='d'
    )
    df['SHIP_DAYS_CAP_OLD'] = (df['SIM_SHIP_DATE'] - df['ORDERDATE']).dt.days

    np.random.seed(42)
    df['cust_x'] = np.random.uniform(8.5, 23.5, len(df))  # vĩ độ VN
    df['cust_y'] = np.random.uniform(102, 109.5, len(df))  # kinh độ VN
    # Mới
    return df

