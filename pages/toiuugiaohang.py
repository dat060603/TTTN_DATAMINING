import streamlit as st
import pandas as pd
import numpy as np
from pulp import LpProblem, LpVariable, lpSum, LpMinimize, LpBinary, PULP_CBC_CMD
from components.data_loader import load_data


def app():
    st.title("🚚 Mô phỏng lịch giao hàng tối ưu (3–7 ngày)")

    # Đọc file
    df = load_data()
    df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'])

    # Sắp xếp theo ORDERDATE
    df = df.sort_values('ORDERDATE').reset_index(drop=True)

    # Capacity mỗi ngày (số đơn tối đa/ngày) = percentile 70 số đơn/ngày
    capacity = int(np.percentile(df.groupby('ORDERDATE').size(), 70))
    st.write(f"📦 Công suất tối đa mỗi ngày: **{capacity} đơn**")

    # Khởi tạo mô hình LP
    model = LpProblem("Simulate_Shipping", LpMinimize)

    # Giới hạn số ngày tối đa sau ORDERDATE
    min_delay_days = 3
    max_delay_days = 7

    # Biến nhị phân assign[i, d]
    assign = {}
    for i in df.index:
        start_date = df.loc[i, 'ORDERDATE'] + pd.Timedelta(days=min_delay_days)
        for d in pd.date_range(start_date, df.loc[i, 'ORDERDATE'] + pd.Timedelta(days=max_delay_days)):
            assign[(i, d)] = LpVariable(f"assign_{i}_{d}", cat=LpBinary)

    # Hàm mục tiêu: giao càng sớm càng tốt trong khoảng 3–7 ngày
    model += lpSum((d - df.loc[i, 'ORDERDATE']).days * assign[(i, d)] for (i, d) in assign)

    # Ràng buộc: mỗi đơn giao đúng 1 ngày
    for i in df.index:
        model += lpSum(assign[(ii, d)] for (ii, d) in assign if ii == i) == 1

    # Ràng buộc: công suất mỗi ngày
    all_days = sorted(set(d for (_, d) in assign))
    for d in all_days:
        model += lpSum(assign[(i, dd)] for (i, dd) in assign if dd == d) <= capacity

    # Giải bài toán với giới hạn thời gian 60 giây
    model.solve(PULP_CBC_CMD(msg=False, timeLimit=60))

    # Lấy kết quả SIM_SHIP_DATE
    sim_ship_dates = []
    for i in df.index:
        for d in pd.date_range(df.loc[i, 'ORDERDATE'] + pd.Timedelta(days=min_delay_days),
                               df.loc[i, 'ORDERDATE'] + pd.Timedelta(days=max_delay_days)):
            if assign[(i, d)].value() == 1:
                sim_ship_dates.append(d)
                break

    df['SIM_SHIP_DATE'] = sim_ship_dates

    df['ORDERDATE'] = df['ORDERDATE'].dt.strftime('%Y-%m-%d')
    df['SIM_SHIP_DATE'] = df['SIM_SHIP_DATE'].dt.strftime('%Y-%m-%d')

    # Các đơn không giao đúng 3 ngày
    df_not_3days = df[(pd.to_datetime(df['SIM_SHIP_DATE']) - pd.to_datetime(df['ORDERDATE'])).dt.days != 3]

    st.write("### 📋 Một số đơn không giao đúng sau 3 ngày")
    st.dataframe(df_not_3days[['ORDERNUMBER','ORDERLINENUMBER','ORDERDATE','SIM_SHIP_DATE']].head(20))

    st.write(f"👉 Tổng số đơn **không giao đúng 3 ngày**: {len(df_not_3days)}")


# Dùng trong Streamlit

