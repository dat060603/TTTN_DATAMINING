import pandas as pd
import numpy as np
import pulp
import streamlit as st
import matplotlib.pyplot as plt
from components.data_loader import load_data

def optimize_shipping(df, warehouses, capacity):
    # Tạo bảng chi phí
    options = []
    for _, row in df.iterrows():
        for wh, wh_loc in warehouses.items():
            distance = np.sqrt((row['cust_x'] - wh_loc[0])**2 + (row['cust_y'] - wh_loc[1])**2)
            for day_offset in range(3):  # giao trong vòng 3 ngày
                delivery_date = row['ORDERDATE'] + pd.Timedelta(days=day_offset)

                base_cost = distance * 0.4
                if delivery_date.weekday() >= 5:
                    base_cost += 5
                if distance > 50:
                    base_cost += 8

                options.append({
                    'order_id': row['ORDERNUMBER'],
                    'order_line': row['ORDERLINENUMBER'],
                    'warehouse': wh,
                    'delivery_date': delivery_date.strftime('%Y-%m-%d'),
                    'cost': base_cost,
                    'distance': distance
                })

    cost_df = pd.DataFrame(options)

    cost_dict = {
        (r['order_id'], r['order_line'], r['warehouse'], r['delivery_date']): r['cost']
        for _, r in cost_df.iterrows()
    }

    # Biến quyết định
    orders = cost_df[['order_id', 'order_line']].drop_duplicates().to_records(index=False)
    x = pulp.LpVariable.dicts("x", cost_dict.keys(), cat="Binary")

    # Mô hình
    model = pulp.LpProblem("Minimize_Shipping_Cost", pulp.LpMinimize)
    model += pulp.lpSum(cost_dict[k] * x[k] for k in cost_dict.keys())

    # Ràng buộc: mỗi order line chỉ được chọn 1 phương án
    for oid, line in orders:
        model += pulp.lpSum(
            x[(oid, line, wh, date)]
            for (oi, li, wh, date) in cost_dict.keys()
            if oi == oid and li == line
        ) == 1

    # Ràng buộc: capacity theo kho và ngày
    for wh in warehouses.keys():
        for d in cost_df['delivery_date'].unique():
            model += pulp.lpSum(
                x[(oid, line, w, dd)]
                for (oid, line, w, dd) in cost_dict.keys()
                if w == wh and dd == d
            ) <= capacity

    # Giải
    model.solve(pulp.PULP_CBC_CMD(msg=False))

    # Lấy kết quả
    assignments = [
        (oid, line, wh, date, cost_dict[(oid, line, wh, date)])
        for (oid, line, wh, date) in cost_dict.keys()
        if pulp.value(x[(oid, line, wh, date)]) > 0.5
    ]

    result_df = pd.DataFrame(assignments, columns=['order_id', 'order_line', 'warehouse', 'delivery_date', 'shipping_cost'])
    result_df = result_df.merge(
        df[['ORDERNUMBER', 'ORDERLINENUMBER', 'ORDERDATE', 'cust_x', 'cust_y']],
        left_on=['order_id', 'order_line'],
        right_on=['ORDERNUMBER', 'ORDERLINENUMBER'],
        how='left'
    ).drop(columns=['ORDERNUMBER', 'ORDERLINENUMBER'])
    result_df['ORDERDATE'] = result_df['ORDERDATE'].dt.strftime('%Y-%m-%d')

    total_cost = result_df['shipping_cost'].sum()
    return result_df, total_cost, cost_df

def app():
    st.title("🚚 So sánh tối ưu giao hàng: 1 kho vs 3 kho")

    # Load data
    df = load_data()
    df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'])

    # --- Scenario 1: 1 kho ---
    warehouses_1 = {"WH_ONLY": (0, 0)}
    res1, cost1, cost_df1 = optimize_shipping(df, warehouses_1, capacity=18)

    # --- Scenario 2: 3 kho ---
    warehouses_3 = {
        "WH_A": (0, 0),
        "WH_B": (50, 0),
        "WH_C": (25, 43)
    }
    res3, cost3, cost_df3 = optimize_shipping(df, warehouses_3, capacity=13)

    # Hiển thị kết quả
    st.subheader("📦 Kết quả kịch bản 1 kho")
    st.dataframe(res1.head(20))
    st.success(f"Tổng chi phí: {cost1:,.2f}")

    st.subheader("📦 Kết quả kịch bản 3 kho")
    st.dataframe(res3.head(20))
    st.success(f"Tổng chi phí: {cost3:,.2f}")

    # --- Biểu đồ so sánh ---
    st.subheader("📊 So sánh tổng chi phí giao hàng")
    fig, ax = plt.subplots()
    ax.bar(["1 kho"], [cost1], alpha=0.7, label="1 kho")
    ax.bar(["3 kho"], [cost3], alpha=0.7, label="3 kho")
    ax.set_ylabel("Chi phí")
    st.pyplot(fig)

    st.subheader("📊 Histogram chi phí từng đơn")
    fig, ax = plt.subplots()
    ax.hist(res1['shipping_cost'], bins=20, alpha=0.5, label="1 kho")
    ax.hist(res3['shipping_cost'], bins=20, alpha=0.5, label="3 kho")
    ax.set_xlabel("Chi phí đơn hàng")
    ax.set_ylabel("Số lượng")
    ax.legend()
    st.pyplot(fig)
