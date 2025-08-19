import pandas as pd
import numpy as np
import pulp
import streamlit as st
from components.data_loader import load_data  # nếu bạn đã có sẵn

def app():
    st.title("🚚 Tối ưu lịch giao hàng từ 3 kho (Minimize Shipping Cost)")

    # ===== 1. Đọc dữ liệu gốc =====
    df = load_data()
    df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'])

    # ===== 2. Giả lập vị trí 3 kho =====
    np.random.seed(42)
    warehouses = {
        "WH_A": (0, 0),     # Tọa độ kho 1
        "WH_B": (50, 0),    # Tọa độ kho 2
        "WH_C": (25, 43)    # Tọa độ kho 3
    }

    # Giả lập tọa độ khách hàng
    df['cust_x'] = np.random.randint(0, 60, size=len(df))
    df['cust_y'] = np.random.randint(0, 60, size=len(df))

    # ===== 3. Tạo bảng chi phí cho từng kho và ngày =====
    options = []
    for _, row in df.iterrows():
        for wh, wh_loc in warehouses.items():
            # Khoảng cách Euclidean
            distance = np.sqrt((row['cust_x'] - wh_loc[0])**2 + (row['cust_y'] - wh_loc[1])**2)

            for day_offset in range(3):  # giao trong vòng 3 ngày
                delivery_date = row['ORDERDATE'] + pd.Timedelta(days=day_offset)

                # Tính chi phí
                base_cost = distance * 0.4  # giá 0.4 USD/km
                if delivery_date.weekday() >= 5:  # cuối tuần
                    base_cost += 5
                if distance > 50:
                    base_cost += 8

                options.append({
                    'order_id': row['ORDERNUMBER'],
                    'order_line': row['ORDERLINENUMBER'],
                    'warehouse': wh,
                    'delivery_date': delivery_date.strftime('%Y-%m-%d'),
                    'cost': base_cost
                })

    cost_df = pd.DataFrame(options)

    # ===== 4. Dictionary chi phí =====
    cost_dict = {
        (r['order_id'], r['order_line'], r['warehouse'], r['delivery_date']): r['cost']
        for _, r in cost_df.iterrows()
    }

    # ===== 5. Tạo biến quyết định =====
    orders = cost_df[['order_id', 'order_line']].drop_duplicates().to_records(index=False)
    x = pulp.LpVariable.dicts("x", cost_dict.keys(), cat="Binary")

    # ===== 6. Khởi tạo mô hình =====
    model = pulp.LpProblem("Minimize_Shipping_Cost_3WH", pulp.LpMinimize)

    # Mục tiêu: tối thiểu hóa chi phí
    model += pulp.lpSum(cost_dict[k] * x[k] for k in cost_dict.keys())

    # ===== 7. Ràng buộc =====
    # Mỗi order line giao đúng 1 lần (1 kho + 1 ngày)
    for oid, line in orders:
        model += pulp.lpSum(
            x[(oid, line, wh, date)]
            for (oi, li, wh, date) in cost_dict.keys()
            if oi == oid and li == line
        ) == 1

    # Năng lực tối đa 8 đơn/kho/ngày
    MAX_CAPACITY = 8
    for wh in warehouses.keys():
        for d in cost_df['delivery_date'].unique():
            model += pulp.lpSum(
                x[(oid, line, w, dd)]
                for (oid, line, w, dd) in cost_dict.keys()
                if w == wh and dd == d
            ) <= MAX_CAPACITY

    # ===== 8. Giải =====
    model.solve(pulp.PULP_CBC_CMD(msg=False))

    # ===== 9. Lấy kết quả =====
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

    # ===== 10. Thống kê =====
    orders_per_day_wh = result_df.groupby(['delivery_date', 'warehouse']).size().reset_index(name='num_orders')

    # Hiển thị
    st.subheader("📦 Lịch giao hàng tối ưu")
    st.dataframe(result_df)

    st.subheader("📊 Số đơn mỗi ngày tại từng kho")
    st.dataframe(orders_per_day_wh)

    total_cost = result_df['shipping_cost'].sum()
    st.success(f"💰 Tổng chi phí giao hàng: {total_cost:,.2f}")
