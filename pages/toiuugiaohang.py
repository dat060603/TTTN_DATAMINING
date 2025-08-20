import streamlit as st
import pandas as pd
import numpy as np
from pulp import LpProblem, LpVariable, lpSum, LpMinimize, LpBinary, PULP_CBC_CMD
from io import BytesIO
from components.data_loader import load_data   # bạn đã có sẵn
import matplotlib.pyplot as plt
def app():
    st.title("🚚 So sánh mô phỏng giao hàng (Capacity=18 vs Gốc)")

    # --- Load dữ liệu gốc ---
    df = load_data()
    df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'])
    df = df.sort_values('ORDERDATE').reset_index(drop=True)
    st.write("📦 Tổng số đơn:", len(df))

    # --- Tham số ---
    capacity = 18
    min_delay_days = 3
    max_extend_days = 7

    if st.button("▶️ Chạy mô phỏng Capacity=18"):
        # Khởi tạo mô hình tối ưu
        model = LpProblem("Simulate_Shipping", LpMinimize)
        assign = {}
        for i in df.index:
            start_date = df.loc[i, 'ORDERDATE'] + pd.Timedelta(days=min_delay_days)
            end_date   = df.loc[i, 'ORDERDATE'] + pd.Timedelta(days=max_extend_days)
            for d in pd.date_range(start_date, end_date):
                assign[(i, d)] = LpVariable(f"assign_{i}_{d.date()}", cat=LpBinary)

        # Hàm mục tiêu: giao càng sớm càng tốt
        model += lpSum((d - df.loc[i, 'ORDERDATE']).days * assign[(i, d)] for (i, d) in assign)

        # Ràng buộc: mỗi đơn giao đúng 1 ngày
        for i in df.index:
            model += lpSum(assign[(ii, d)] for (ii, d) in assign if ii == i) == 1

        # Ràng buộc: capacity mỗi ngày
        all_days = sorted(set(d for (_, d) in assign))
        for d in all_days:
            model += lpSum(assign[(i, dd)] for (i, dd) in assign if dd == d) <= capacity

        # Giải mô hình
        st.info("⏳ Đang giải mô hình tối ưu...")
        model.solve(PULP_CBC_CMD(msg=False, timeLimit=60))
        st.success("✅ Giải xong!")

        # Lấy kết quả SIM_SHIP_DATE (mới)
        sim_ship_dates = []
        for i in df.index:
            ship_date = None
            start_date = df.loc[i, 'ORDERDATE'] + pd.Timedelta(days=min_delay_days)
            end_date   = df.loc[i, 'ORDERDATE'] + pd.Timedelta(days=max_extend_days)
            for d in pd.date_range(start_date, end_date):
                if assign[(i, d)].value() == 1:
                    ship_date = d
                    break
            sim_ship_dates.append(ship_date)

        df["SIM_SHIP_DATE_CAP18"] = sim_ship_dates

        # Nếu dataset gốc đã có SIM_SHIP_DATE cũ thì giữ lại để so sánh
        if "SIM_SHIP_DATE" in df.columns:
            df["SIM_SHIP_DATE"] = pd.to_datetime(df["SIM_SHIP_DATE"])
            df["SHIP_DAYS_CAP_OLD"] = (df["SIM_SHIP_DATE"] - df["ORDERDATE"]).dt.days
        else:
            st.warning("⚠️ Dataset gốc chưa có SIM_SHIP_DATE để so sánh!")

        # Tính số ngày giao hàng mới
        df["SHIP_DAYS_CAP18"] = (df["SIM_SHIP_DATE_CAP18"] - df["ORDERDATE"]).dt.days

        # Hiển thị bảng preview
        st.subheader("🔍 Xem trước dữ liệu")
        st.dataframe(df[["ORDERNUMBER", "ORDERDATE", "SIM_SHIP_DATE", "SIM_SHIP_DATE_CAP18", "SHIP_DAYS_CAP_OLD", "SHIP_DAYS_CAP18"]].head(20))

        # Thống kê số lượng đơn theo số ngày giao hàng từng trường hợp
        st.subheader("📋 Thống kê số lượng đơn theo số ngày giao hàng")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Gốc**")
            if "SHIP_DAYS_CAP_OLD" in df:
                st.dataframe(df["SHIP_DAYS_CAP_OLD"].value_counts().sort_index().rename_axis("Số ngày giao hàng").reset_index(name="Số đơn"))
            else:
                st.info("Không có dữ liệu giao hàng gốc để thống kê.")
        with col2:
            st.markdown("**Capacity=18**")
            st.dataframe(df["SHIP_DAYS_CAP18"].value_counts().sort_index().rename_axis("Số ngày giao hàng").reset_index(name="Số đơn"))

        # Biểu đồ histogram so sánh
        st.subheader("📊 So sánh phân phối số ngày giao hàng")
        fig, ax = plt.subplots(figsize=(10,6))
        if "SHIP_DAYS_CAP_OLD" in df:
            ax.hist(df["SHIP_DAYS_CAP_OLD"].dropna(), bins=range(2,12), alpha=0.6, label="Gốc")
        ax.hist(df["SHIP_DAYS_CAP18"], bins=range(2,12), alpha=0.6, label="Capacity=18")
        ax.set_xlabel("Số ngày giao hàng")
        ax.set_ylabel("Số lượng đơn hàng")
        ax.set_title("So sánh số ngày giao hàng: Gốc vs Capacity=18")
        ax.legend()
        st.pyplot(fig)
