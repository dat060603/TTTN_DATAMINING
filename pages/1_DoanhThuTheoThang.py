import streamlit as st
import pandas as pd
import plotly.express as px
from components.data_loader import load_data

# Tiêu đề trang
st.title("📈1_ Doanh thu theo tháng")

# Load dữ liệu
df = load_data()

if "MONTH" not in df.columns:
    df["ORDERDATE"] = pd.to_datetime(df["ORDERDATE"])
    df["MONTH"] = df["ORDERDATE"].dt.to_period("M").astype(str)

monthly_sales = df.groupby("MONTH")["SALES"].sum().reset_index()

fig = px.line(monthly_sales, x="MONTH", y="SALES",
              title="📆 Tổng doanh thu theo tháng",
              markers=True)
st.plotly_chart(fig, use_container_width=True)
