
# pages/06_Simulate_WhatIf.py
# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import plotly.graph_objects as go
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor
# from xgboost import XGBRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
# from components.data_loader import load_data
#
# def app():
#     # --- Load CSS ---
#     def local_css(file_name):
#         with open(file_name) as f:
#             st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
#     local_css("style.css")
#
#     st.set_page_config(page_title="Simulate & What-if", layout="wide")
#     st.title("🔄 Simulate & What-if – Mô phỏng kịch bản kinh doanh")
#
#     # --- Load data ---
#     df = load_data()
#     df = df.dropna(subset=['PRICEEACH', 'QUANTITYORDERED', 'MSRP', 'MONTH_ID'])
#
#     # Tabs
#     tab1, tab2, tab3 = st.tabs(["⚙️ Thiết lập kịch bản", "🧠 Mô hình", "📊 Kết quả & Biểu đồ"])
#
#     # ---------- TAB 1: Thiết lập ----------
#     with tab1:
#         st.header("⚙️ 1. Thiết lập kịch bản mô phỏng")
#         productlines = st.multiselect("Chọn PRODUCTLINE để mô phỏng", df['PRODUCTLINE'].unique(), default=["Classic Cars"])
#         df_filtered = df[df['PRODUCTLINE'].isin(productlines)].copy()
#
#         col1, col2, col3 = st.columns(3)
#         with col1:
#             price_change_pct = st.slider("Thay đổi giá bán (%)", -50, 50, 0)
#         with col2:
#             discount_change_pct = st.slider("Thay đổi chiết khấu (%)", -50, 50, 0)
#         with col3:
#             use_custom_cost = st.checkbox("Tùy chỉnh tỷ lệ chi phí theo PRODUCTLINE", value=False)
#
#         # Clone data
#         df_sim = df_filtered.copy()
#         df_sim['PRICEEACH_NEW'] = df_sim['PRICEEACH'] * (1 + price_change_pct / 100) * (1 - discount_change_pct / 100)
#         df_sim['DISCOUNT_NEW'] = discount_change_pct
#
#         if use_custom_cost:
#             st.subheader("🧮 Nhập tỷ lệ chi phí cho từng PRODUCTLINE (% trên SALES)")
#             cost_map = {}
#             for pl in productlines:
#                 cost_map[pl] = st.slider(f"{pl}", 30, 100, 70, key=f"cost_{pl}")
#             df_sim['COST_PCT'] = df_sim['PRODUCTLINE'].map(cost_map)
#         else:
#             default_cost_pct = st.slider("Tỷ lệ chi phí (% trên SALES)", 30, 100, 70)
#             df_sim['COST_PCT'] = default_cost_pct
#
#         st.toast("✅ Kịch bản đã được thiết lập, chuyển sang tab 'Mô hình' để chạy mô phỏng.")
#
#     # ---------- TAB 2: Mô hình ----------
#     with tab2:
#         st.header("🧠 2. Cấu hình mô hình học máy (tùy chọn)")
#         mode = st.radio("🔧 Chọn phương pháp mô phỏng", ["🔹 Rule-based", "🔸 Machine Learning"], horizontal=True)
#
#         if mode == "🔸 Machine Learning":
#             ml_model_choice = st.selectbox("Chọn mô hình", ["Linear Regression", "Random Forest", "XGBoost"])
#             compare_models = st.checkbox("📊 So sánh các mô hình")
#
#             df_encoded = pd.get_dummies(df_sim, columns=['PRODUCTLINE'], drop_first=True)
#             feature_cols_all = ['PRICEEACH_NEW', 'MSRP', 'MONTH_ID'] + [col for col in df_encoded.columns if col.startswith('PRODUCTLINE_')]
#             selected_features = st.multiselect("Chọn biến đầu vào", feature_cols_all, default=['PRICEEACH_NEW', 'MSRP', 'MONTH_ID'])
#
#             X = df_encoded[selected_features]
#             y = df_encoded['QUANTITYORDERED']
#             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
#             models = {
#                 "Linear Regression": LinearRegression(),
#                 "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
#                 "XGBoost": XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
#             }
#
#             if compare_models:
#                 results = []
#                 for name, mdl in models.items():
#                     mdl.fit(X_train, y_train)
#                     preds = mdl.predict(X_test)
#                     results.append({
#                         "Mô hình": name,
#                         "R2 Score": r2_score(y_test, preds),
#                         "MAE": mean_absolute_error(y_test, preds),
#                         "RMSE": np.sqrt(mean_squared_error(y_test, preds))
#                     })
#
#                 results_df = pd.DataFrame(results)
#                 st.dataframe(results_df.style.format({"R2 Score": "{:.3f}", "MAE": "{:.2f}", "RMSE": "{:.2f}"}))
#                 best_model = results_df.sort_values(by="R2 Score", ascending=False).iloc[0]
#                 st.success(f"✅ Mô hình tốt nhất: **{best_model['Mô hình']}** (R²={best_model['R2 Score']:.3f})")
#
#             # Train selected model
#             model = models[ml_model_choice]
#             model.fit(X_train, y_train)
#             df_sim['QUANTITYORDERED_NEW'] = model.predict(X[selected_features])
#         else:
#             df_sim['QUANTITYORDERED_NEW'] = df_sim['QUANTITYORDERED']
#
#         st.toast("✅ Mô hình đã chạy xong, chuyển sang tab 'Kết quả' để xem phân tích.")
#
#     # ---------- TAB 3: Kết quả ----------
#     with tab3:
#         st.header("📊 3. Kết quả mô phỏng")
#         df_sim['SALES_NEW'] = df_sim['PRICEEACH_NEW'] * df_sim['QUANTITYORDERED_NEW']
#         df_sim['COST_NEW'] = df_sim['SALES_NEW'] * (df_sim['COST_PCT'] / 100)
#         df_sim['PROFIT_NEW'] = df_sim['SALES_NEW'] - df_sim['COST_NEW']
#
#         summary = []
#         for pl in productlines:
#             df_orig_pl = df_filtered[df_filtered['PRODUCTLINE'] == pl]
#             df_sim_pl = df_sim[df_sim['PRODUCTLINE'] == pl]
#
#             orig_sales = df_orig_pl['SALES'].sum()
#             orig_profit = orig_sales - orig_sales * (df_sim_pl['COST_PCT'].iloc[0] / 100)
#             orig_qty = df_orig_pl['QUANTITYORDERED'].sum()
#
#             sim_sales = df_sim_pl['SALES_NEW'].sum()
#             sim_profit = df_sim_pl['PROFIT_NEW'].sum()
#             sim_qty = df_sim_pl['QUANTITYORDERED_NEW'].sum()
#
#             summary.append({
#                 "PRODUCTLINE": pl,
#                 "Doanh thu gốc": orig_sales,
#                 "Doanh thu mô phỏng": sim_sales,
#                 "% thay đổi doanh thu": 100 * (sim_sales - orig_sales) / orig_sales if orig_sales else 0,
#                 "Lợi nhuận gốc": orig_profit,
#                 "Lợi nhuận mô phỏng": sim_profit,
#                 "% thay đổi lợi nhuận": 100 * (sim_profit - orig_profit) / orig_profit if orig_profit else 0,
#                 "SL gốc": orig_qty,
#                 "SL mô phỏng": sim_qty,
#                 "% thay đổi SL": 100 * (sim_qty - orig_qty) / orig_qty if orig_qty else 0
#             })
#
#         summary_df = pd.DataFrame(summary)
#         st.dataframe(summary_df.style.format({
#             "Doanh thu gốc": "${:,.2f}",
#             "Doanh thu mô phỏng": "${:,.2f}",
#             "% thay đổi doanh thu": "{:.2f}%",
#             "Lợi nhuận gốc": "${:,.2f}",
#             "Lợi nhuận mô phỏng": "${:,.2f}",
#             "% thay đổi lợi nhuận": "{:.2f}%",
#             "SL gốc": "{:,.0f}",
#             "SL mô phỏng": "{:,.0f}",
#             "% thay đổi SL": "{:.2f}%"
#         }), use_container_width=True)
#
#         # Biểu đồ so sánh
#         st.subheader("📈 So sánh doanh thu & lợi nhuận")
#         fig = go.Figure()
#         fig.add_trace(go.Bar(name="Doanh thu gốc", x=summary_df['PRODUCTLINE'], y=summary_df['Doanh thu gốc']))
#         fig.add_trace(go.Bar(name="Doanh thu mô phỏng", x=summary_df['PRODUCTLINE'], y=summary_df['Doanh thu mô phỏng']))
#         fig.add_trace(go.Bar(name="Lợi nhuận gốc", x=summary_df['PRODUCTLINE'], y=summary_df['Lợi nhuận gốc']))
#         fig.add_trace(go.Bar(name="Lợi nhuận mô phỏng", x=summary_df['PRODUCTLINE'], y=summary_df['Lợi nhuận mô phỏng']))
#         fig.update_layout(barmode='group', xaxis_title="Product Line", yaxis_title="Giá trị ($)")
#         st.plotly_chart(fig, use_container_width=True)
#
#         # Biểu đồ % thay đổi
#         st.subheader("📊 % Thay đổi")
#         fig_change = px.bar(summary_df.melt(id_vars=["PRODUCTLINE"],
#                                             value_vars=["% thay đổi doanh thu", "% thay đổi lợi nhuận", "% thay đổi SL"]),
#                             x="PRODUCTLINE", y="value", color="variable", barmode="group",
#                             labels={"value": "% thay đổi", "variable": "Chỉ số"})
#         st.plotly_chart(fig_change, use_container_width=True)
#
#         # Download kết quả
#         st.download_button("📥 Tải kết quả CSV", data=summary_df.to_csv(index=False), file_name="simulate_result.csv", mime="text/csv")
# phần new
# pages/06_Simulate_WhatIf.py
"""
Simulate & What-if — Streamlit page
Phiên bản nâng cấp theo yêu cầu:
- Inputs: Price Change %, Cost Change %, Demand Change %, Discount Rate, Marketing Budget Increase %, Marketing Elasticity %
- Filters: ProductLine, Country, Year, ProductCode (từng sản phẩm)
- Simulation engine: áp dụng thay đổi để tạo dataset giả định và tính Revenue / Profit mới
- Outputs: KPIs, biểu đồ so sánh, choropleth bản đồ thị trường (nếu có COUNTRY), top products/regions, insight tự động, cảnh báo lợi nhuận < 0

Lưu ý: file giả định bạn có hàm `load_data()` trong components.data_loader trả về dataframe với cột tối thiểu:
['ORDERNUMBER','QUANTITYORDERED','PRICEEACH','SALES','ORDERDATE','PRODUCTLINE','PRODUCTCODE','COUNTRY','YEAR_ID','MONTH_ID']
Nếu không có cột COST, ta giả định COST = SALES * (cost_pct_default / 100) hoặc COST = PRICEEACH * cost_ratio.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from components.data_loader import load_data

# Page config
st.set_page_config(page_title="Simulate & What-if", layout="wide")

# Helper functions
@st.cache_data
def load_and_prepare():
    df = load_data()  # your loader
    df = df.copy()

    if 'ORDERDATE' in df.columns:
        try:
            df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'])
        except Exception:
            pass
    if 'YEAR_ID' not in df.columns and 'ORDERDATE' in df.columns:
        df['YEAR_ID'] = df['ORDERDATE'].dt.year
    for col in ['PRICEEACH', 'QUANTITYORDERED', 'SALES']:
        if col not in df.columns:
            df[col] = 0
    if 'PRODUCTLINE' not in df.columns:
        df['PRODUCTLINE'] = 'Unknown'
    if 'COUNTRY' not in df.columns:
        df['COUNTRY'] = 'Unknown'
    if 'PRODUCTCODE' not in df.columns:
        df['PRODUCTCODE'] = df.index.astype(str)
    return df
def app():
    st.title("🔄 Simulate & What-if — Business Scenario Simulator")
    st.markdown("Ứng dụng mô phỏng kịch bản kinh doanh: thay đổi giá, chi phí, nhu cầu, marketing, discount...")

    df = load_and_prepare()

    # Bộ lọc & Tham số mô phỏng
    with st.expander("🛠 Bộ lọc & Tham số mô phỏng", expanded=True):
        with st.form("filters_form"):
            st.subheader("Bộ lọc dữ liệu")

            productlines = st.multiselect(
                "ProductLine",
                sorted(df['PRODUCTLINE'].unique()),
                default=sorted(df['PRODUCTLINE'].unique())[:2]
            )
            countries = st.multiselect(
                "Country",
                sorted(df['COUNTRY'].fillna('Unknown').unique()),
                default=sorted(df['COUNTRY'].fillna('Unknown').unique())[:3]
            )
            years = st.multiselect(
                "Year",
                sorted(df['YEAR_ID'].unique()),
                default=sorted(df['YEAR_ID'].unique())[:3]
            )
            # Cập nhật ProductCode theo ProductLine
            if productlines:
                available_products = sorted(df[df['PRODUCTLINE'].isin(productlines)]['PRODUCTCODE'].unique())
            else:
                available_products = sorted(df['PRODUCTCODE'].unique())
            products = st.multiselect("ProductCode (tùy chọn)", available_products)
            st.markdown("---")
            st.markdown("---")
            st.subheader("Giả định chi phí (nếu không có COST trong data)")
            cost_pct_default = st.slider("Tỷ lệ chi phí trung bình trên Sales (%)", 0, 100, 50)
            cost_ratio_price = st.slider("Tỷ lệ chi phí trên PRICEEACH (nếu muốn) (%)", 0, 100, 30)

            st.markdown("---")
            st.subheader("Input: Thay đổi kịch bản")
            price_change_pct = st.slider("% Thay đổi giá bán (Price Change %)", -80, 200, 0)
            cost_change_pct = st.slider("% Thay đổi chi phí (Cost Change %)", -80, 200, 0)
            demand_change_pct = st.slider("% Thay đổi số lượng (Demand Change %)", -90, 500, 0)
            discount_rate = st.slider("Tỷ lệ chiết khấu / khuyến mãi (%)", 0, 100, 0)

            st.markdown("---")
            st.subheader("Marketing")
            marketing_budget_pct = st.slider("Ngân sách Marketing tăng thêm (%)", -100, 1000, 0)
            marketing_elasticity = st.slider("Tỷ lệ tác động Marketing lên nhu cầu (elasticity %)", 0, 500, 10)

            st.markdown("---")
            submitted = st.form_submit_button("🚀 Chạy mô phỏng")
    if not submitted:
        st.info("⬆️ Chọn bộ lọc và tham số mô phỏng, sau đó bấm **Chạy mô phỏng**.")
        return
    # Filter df
    df_filtered = df.copy()
    if productlines:
        df_filtered = df_filtered[df_filtered['PRODUCTLINE'].isin(productlines)]
    if countries:
        df_filtered = df_filtered[df_filtered['COUNTRY'].isin(countries)]
    if years:
        df_filtered = df_filtered[df_filtered['YEAR_ID'].isin(years)]
    if products:
        if len(products) > 0:
            df_filtered = df_filtered[df_filtered['PRODUCTCODE'].isin(products)]
    if df_filtered.empty:
        st.warning("Không có dữ liệu sau khi lọc — vui lòng điều chỉnh bộ lọc.")
        return
    # Baseline metrics
    baseline = {}
    baseline['total_sales'] = df_filtered['SALES'].sum()
    baseline['total_qty'] = df_filtered['QUANTITYORDERED'].sum()
    if 'COST' in df_filtered.columns:
        baseline['total_cost'] = df_filtered['COST'].sum()
    else:
        baseline['total_cost'] = baseline['total_sales'] * (cost_pct_default / 100.0)
    baseline['profit'] = baseline['total_sales'] - baseline['total_cost']

    # Simulation
    df_sim = df_filtered.copy()
    if 'COST' not in df_sim.columns:
        df_sim['COST_EST_PER_UNIT'] = df_sim['PRICEEACH'] * (cost_ratio_price / 100.0)
        df_sim['COST'] = df_sim['COST_EST_PER_UNIT'] * df_sim['QUANTITYORDERED']

    df_sim['PRICE_NEW'] = df_sim['PRICEEACH'] * (1 + price_change_pct / 100.0)
    df_sim['PRICE_NEW'] *= (1 - discount_rate / 100.0)

    df_sim['COST_PER_UNIT_OLD'] = np.where(df_sim['QUANTITYORDERED'] > 0,
                                           df_sim['COST'] / df_sim['QUANTITYORDERED'],
                                           df_sim['PRICEEACH'] * (cost_ratio_price / 100.0))
    df_sim['COST_PER_UNIT_NEW'] = df_sim['COST_PER_UNIT_OLD'] * (1 + cost_change_pct / 100.0)

    demand_from_marketing_pct = marketing_budget_pct * (marketing_elasticity / 100.0)
    df_sim['QUANTITY_NEW'] = df_sim['QUANTITYORDERED'] * (
        1 + demand_change_pct / 100.0 + demand_from_marketing_pct / 100.0
    )
    df_sim['QUANTITY_NEW'] = df_sim['QUANTITY_NEW'].clip(lower=0)

    df_sim['SALES_NEW'] = df_sim['PRICE_NEW'] * df_sim['QUANTITY_NEW']
    df_sim['COST_NEW'] = df_sim['COST_PER_UNIT_NEW'] * df_sim['QUANTITY_NEW']
    df_sim['PROFIT_NEW'] = df_sim['SALES_NEW'] - df_sim['COST_NEW']

    # Aggregation
    agg_sim = df_sim.groupby('PRODUCTLINE').agg(
        sales_orig=('SALES', 'sum'),
        sales_new=('SALES_NEW', 'sum'),
        qty_orig=('QUANTITYORDERED', 'sum'),
        qty_new=('QUANTITY_NEW', 'sum'),
        profit_new=('PROFIT_NEW', 'sum')
    ).reset_index()

    profit_by_pl = df_sim.groupby('PRODUCTLINE').apply(lambda d: d['SALES'].sum() - d['COST'].sum()).rename('profit_orig_correct')
    agg_sim = agg_sim.merge(profit_by_pl.reset_index(), on='PRODUCTLINE', how='left')

    agg_sim['%change_sales'] = np.where(agg_sim['sales_orig'] != 0,
                                        100 * (agg_sim['sales_new'] - agg_sim['sales_orig']) / agg_sim['sales_orig'], np.nan)
    agg_sim['%change_profit'] = np.where(agg_sim['profit_orig_correct'] != 0,
                                         100 * (agg_sim['profit_new'] - agg_sim['profit_orig_correct']) / agg_sim['profit_orig_correct'], np.nan)
    # KPIs
    total_sales_new = df_sim['SALES_NEW'].sum()
    total_profit_new = df_sim['PROFIT_NEW'].sum()
    total_qty_new = df_sim['QUANTITY_NEW'].sum()

    sales_change_pct = 100 * (total_sales_new - baseline['total_sales']) / baseline['total_sales'] if baseline['total_sales'] else np.nan
    profit_change_pct = 100 * (total_profit_new - baseline['profit']) / baseline['profit'] if baseline['profit'] else np.nan
    st.subheader("📊 KPIs chính (Tổng)")
    col1, col2, col3 = st.columns(3)
    col1.metric("Doanh thu (gốc)", f"${baseline['total_sales']:,.2f}")
    col2.metric("Doanh thu (mô phỏng)", f"${total_sales_new:,.2f}", f"{sales_change_pct:.2f}%")
    col3.metric("Tổng SL (gốc → mô phỏng)", f"{int(baseline['total_qty']):,}", f"{(total_qty_new - baseline['total_qty']):,.0f}")
    col4, col5 = st.columns(2)
    col4.metric("Lợi nhuận (gốc)", f"${baseline['profit']:,.2f}")
    col5.metric("Lợi nhuận (mô phỏng)", f"${total_profit_new:,.2f}", f"{profit_change_pct:.2f}%")
    if total_profit_new < 0:
        st.error("⚠️ Lợi nhuận tổng sau mô phỏng < 0 — xem lại kịch bản.")
    # Insights
    st.subheader("🔍 Phân tích & Insight tự động")
    top_pl_sales = agg_sim.sort_values('%change_sales', ascending=False).iloc[0]
    worst_pl_sales = agg_sim.sort_values('%change_sales', ascending=True).iloc[0]
    top_pl_profit = agg_sim.sort_values('%change_profit', ascending=False).iloc[0]
    st.markdown(f"- 📈 **Tăng trưởng doanh thu mạnh nhất**: **{top_pl_sales['PRODUCTLINE']}** — {top_pl_sales['%change_sales']:.2f}%")
    st.markdown(f"- 📉 **Giảm nhiều nhất**: **{worst_pl_sales['PRODUCTLINE']}** — {worst_pl_sales['%change_sales']:.2f}%")
    st.markdown(f"- 💰 **Tăng lợi nhuận mạnh nhất**: **{top_pl_profit['PRODUCTLINE']}** — {top_pl_profit['%change_profit']:.2f}%")
    heuristics = []
    if price_change_pct > 0 and demand_change_pct < 0:
        heuristics.append("Giá tăng có thể làm giảm sản lượng — cân nhắc giảm giá cho nhóm nhạy cảm.")
    if discount_rate > 0 and marketing_budget_pct > 0:
        heuristics.append("Giảm giá + marketing có thể tăng sản lượng nhưng biên lợi nhuận cần theo dõi.")
    if marketing_budget_pct > 0 and marketing_elasticity <= 0:
        heuristics.append("Elasticity nhỏ/âm => marketing có thể không hiệu quả.")
    for h in heuristics:
        st.info(h)
    # tự động
    # === Giải thích kết quả mô phỏng ===
    st.subheader("🧠 Diễn giải mô phỏng & Gợi ý hành động")
    if sales_change_pct > 0 and profit_change_pct < 0:
        st.warning(
            "🚨 Doanh thu tăng nhưng lợi nhuận giảm — có thể do chi phí tăng hoặc mức chiết khấu/marketing quá cao.")
    elif sales_change_pct < 0 and profit_change_pct > 0:
        st.info("💡 Lợi nhuận tăng dù doanh thu giảm — bạn có thể đã cắt giảm chi phí hiệu quả.")
    elif sales_change_pct > 0 and profit_change_pct > 0:
        st.success("✅ Cả doanh thu và lợi nhuận đều tăng — đây là một kịch bản khả quan.")
    elif sales_change_pct < 0 and profit_change_pct < 0:
        st.error("⚠️ Cả doanh thu và lợi nhuận đều giảm — nên điều chỉnh lại kịch bản.")

    # === Gợi ý hành động dựa trên mô phỏng ===
    suggestions = []
    if price_change_pct > 20:
        suggestions.append("📌 Giá tăng mạnh — hãy đảm bảo rằng thị trường chấp nhận mức giá mới.")
    if discount_rate > 30:
        suggestions.append("📌 Mức chiết khấu cao — cần kiểm tra lại biên lợi nhuận.")
    if marketing_budget_pct > 50 and marketing_elasticity < 20:
        suggestions.append("📌 Ngân sách marketing lớn nhưng độ nhạy thấp — có thể không hiệu quả.")
    if demand_change_pct > 100:
        suggestions.append("📌 Dự đoán nhu cầu tăng quá cao — cần đánh giá tính khả thi.")

    if suggestions:
        st.subheader("🤖 Gợi ý từ hệ thống")
        for s in suggestions:
            st.info(s)
    else:
        st.success("✅ Không có cảnh báo — kịch bản mô phỏng hợp lý.")

    # Charts
    st.subheader("📊 Biểu đồ so sánh theo ProductLine")
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Sales Baseline', x=agg_sim['PRODUCTLINE'], y=agg_sim['sales_orig']))
    fig.add_trace(go.Bar(name='Sales Sim', x=agg_sim['PRODUCTLINE'], y=agg_sim['sales_new']))
    fig.add_trace(go.Bar(name='Profit Baseline', x=agg_sim['PRODUCTLINE'], y=agg_sim['profit_orig_correct']))
    fig.add_trace(go.Bar(name='Profit Sim', x=agg_sim['PRODUCTLINE'], y=agg_sim['profit_new']))
    fig.update_layout(barmode='group', xaxis_title='ProductLine', yaxis_title='USD')
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("% Thay đổi theo ProductLine")
    melt = agg_sim[['PRODUCTLINE', '%change_sales', '%change_profit']].melt(id_vars='PRODUCTLINE')
    fig2 = px.bar(melt, x='PRODUCTLINE', y='value', color='variable', barmode='group',
                  labels={'value': '% change', 'variable': 'Metric'})
    st.plotly_chart(fig2, use_container_width=True)
    st.subheader("Top sản phẩm tăng trưởng mạnh nhất (theo doanh thu)")
    prod_growth = df_sim.groupby('PRODUCTCODE').agg(sales_orig=('SALES', 'sum'),
                                                    sales_new=('SALES_NEW', 'sum')).reset_index()
    prod_growth['pct_change'] = np.where(prod_growth['sales_orig'] != 0,
                                         100 * (prod_growth['sales_new'] - prod_growth['sales_orig']) / prod_growth['sales_orig'], np.nan)
    top_products = prod_growth.sort_values('pct_change', ascending=False).head(10)
    st.dataframe(top_products.style.format({'sales_orig': '${:,.2f}',
                                            'sales_new': '${:,.2f}',
                                            'pct_change': '{:.2f}%'}))
    if 'COUNTRY' in df_sim.columns and df_sim['COUNTRY'].nunique() > 1:
        st.subheader("🗺 Bản đồ thị trường — thay đổi doanh thu theo quốc gia")
        country_agg = df_sim.groupby('COUNTRY').agg(sales_orig=('SALES', 'sum'),
                                                    sales_new=('SALES_NEW', 'sum')).reset_index()
        country_agg['pct_change'] = np.where(country_agg['sales_orig'] != 0,
                                             100 * (country_agg['sales_new'] - country_agg['sales_orig']) / country_agg['sales_orig'], 0)
        try:
            fig_map = px.choropleth(country_agg, locations='COUNTRY', locationmode='country names',
                                    color='pct_change', hover_name='COUNTRY', color_continuous_scale='RdYlGn')
            fig_map.update_layout(coloraxis_colorbar={'title': '% change Sales'})
            st.plotly_chart(fig_map, use_container_width=True)
        except:
            st.warning('Không thể vẽ bản đồ — kiểm tra tên quốc gia hoặc cung cấp ISO codes.')
    # Download results
    st.subheader("📥 Tải kết quả")
    export_df = df_sim[['PRODUCTLINE', 'PRODUCTCODE', 'COUNTRY', 'PRICEEACH', 'PRICE_NEW',
                        'QUANTITYORDERED', 'QUANTITY_NEW', 'SALES', 'SALES_NEW', 'COST', 'COST_NEW', 'PROFIT_NEW']]
    st.download_button("📥 Tải CSV kết quả chi tiết", data=export_df.to_csv(index=False),
                       file_name='simulate_detailed.csv', mime='text/csv')
if __name__ == '__main__':
    app()



