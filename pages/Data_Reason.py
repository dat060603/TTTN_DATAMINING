import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy import stats
import statsmodels.formula.api as smf

from components.data_loader import load_data

def compute_rmse(y_true, y_pred):
    try:
        return mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        return np.sqrt(mean_squared_error(y_true, y_pred))

def app():
    # ======= UI setup =======
    st.set_page_config(page_title="Data Reason — Phân tích mối quan hệ", layout="wide")
    st.title("🔍 Data Reason — Phân tích mối quan hệ giữa các biến")

    # ======= Load CSS =======
    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    local_css("style.css")

    # ======= Load data =======
    df = load_data()

    # ======= Filters =======
    with st.expander("🔎 Bộ lọc (Data Reason)", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            years = sorted(df['YEAR_ID'].dropna().unique()) if 'YEAR_ID' in df.columns else []
            selected_year = st.selectbox("Chọn năm", options=years, index=len(years)-1 if years else 0) if years else None
        with c2:
            countries = sorted(df['COUNTRY'].dropna().unique()) if 'COUNTRY' in df.columns else []
            selected_country = st.multiselect("Chọn quốc gia", options=countries, default=countries if countries else [])
        with c3:
            product_lines = sorted(df['PRODUCTLINE'].dropna().unique()) if 'PRODUCTLINE' in df.columns else []
            selected_productline = st.multiselect("Chọn PRODUCTLINE", options=product_lines, default=product_lines if product_lines else [])

    # ======= Apply filters =======
    _data = df.copy()
    if selected_year is not None:
        _data = _data[_data['YEAR_ID'] == selected_year]
    if selected_country:
        _data = _data[_data['COUNTRY'].isin(selected_country)]
    if selected_productline:
        _data = _data[_data['PRODUCTLINE'].isin(selected_productline)]

    # Chuyển đổi numeric
    for c in ['SALES', 'PRICEEACH', 'QUANTITYORDERED', 'MSRP']:
        if c in _data.columns:
            _data[c] = pd.to_numeric(_data[c], errors='coerce')

    # ======= Tabs =======
    tabs = st.tabs([
        "1️⃣ Correlation Heatmap",
        "2️⃣ Scatter + Trendline",
        "3️⃣ SALES Diff",
        "4️⃣ Regression Summary",
        "5️⃣ Group Analysis"
    ])

    # ===========================
    # Tab 1: Correlation heatmap
    # ===========================
    with tabs[0]:
        st.subheader("1️⃣ Ma trận tương quan giữa các biến")
        corr_cols = [c for c in ['SALES', 'QUANTITYORDERED', 'PRICEEACH', 'MSRP'] if c in _data.columns]
        if len(corr_cols) >= 2:
            corr_df = _data[corr_cols].corr()
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_df.values, x=corr_df.columns, y=corr_df.columns,
                colorscale='RdBu', zmin=-1, zmax=1, text=np.round(corr_df.values, 2), texttemplate="%{text}"
            ))
            fig_corr.update_layout(title="Ma trận tương quan", width=700, height=500)
            st.plotly_chart(fig_corr, use_container_width=False)
            st.dataframe(corr_df.style.format("{:.3f}"))
        else:
            st.info("Cần ít nhất 2 cột numeric.")

    # ===============================
    # Tab 2: Scatter + Trendline OLS
    # ===============================
    with tabs[1]:
        st.subheader("2️⃣ Scatter PRICEEACH vs QUANTITYORDERED")
        if 'PRICEEACH' in _data.columns and 'QUANTITYORDERED' in _data.columns:
            st.markdown("**Tùy chọn hiển thị**")
            col1, col2 = st.columns([2, 1])
            with col1:
                show_trend = st.checkbox("Hiển thị trendline OLS", value=True)
                use_log = st.checkbox("Log-log scale", help="Dùng log(PRICEEACH), log(QUANTITYORDERED)")
            with col2:
                sample_slider = st.slider("Lọc số dòng hiển thị", min_value=500, max_value=20000, value=5000, step=500)

            plot_df = _data.dropna(subset=['PRICEEACH', 'QUANTITYORDERED'])
            if use_log:
                plot_df = plot_df[(plot_df['PRICEEACH'] > 0) & (plot_df['QUANTITYORDERED'] > 0)]

            if len(plot_df) > sample_slider:
                plot_df = plot_df.sample(sample_slider, random_state=42)

            facet_col = 'PRODUCTLINE' if plot_df['PRODUCTLINE'].nunique() <= 4 else None

            fig = px.scatter(
                plot_df,
                x=np.log(plot_df['PRICEEACH']) if use_log else plot_df['PRICEEACH'],
                y=np.log(plot_df['QUANTITYORDERED']) if use_log else plot_df['QUANTITYORDERED'],
                color='PRODUCTLINE' if 'PRODUCTLINE' in plot_df.columns else None,
                facet_col=facet_col,
                labels={'x': 'log(PRICEEACH)' if use_log else 'PRICEEACH',
                        'y': 'log(QUANTITYORDERED)' if use_log else 'QUANTITYORDERED'}
            )
            st.plotly_chart(fig, use_container_width=True)

            if show_trend:
                try:
                    X = np.log(plot_df['PRICEEACH']).values.reshape(-1, 1) if use_log else plot_df[['PRICEEACH']].values
                    y = np.log(plot_df['QUANTITYORDERED']).values if use_log else plot_df['QUANTITYORDERED'].values
                    model = LinearRegression().fit(X, y)
                    st.markdown(f"**OLS Trendline:** coef = {model.coef_[0]:.4f}, intercept = {model.intercept_:.4f}, R² = {r2_score(y, model.predict(X)):.4f}")
                except Exception as e:
                    st.warning(f"Lỗi khi fit trendline: {e}")
        else:
            st.warning("Thiếu PRICEEACH hoặc QUANTITYORDERED.")

    # =========================
    # Tab 3: SALES_DIFF phân tích
    # =========================
    with tabs[2]:
        st.subheader("3️⃣ Phân tích chênh lệch SALES vs ORDER_VALUE")
        if 'SALES' in _data.columns and all(col in _data.columns for col in ['QUANTITYORDERED', 'PRICEEACH']):
            _data['TOTAL_ORDER_VALUE'] = _data['QUANTITYORDERED'] * _data['PRICEEACH']
            _data['SALES_DIFF'] = _data['SALES'] - _data['TOTAL_ORDER_VALUE']
            desc = _data['SALES_DIFF'].describe()
            st.write(desc)
            thresh = st.number_input("Ngưỡng chênh lệch đáng kể", value=0.01)
            df_diff = _data[_data['SALES_DIFF'].abs() > thresh]
            st.markdown(f"Số dòng chênh lệch lớn: **{len(df_diff)}**")
            st.dataframe(df_diff.head(200))
            st.download_button("📥 Tải CSV", data=df_diff.to_csv(index=False).encode('utf-8'), file_name="sales_diff.csv")
        else:
            st.warning("Không đủ cột để tính SALES_DIFF.")

    # ===============================
    # Tab 4: Hồi quy đơn giản
    # ===============================
    with tabs[3]:
        st.subheader("4️⃣ Mô hình hồi quy đơn giản")
        reg_choice = st.radio("Chọn mô hình:", options=[
            "QUANTITYORDERED ~ PRICEEACH",
            "SALES ~ QUANTITYORDERED + PRICEEACH"
        ])
        if reg_choice == "QUANTITYORDERED ~ PRICEEACH":
            req_cols = ['PRICEEACH', 'QUANTITYORDERED']
            formula = "QUANTITYORDERED ~ PRICEEACH"
        else:
            req_cols = ['SALES', 'QUANTITYORDERED', 'PRICEEACH']
            formula = "SALES ~ QUANTITYORDERED + PRICEEACH"

        if all(c in _data.columns for c in req_cols):
            reg_df = _data[req_cols].dropna()
            if len(reg_df) >= 10:
                model = smf.ols(formula, data=reg_df).fit()
                st.text(model.summary().as_text())
                preds = model.predict(reg_df)
                st.write("MAE:", mean_absolute_error(reg_df[req_cols[0]], preds))
                st.write("RMSE:", compute_rmse(reg_df[req_cols[0]], preds))
            else:
                st.warning("Không đủ dữ liệu để chạy mô hình.")
        else:
            st.warning("Thiếu dữ liệu cần thiết.")

    # ===============================
    # Tab 5: Phân tích theo nhóm
    # ===============================
    with tabs[4]:
        st.subheader("5️⃣ Phân tích theo nhóm (PRODUCTLINE / DEALSIZE)")
        group_options = [g for g in ['PRODUCTLINE', 'DEALSIZE'] if g in _data.columns]
        if not group_options:
            st.warning("Không có biến nhóm.")
        else:
            group_by = st.selectbox("Nhóm theo:", options=group_options)
            chosen_groups = st.multiselect("Chọn nhóm:", options=_data[group_by].unique().tolist())
            sub = _data[_data[group_by].isin(chosen_groups)]

            if len(sub) == 0:
                st.warning("Không có dữ liệu nhóm.")
            else:
                st.plotly_chart(px.scatter(sub, x='PRICEEACH', y='QUANTITYORDERED', color=group_by), use_container_width=True)

                results = []
                for g in chosen_groups:
                    dfg = sub[sub[group_by] == g].dropna(subset=['PRICEEACH', 'QUANTITYORDERED'])
                    row = {'group': g, 'n': len(dfg)}
                    try:
                        pr, _ = stats.pearsonr(dfg['PRICEEACH'], dfg['QUANTITYORDERED'])
                        row['pearson_r'] = round(pr, 4)
                        X = dfg[['PRICEEACH']]
                        y = dfg['QUANTITYORDERED']
                        model = LinearRegression().fit(X, y)
                        row['coef'] = round(model.coef_[0], 4)
                        row['rmse'] = round(compute_rmse(y, model.predict(X)), 2)
                    except:
                        row.update({'pearson_r': None, 'coef': None, 'rmse': None})
                    results.append(row)
                st.dataframe(pd.DataFrame(results))
