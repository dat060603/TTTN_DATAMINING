import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
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
        with open(file_name, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    local_css("style.css")

    # ======= Load data =======
    df = load_data()
    # ======= Filters =======
    with st.expander("🔎 Bộ lọc (Data Reason)", expanded=True):
        c1, c2, c3 = st.columns(3)
        # --- Bộ lọc năm ---
        with c1:
            years = sorted(df['YEAR_ID'].dropna().unique()) if 'YEAR_ID' in df.columns else []
            year_options = ["Tất cả các năm"] + [str(int(y)) for y in years]  # Thêm "Tất cả các năm"
            selected_year = st.selectbox("Chọn năm", options=year_options, index=0 if years else 0)
        # --- Bộ lọc quốc gia ---
        with c2:
            countries = sorted(df['COUNTRY'].dropna().unique()) if 'COUNTRY' in df.columns else []
            selected_country = st.multiselect("Chọn quốc gia", options=countries,
                                              default=countries if countries else [])
        # --- Bộ lọc dòng sản phẩm ---
        with c3:
            product_lines = sorted(df['PRODUCTLINE'].dropna().unique()) if 'PRODUCTLINE' in df.columns else []
            selected_productline = st.multiselect("Chọn PRODUCTLINE", options=product_lines,
                                                  default=product_lines if product_lines else [])
    # ======= Apply filters =======
    _data = df.copy()
    # Lọc theo năm nếu không chọn "Tất cả các năm"
    if selected_year != "Tất cả các năm":
        _data = _data[_data['YEAR_ID'] == int(selected_year)]
    # Lọc quốc gia và product line
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
        st.markdown(
            """
            <div class="recommendation-box">
                <h3> Giải thích:</h3>
                <ul>
                    <li>Ma trận tương quan cho biết <b>mức độ liên quan tuyến tính giữa các biến</b>.</li>
                    <li>Ví dụ: Nếu <code>PRICEEACH</code> và <code>SALES</code> có hệ số tương quan cao (gần 1 hoặc -1), tức là khi giá thay đổi thì doanh thu cũng thay đổi theo hướng tương ứng.</li>
                    <li>Màu đỏ biểu thị mối tương quan âm, màu xanh dương biểu thị mối tương quan dương.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ===============================
    # Tab 2: Scatter + Trendline OLS
    # ===============================
    with tabs[1]:
        st.subheader("2️⃣ Scatter SALES vs QUANTITYORDERED")
        if 'SALES' in _data.columns and 'QUANTITYORDERED' in _data.columns:
            st.markdown("**Tùy chọn hiển thị**")
            col1, col2 = st.columns([2, 1])
            with col1:
                show_trend = st.checkbox("Hiển thị trendline OLS", value=True)
                use_log = st.checkbox("Log-log scale", help="Dùng log(SALES), log(QUANTITYORDERED)")
            with col2:
                sample_slider = st.slider("Lọc số dòng hiển thị", min_value=500, max_value=20000, value=5000, step=500)

            plot_df = _data.dropna(subset=['SALES', 'QUANTITYORDERED'])
            if use_log:
                plot_df = plot_df[(plot_df['SALES'] > 0) & (plot_df['QUANTITYORDERED'] > 0)]

            if len(plot_df) > sample_slider:
                plot_df = plot_df.sample(sample_slider, random_state=42)

            facet_col = 'PRODUCTLINE' if plot_df['PRODUCTLINE'].nunique() <= 4 else None

            fig = px.scatter(
                plot_df,
                x=np.log(plot_df['SALES']) if use_log else plot_df['SALES'],
                y=np.log(plot_df['QUANTITYORDERED']) if use_log else plot_df['QUANTITYORDERED'],
                color='PRODUCTLINE' if 'PRODUCTLINE' in plot_df.columns else None,
                facet_col=facet_col,
                labels={'x': 'log(SALES)' if use_log else 'SALES',
                        'y': 'log(QUANTITYORDERED)' if use_log else 'QUANTITYORDERED'}
            )
            st.plotly_chart(fig, use_container_width=True)

            if show_trend:
                try:
                    X = np.log(plot_df['SALES']).values.reshape(-1, 1) if use_log else plot_df[['SALES']].values
                    y = np.log(plot_df['QUANTITYORDERED']).values if use_log else plot_df['QUANTITYORDERED'].values
                    model = LinearRegression().fit(X, y)
                    st.markdown(
                        f"**OLS Trendline:** coef = {model.coef_[0]:.4f}, intercept = {model.intercept_:.4f}, R² = {r2_score(y, model.predict(X)):.4f}")
                except Exception as e:
                    st.warning(f"Lỗi khi fit trendline: {e}")
        else:
            st.warning("Thiếu SALES hoặc QUANTITYORDERED.")

        # st.markdown("""
        # **🧠 Giải thích:**
        # - Biểu đồ phân tán (scatter plot) giúp nhận diện **mối quan hệ giữa doanh thu (SALES) và số lượng bán (QUANTITYORDERED)**.
        # - Đường trendline (OLS) cho biết xu hướng tổng thể: Nếu dốc lên → số lượng bán tăng thì doanh thu tăng.
        # - Dùng log scale để dễ quan sát khi dữ liệu có độ chênh lệch lớn.
        # """)
        st.markdown(
            """
            <div class="recommendation-box">
                <h3> Giải thích:</h3>
                <ul>
                    <li>Biểu đồ phân tán (scatter plot) giúp nhận diện <b>mối quan hệ giữa doanh thu (SALES) và số lượng bán (QUANTITYORDERED)</b>.</li>
                    <li>Đường trendline (OLS) cho biết xu hướng tổng thể: <b>Nếu dốc lên → số lượng bán tăng thì doanh thu tăng</b>.</li>
                    <li>Dùng log scale để dễ quan sát khi dữ liệu có độ chênh lệch lớn.</b>.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.subheader("2️⃣ Scatter SALES vs PRICEEACH")

        if 'SALES' in _data.columns and 'PRICEEACH' in _data.columns:
            col1, col2 = st.columns([2, 1])
            with col1:
                show_trend2 = st.checkbox("Hiển thị trendline OLS", value=True, key="trend_price_sales")
                use_log2 = st.checkbox("Log-log scale", help="Dùng log(SALES), log(PRICEEACH)", key="log_price_sales")
            with col2:
                sample_slider2 = st.slider("Lọc số dòng hiển thị", min_value=500, max_value=20000, value=5000, step=500,
                                           key="slider_price_sales")

            df_plot2 = _data.dropna(subset=['SALES', 'PRICEEACH'])
            if use_log2:
                df_plot2 = df_plot2[(df_plot2['SALES'] > 0) & (df_plot2['PRICEEACH'] > 0)]

            if len(df_plot2) > sample_slider2:
                df_plot2 = df_plot2.sample(sample_slider2, random_state=42)

            fig2 = px.scatter(
                df_plot2,
                x=np.log(df_plot2['PRICEEACH']) if use_log2 else df_plot2['PRICEEACH'],
                y=np.log(df_plot2['SALES']) if use_log2 else df_plot2['SALES'],
                color='PRODUCTLINE' if 'PRODUCTLINE' in df_plot2.columns else None,
                labels={'x': 'log(PRICEEACH)' if use_log2 else 'PRICEEACH',
                        'y': 'log(SALES)' if use_log2 else 'SALES'}
            )
            st.plotly_chart(fig2, use_container_width=True)

            if show_trend2:
                X = np.log(df_plot2['PRICEEACH']).values.reshape(-1, 1) if use_log2 else df_plot2[['PRICEEACH']].values
                y = np.log(df_plot2['SALES']).values if use_log2 else df_plot2['SALES'].values
                model2 = LinearRegression().fit(X, y)
                st.markdown(
                    f"**OLS Trendline:** coef = {model2.coef_[0]:.4f}, intercept = {model2.intercept_:.4f}, R² = {r2_score(y, model2.predict(X)):.4f}")

        st.markdown(
            """
            <div class="recommendation-box">
                <h3> Giải thích:</h3>
                <ul>
                    <li>Biểu đồ này giúp kiểm tra <b>giá cao có giúp tăng doanh thu không</b>.</li>
                    <li>Nếu trendline lên → <b>các sản phẩm giá cao vẫn bán chạy → có thể tăng giá dòng cao cấp</b>.</li>                
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.subheader("3️⃣ Hồi quy đa biến: Dự đoán SALES từ PRICEEACH và QUANTITYORDERED")

        # Chuẩn bị dữ liệu
        plot_df = _data[['SALES', 'PRICEEACH', 'QUANTITYORDERED']].dropna()
        plot_df = plot_df[(plot_df['SALES'] > 0) & (plot_df['PRICEEACH'] > 0) & (plot_df['QUANTITYORDERED'] > 0)]

        # Log-transform
        plot_df['SALES_LOG'] = np.log(plot_df['SALES'])
        plot_df['PRICEEACH_LOG'] = np.log(plot_df['PRICEEACH'])
        plot_df['QUANTITYORDERED_LOG'] = np.log(plot_df['QUANTITYORDERED'])

        # Hiển thị dữ liệu đã biến đổi log
        with st.expander("📋 Xem dữ liệu đã log-transform"):
            st.dataframe(plot_df[['PRICEEACH', 'QUANTITYORDERED', 'SALES',
                                  'PRICEEACH_LOG', 'QUANTITYORDERED_LOG', 'SALES_LOG']].head(10))

        # Huấn luyện mô hình hồi quy
        X = plot_df[['PRICEEACH_LOG', 'QUANTITYORDERED_LOG']]
        y = plot_df['SALES_LOG']

        model = LinearRegression()
        model.fit(X, y)

        # Dự đoán và đánh giá
        y_pred = model.predict(X)
        r2 = model.score(X, y)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        #
        st.subheader("🌐 Biểu đồ 3D: Quan hệ giữa Giá, Số lượng và Doanh thu")
        fig3d = go.Figure(data=[go.Scatter3d(
            x=plot_df['PRICEEACH'],
            y=plot_df['QUANTITYORDERED'],
            z=plot_df['SALES'],
            mode='markers',
            marker=dict(
                size=4,
                color=plot_df['SALES'],  # màu theo giá trị SALES
                colorscale='Viridis',
                opacity=0.7
            )
        )])
        fig3d.update_layout(
            scene=dict(
                xaxis_title='Giá bán (PRICEEACH)',
                yaxis_title='Số lượng (QUANTITYORDERED)',
                zaxis_title='Doanh thu (SALES)'
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        st.plotly_chart(fig3d, use_container_width=True)
        # Hiển thị hệ số
        st.markdown(f"""
        ✅ **Hệ số hồi quy:**
        - PRICEEACH_LOG: **{model.coef_[0]:.4f}**
        - QUANTITYORDERED_LOG: **{model.coef_[1]:.4f}**
        - Intercept: **{model.intercept_:.4f}**

        📊 **Đánh giá độ phù hợp mô hình:**
        - R²: **{r2:.4f}**
        - MAE: **{mae:.2f}**
        - RMSE: **{rmse:.2f}**
        """)

        fig = px.scatter(
            x=y,
            y=y_pred,
            labels={'x': 'SALES_LOG thực tế', 'y': 'SALES_LOG dự đoán'},
            title='📊 So sánh SALES_LOG: Thực tế vs Dự đoán',
            trendline='ols'
        )
        st.plotly_chart(fig, use_container_width=True)

        # 🎯 Recommendation
        st.subheader("🤖 Recommendation từ mô hình")
        rec_lines = []

        if model.coef_[0] > model.coef_[1]:
            rec_lines.append(
                "📌 Doanh thu nhạy cảm hơn với **giá bán** — bạn có thể thử tăng giá nhẹ để cải thiện doanh thu.")
        else:
            rec_lines.append(
                "📌 Doanh thu phụ thuộc nhiều vào **số lượng bán** — cần tập trung tăng sản lượng hoặc marketing.")

        if r2 > 0.85:
            rec_lines.append(
                "✅ Mô hình có **độ giải thích cao** (R² > 0.85) — có thể tin cậy để làm công cụ dự báo nhanh.")

        for rec in rec_lines:
            st.info(rec)



    # =========================
    # Tab 3: SALES_DIFF phân tích
    # =========================
    with tabs[2]:
        st.subheader("3️⃣ Phân tích chênh lệch SALES vs ORDER_VALUE")
        if 'SALES' in _data.columns and all(col in _data.columns for col in ['QUANTITYORDERED', 'PRICEEACH']):
            # Tính toán tổng giá trị đơn hàng dựa trên số lượng và đơn giá
            _data['TOTAL_ORDER_VALUE'] = _data['QUANTITYORDERED'] * _data['PRICEEACH']
            _data['SALES_DIFF'] = _data['SALES'] - _data['TOTAL_ORDER_VALUE']

            # Lọc dữ liệu hợp lệ: SALES và ORDER_VALUE dương và khác NaN
            valid_data = _data[(_data['SALES'] > 0) & (_data['TOTAL_ORDER_VALUE'] > 0)].copy()
            valid_data = valid_data[valid_data['SALES_DIFF'].notna()]

            # Loại bỏ ngoại lệ nếu cần
            z_scores = np.abs(stats.zscore(valid_data['SALES_DIFF']))
            valid_data['Z_SCORE'] = z_scores
            cleaned_data = valid_data[z_scores < 3]  # Loại bỏ các điểm có chênh lệch quá bất thường

            # Hiển thị thống kê mô tả
            st.markdown("**📉 Mô tả chênh lệch doanh thu:**")
            desc = cleaned_data['SALES_DIFF'].describe().apply(lambda x: f"{x:,.2f}")
            st.dataframe(desc.rename("Giá trị").to_frame())

            # Ngưỡng để lọc chênh lệch đáng kể
            thresh = st.number_input("Ngưỡng chênh lệch đáng kể", value=100.0, step=10.0)
            df_diff = cleaned_data[cleaned_data['SALES_DIFF'].abs() > thresh]

            st.markdown(f"Số dòng có chênh lệch lớn hơn {thresh}: **{len(df_diff):,} dòng**")
            st.dataframe(df_diff[['ORDERNUMBER', 'PRODUCTCODE', 'SALES', 'TOTAL_ORDER_VALUE', 'SALES_DIFF']].head(200))

            st.download_button("📥 Tải CSV", data=df_diff.to_csv(index=False).encode('utf-8'),
                               file_name="sales_diff_filtered.csv")

            # Gợi ý phân tích
            st.markdown(
                "💡 **Ý nghĩa:** Những chênh lệch này có thể đến từ chiết khấu, sai lệch nhập liệu hoặc điều chỉnh thủ công. Quản lý nên kiểm tra các đơn có chênh lệch lớn để đảm bảo dữ liệu chính xác.")
        else:
            st.warning("Không đủ cột để tính SALES_DIFF.")
        # st.markdown("""
        # **🧠 Giải thích:**
        # - Chênh lệch giữa `SALES` (doanh thu thực tế) và `QUANTITYORDERED * PRICEEACH` giúp **phát hiện bất thường trong đơn hàng**, ví dụ:
        #   - Có chiết khấu ẩn.
        #   - Giá khuyến mãi không đồng nhất.
        #   - Lỗi nhập dữ liệu hoặc điều chỉnh giá.
        # - Những dòng có `SALES_DIFF` lớn cần được kiểm tra kỹ.
        # """)
        st.markdown(
            """
            <div class="recommendation-box">
                <h3> Giải thích:</h3>
                <ul>
                    <li>Chênh lệch giữa `SALES` (doanh thu thực tế) và `QUANTITYORDERED * PRICEEACH` giúp <b>phát hiện bất thường trong đơn hàng</b>.</li>
                    <li>Có chiết khấu ẩn</li> 
                     <li>Giá khuyến mãi không đồng nhất</li> 
                    <li>Lỗi nhập dữ liệu hoặc điều chỉnh giá</li>  
                     <li>Những dòng có `SALES_DIFF` lớn cần được kiểm tra kỹ    </li>       
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ===============================
    # Tab 4: Hồi quy đơn giản
    # ===============================
    with tabs[3]:
        def compute_rmse(y_true, y_pred):
            return np.sqrt(np.mean((y_true - y_pred) ** 2))

        st.subheader("4️⃣ Mô hình hồi quy nâng cao và đánh giá")

        # Lựa chọn mô hình
        reg_choice = st.radio("Chọn mô hình hồi quy:", options=[
            "QUANTITYORDERED ~ PRICEEACH",
            "SALES ~ QUANTITYORDERED + PRICEEACH",
            "SALES ~ QUANTITYORDERED + PRICEEACH + PRODUCTLINE (dummy variables)"
        ])

        # Xác định công thức và các cột cần
        if reg_choice == "QUANTITYORDERED ~ PRICEEACH":
            req_cols = ['PRICEEACH', 'QUANTITYORDERED']
            formula = "QUANTITYORDERED ~ PRICEEACH"
        elif reg_choice == "SALES ~ QUANTITYORDERED + PRICEEACH":
            req_cols = ['SALES', 'QUANTITYORDERED', 'PRICEEACH']
            formula = "SALES ~ QUANTITYORDERED + PRICEEACH"
        else:
            req_cols = ['SALES', 'QUANTITYORDERED', 'PRICEEACH', 'PRODUCTLINE']
            formula = "SALES ~ QUANTITYORDERED + PRICEEACH + C(PRODUCTLINE)"

        # Kiểm tra dữ liệu đầy đủ
        if all(col in _data.columns for col in req_cols):
            reg_df = _data[req_cols].dropna()
            if len(reg_df) >= 10:
                try:
                    # Huấn luyện mô hình
                    model = smf.ols(formula, data=reg_df).fit()

                    # Dự đoán và đánh giá
                    target_var = formula.split("~")[0].strip()
                    preds = model.predict(reg_df)

                    if target_var in reg_df.columns:
                        y_true = reg_df[target_var]
                        mae = mean_absolute_error(y_true, preds)
                        rmse = compute_rmse(y_true, preds)
                        r2 = model.rsquared

                        # Tóm tắt mô hình
                        st.markdown("### 📈 Kết quả mô hình")
                        st.markdown(f"**R² (Độ phù hợp):** {r2:.4f}  ")
                        st.markdown(f"**MAE (Sai số tuyệt đối TB):** {mae:.2f}  ")
                        st.markdown(f"**RMSE (Căn sai số bình phương):** {rmse:.2f}  ")

                        # Diễn giải hệ số
                        # Khuyến nghị tự động từ hệ số hồi quy
                        st.markdown("### ✅ Khuyến nghị từ mô hình:")
                        recommendations = []
                        for var, coef in model.params.items():
                            if var == 'Intercept':
                                continue
                            abs_coef = abs(coef)
                            # Xử lý tên biến nếu là biến phân loại (dummy)
                            if "C(PRODUCTLINE)" in var:
                                product = var.replace("C(PRODUCTLINE)[T.", "").replace("]", "")
                                if coef < 0:
                                    recommendations.append(
                                        f"• Doanh thu từ dòng sản phẩm **{product}** thấp hơn so với dòng chuẩn. "
                                        f"👉 Xem xét **khuyến mãi, cải thiện sản phẩm hoặc chiến dịch marketing** cho dòng này.")
                                else:
                                    recommendations.append(f"• Dòng sản phẩm **{product}** mang lại doanh thu cao hơn. "
                                                           f"👉 Có thể **tăng đầu tư hoặc tập trung bán hàng** cho dòng này.")
                            elif var == "PRICEEACH":
                                if coef < 0:
                                    recommendations.append(
                                        "• **Tăng giá bán có xu hướng làm giảm doanh thu/số lượng**. "
                                        "👉 Cân nhắc giữ giá ổn định hoặc dùng chiến lược giá hợp lý.")
                                else:
                                    recommendations.append("• **Tăng giá có thể vẫn giúp tăng doanh thu**. "
                                                           "👉 Có thể xem xét tăng giá ở mức hợp lý.")
                            elif var == "QUANTITYORDERED":
                                if coef > 0:
                                    recommendations.append("• **Số lượng đặt hàng ảnh hưởng tích cực đến doanh thu**. "
                                                           "👉 Ưu tiên thúc đẩy bán gói combo/số lượng lớn.")
                                else:
                                    recommendations.append(
                                        "• **Số lượng đặt hàng tăng không làm tăng doanh thu tương ứng**. "
                                        "👉 Xem xét lại chiến lược giá theo số lượng.")

                        # Hiển thị khuyến nghị
                        if recommendations:
                            for rec in recommendations:
                                st.markdown(rec)
                        else:
                            st.info("Không tìm thấy khuyến nghị rõ ràng từ hệ số mô hình.")

                        # Biểu đồ residuals
                        st.subheader("📊 Biểu đồ sai số (Residual Plot)")
                        residuals = y_true - preds
                        fig_residual = px.scatter(
                            x=preds,
                            y=residuals,
                            labels={"x": "Giá trị dự đoán", "y": "Sai số (Residuals)"},
                            title="Residual Plot: Giá trị dự đoán vs Sai số"
                        )
                        fig_residual.add_hline(y=0, line_dash="dash", line_color="red")
                        st.plotly_chart(fig_residual, use_container_width=True)

                        # Tùy chọn xem chi tiết
                        with st.expander("📄 Xem bảng thống kê mô hình chi tiết (dành cho phân tích chuyên sâu)"):
                            st.text(model.summary().as_text())

                    else:
                        st.warning(f"Không tìm thấy biến mục tiêu '{target_var}' trong dữ liệu.")
                except Exception as e:
                    st.error(f"❌ Lỗi khi chạy mô hình hồi quy: {e}")
            else:
                st.warning("Không đủ dữ liệu để chạy mô hình (cần ít nhất 10 dòng dữ liệu).")
        else:
            st.warning("Thiếu dữ liệu cần thiết để huấn luyện mô hình.")
        st.markdown(
            """
            <div class="recommendation-box">
                <h3>Giải thích:</h3>
                <ol>
                    <li>
                        Mô hình hồi quy giúp đánh giá <b>ảnh hưởng của các biến đầu vào (giá, số lượng, loại sản phẩm)</b> lên <b>doanh thu hoặc số lượng bán</b>.
                    </li>
                    <li>
                        Hệ số hồi quy cho biết: Khi một biến tăng 1 đơn vị, biến mục tiêu tăng/giảm bao nhiêu (giữ các yếu tố khác không đổi).
                    </li>
                    <li>
                        Residual plot giúp kiểm tra <b>sai số mô hình</b> — nếu phân tán đều quanh 0 thì mô hình tốt.
                    </li>
                </ol>
                
            <h3>Mô hình 3: SALES ~ QUANTITYORDERED + PRICEEACH + PRODUCTLINE</h3>
            <ul>
                <li>Giống mô hình 2 nhưng thêm biến phân loại <code>PRODUCTLINE</code> (mã hóa dạng dummy).</li>
                <li>
                    Hệ số các dòng sản phẩm đều âm, ví dụ:
                    <div style="margin-left:20px;">
                        <ul>
                            <li><code>PRODUCTLINE = Ships</code>: -752 → doanh thu trung bình thấp hơn dòng gốc ~752 USD.</li>
                        </ul>
                    </div>
                </li>
                <li>
                    Sai số giảm:
                    <div style="margin-left:20px;">
                        <ul>
                            <li>MAE giảm từ 689 → 659</li>
                            <li>RMSE giảm từ 951 → 923</li>
                        </ul>
                    </div>
                </li>
                <li>Hiểu rằng doanh thu không chỉ phụ thuộc vào số lượng và giá, mà còn theo từng dòng sản phẩm → mô hình tốt hơn về giải thích & độ chính xác.</li>
                <li>Mô hình tốt nhất để sử dụng nếu muốn ra quyết định dựa trên phân tích dòng sản phẩm.</li>
            </ul>
                </div>
            """,
            unsafe_allow_html=True
        )

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
                # 📊 Scatter chart
                st.plotly_chart(
                    px.scatter(sub, x='PRICEEACH', y='QUANTITYORDERED', color=group_by),
                    use_container_width=True,
                    key=f"scatter_tab5_{group_by}"
                )

                # 📉 Hồi quy theo nhóm
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

                result_df = pd.DataFrame(results)
                st.dataframe(result_df)

                # 🧠 Giải thích chỉ số
                st.markdown("""
                **🧠 Giải thích:**
                - Phân tích theo nhóm (PRODUCTLINE hoặc DEALSIZE) giúp so sánh **hiệu quả giữa các dòng sản phẩm hoặc quy mô giao dịch**.
                - **Pearson r**: hệ số tương quan (gần 1 hoặc -1 là tương quan mạnh).
                - **Coef**: độ nhạy của số lượng bán khi giá thay đổi.
                - **RMSE**: sai số dự đoán.
                """)

                # 📌 Recommendation tự động
                st.subheader("📌 Recommendation ")

                for _, row in result_df.iterrows():
                    group = row['group']
                    coef = row['coef']
                    r = row['pearson_r']
                    rmse = row['rmse']

                    if coef is None or r is None:
                        st.info(f"ℹ️ **{group}**: Không đủ dữ liệu để đưa ra nhận định.")
                        continue

                    if coef < -0.03 and abs(r) > 0.05:
                        st.warning(
                            f"❗ **{group}**: Giá tăng làm giảm số lượng bán đáng kể (coef = {coef:.4f}, r = {r:.2f}). Nên xem lại chiến lược giá.")
                    elif coef > 0.03 and r > 0.05:
                        st.success(
                            f"✅ **{group}**: Giá tăng đi kèm tăng số lượng bán (coef = {coef:.4f}, r = {r:.2f}) — tiềm năng mở rộng thị trường.")
                    elif abs(coef) < 0.01 and abs(r) < 0.05:
                        st.info(
                            f"ℹ️ **{group}**: Không có mối liên hệ rõ ràng giữa giá và số lượng — có thể không bị ảnh hưởng bởi giá.")
                    else:
                        st.info(
                            f"📌 **{group}**: Tác động giá ở mức trung bình (coef = {coef:.4f}, r = {r:.2f}) — cân nhắc tùy mục tiêu.")
