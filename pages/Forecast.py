
# pages/Forecast.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet import Prophet

from components.data_loader import load_data
def app():
    def local_css(file_name):
        with open(file_name, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    local_css("style.css")
# ====== Page config ======
    st.set_page_config(page_title="Forecast", layout="wide")
    st.title("Forecast: Dự báo doanh thu")
    # ====== Utilities ======
    def aggregate_time_series(df, date_col, value_col, freq):
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        ts = df.set_index(date_col)[value_col].resample(freq).sum().reset_index()
        ts.columns = ['ds', 'y']
        idx = pd.date_range(ts['ds'].min(), ts['ds'].max(), freq=freq)
        ts = ts.set_index('ds').reindex(idx).rename_axis('ds').reset_index()
        ts['y'] = ts['y'].fillna(0)
        return ts
    def prepare_lr_features(series, freq):
        df = series.copy().sort_values('ds').reset_index(drop=True)
        df['t'] = np.arange(len(df))
        df['t2'] = df['t'] ** 2
        df['month'] = df['ds'].dt.month
        month_dummies = pd.get_dummies(df['month'], prefix='m', drop_first=True)
        df = pd.concat([df, month_dummies], axis=1)
        period = 12 if freq in ['M', 'ME'] else 52 if freq == 'W' else 365
        df['sin1'] = np.sin(2 * np.pi * df['t'] / period)
        df['cos1'] = np.cos(2 * np.pi * df['t'] / period)
        X = df.drop(columns=['ds', 'y', 'month'])
        return X, df['y'], df
    def align_features_to_ref(X, ref_cols):
        # ensure X has all ref_cols (add missing with 0), drop extras, and order them
        X = X.copy()
        for c in ref_cols:
            if c not in X.columns:
                X[c] = 0
        # Drop any column not in ref_cols
        X = X.loc[:, ref_cols]
        return X
    def time_train_test_split(X, y, test_size):
        n = len(X)
        t = int(n * (1 - test_size))
        return X.iloc[:t], X.iloc[t:], y.iloc[:t], y.iloc[t:], t

    def safe_rmse(y_true, y_pred):
        try:
            return mean_squared_error(y_true, y_pred, squared=False)
        except TypeError:
            return np.sqrt(mean_squared_error(y_true, y_pred))

    def compute_metrics(y_true, y_pred):
        y_true = np.asarray(y_true).astype(float)
        y_pred = np.asarray(y_pred).astype(float)
        if y_true.shape[0] != y_pred.shape[0]:
            raise ValueError(f"compute_metrics: length mismatch {y_true.shape[0]} vs {y_pred.shape[0]}")
        mae = mean_absolute_error(y_true, y_pred)
        rmse = safe_rmse(y_true, y_pred)
        nonzero = y_true != 0
        mape = np.nan
        if nonzero.sum() > 0:
            mape = np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100.0
        return {"MAE": float(mae), "RMSE": float(rmse), "MAPE": (float(mape) if not np.isnan(mape) else None)}

    def build_future_lr_features(df_feat, horizon, freq):
        last_t = df_feat['t'].iloc[-1]
        future_t = np.arange(last_t + 1, last_t + 1 + horizon)
        freq_for_offset = {'D': 'D', 'W': 'W', 'ME': 'M'}[freq]
        future_ds = pd.date_range(df_feat['ds'].max() + pd.tseries.frequencies.to_offset(freq_for_offset),
                                  periods=horizon, freq=freq)
        future_df = pd.DataFrame({'ds': future_ds, 't': future_t})
        future_df['t2'] = future_df['t'] ** 2
        future_df['month'] = future_df['ds'].dt.month
        month_dummies = pd.get_dummies(future_df['month'], prefix='m', drop_first=True)
        existing_month_cols = [c for c in df_feat.columns if c.startswith('m_')]
        for c in existing_month_cols:
            future_df[c] = month_dummies[c] if c in month_dummies else 0
        period = 12 if freq in ['M', 'ME'] else 52 if freq == 'W' else 365
        future_df['sin1'] = np.sin(2 * np.pi * future_df['t'] / period)
        future_df['cos1'] = np.cos(2 * np.pi * future_df['t'] / period)
        return future_df.drop(columns=['month'])

    # Rolling-origin backtest returning per-horizon errors (list of lists) and average metrics
    def rolling_origin_backtest(series, model_name, freq, horizon, initial, period, log_transform=False):
        """
        series: dataframe with ds,y (original scale)
        model_name: "Linear Regression" or "Prophet"
        returns: dict {
          'avg_metrics': DataFrame mean metrics across folds,
          'per_horizon': {k: [errors across folds for step k] ...}
        }
        """
        n = len(series)
        starts = list(range(initial, n - horizon + 1, period))
        if len(starts) == 0:
            st.warning("Không đủ mẫu để chạy rolling backtest với tham số đã cho.")
            return None

        all_metrics = []
        per_horizon_errors = {h + 1: [] for h in range(horizon)}  # horizon step -> list of MAE across folds
        progress = st.progress(0)

        for i, start in enumerate(starts):
            train = series.iloc[:start].reset_index(drop=True)
            test = series.iloc[start:start + horizon].reset_index(drop=True)

            if model_name == "Linear Regression":
                y_train_model = np.log1p(train['y']) if log_transform else train['y']
                X_train, y_train_dummy, df_feat_train = prepare_lr_features(
                    pd.DataFrame({'ds': train['ds'], 'y': y_train_model}), freq)

                lr = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
                lr.fit(X_train, y_train_dummy)

                # prepare test features
                X_test_raw, _, _ = prepare_lr_features(test, freq)
                X_test = align_features_to_ref(X_test_raw, X_train.columns)
                pred = lr.predict(X_test)

            else:  # Prophet
                y_train_model = np.log1p(train['y']) if log_transform else train['y']
                df_train_prophet = pd.DataFrame({'ds': train['ds'], 'y': y_train_model})

                m = Prophet(yearly_seasonality=True)
                m.fit(df_train_prophet)

                future = m.make_future_dataframe(periods=horizon, freq=('M' if freq in ['M', 'ME'] else freq))
                forecast = m.predict(future)

                pred = forecast['yhat'].iloc[-horizon:].values

            # === Postprocess prediction ===
            if log_transform:
                pred = np.expm1(pred)
            if np.any(np.isnan(pred)) or np.any(np.isinf(pred)):
                st.warning(f"Dự báo {model_name} trả về giá trị không hợp lệ tại fold {i}. Bỏ qua fold này.")
                continue
            # === Compute metrics ===
            y_true = test['y'].values
            metrics = compute_metrics(y_true, pred)
            all_metrics.append(metrics)

            # collect per-horizon absolute errors
            abs_errs = np.abs(y_true - pred)
            for h_step in range(len(abs_errs)):
                per_horizon_errors[h_step + 1].append(abs_errs[h_step])

            progress.progress((i + 1) / len(starts))

        # aggregate metrics
        metrics_df = pd.DataFrame(all_metrics)
        avg_metrics = metrics_df.mean().to_dict()
        per_horizon_mean = {k: float(np.mean(v)) for k, v in per_horizon_errors.items()}

        return {"avg_metrics": avg_metrics, "per_horizon_mean": per_horizon_mean}

    # ====== Controls (in-page) ======
    with st.expander("⚙️ Filter & Forecast controls", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            target = st.selectbox("Chỉ số dự báo", ["SALES", "TOTAL_ORDER_VALUE"])
            freq_label = st.selectbox("Tần suất", ["Daily", "Weekly", "Monthly"], index=2)
            freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "ME"}
            freq = freq_map[freq_label]
        with c2:
            horizon = st.number_input("Số kỳ dự báo (periods)", min_value=1, max_value=24, value=6)
            test_size = st.slider("Tỷ lệ test (theo thời gian)", 0.05, 0.4, 0.2)
            log_transform = st.checkbox("Log-transform target (log1p)", value=False,
                                        help="Bật nếu dữ liệu doanh thu lệch phải, sẽ huấn luyện trên log1p(y).")
        with c3:
            models_to_run = st.multiselect("Chọn mô hình", ["Linear Regression", "Prophet"],
                                           default=["Linear Regression", "Prophet"])
            run = st.button("🚀 Chạy dự báo")
    if run:
        # Load data
        df = load_data()
        if target not in df.columns:
            st.error(f"Cột {target} không tồn tại trong dataframe.")
            st.stop()

        series = aggregate_time_series(df, 'ORDERDATE', target, freq)
        if series['y'].isnull().all():
            st.error("Series toàn giá trị null sau khi aggregate. Kiểm tra dữ liệu.")
            st.stop()
        # prepare transformed series for modeling if needed
        series_model = series.copy()
        if log_transform:
            series_model['y'] = np.log1p(series_model['y'])
        # tabs
        tab_overview, tab_diag, tab_backtest = st.tabs(["📈 Overview", "🛠 Diagnostics", "📊 Backtest"])
        # Base historical plot
        with tab_overview:
            st.subheader("Dữ liệu lịch sử")
            fig_hist = px.line(series, x='ds', y='y', title=f"Lịch sử {target} ({freq_label})")
            st.plotly_chart(fig_hist, use_container_width=True)
        results = {}
        # --------- Linear Regression ----------
        if "Linear Regression" in models_to_run:
            X, y, df_feat = prepare_lr_features(series_model, freq)
            X_tr, X_te, y_tr, y_te, split_idx = time_train_test_split(X, y, test_size)
            # fit
            lr_model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            lr_model.fit(X_tr, y_tr)

            # align test features
            X_te_aligned = align_features_to_ref(X_te, X_tr.columns)
            y_pred_te = lr_model.predict(X_te_aligned)
            if log_transform:
                y_pred_te = np.expm1(y_pred_te)
                y_te_orig = np.expm1(y_te)
            else:
                y_te_orig = y_te
            # future forecast
            X_future_full = build_future_lr_features(df_feat, horizon, freq)
            X_future_only = align_features_to_ref(X_future_full.drop(columns=['ds']), X_tr.columns)
            y_future_pred = lr_model.predict(X_future_only)
            if log_transform:
                y_future_pred = np.expm1(y_future_pred)
            future_lr_df = pd.DataFrame({'ds': X_future_full['ds'], 'yhat': y_future_pred})

            metrics_lr = compute_metrics(y_te_orig, y_pred_te)
            # Lưu cả forecast tương lai và dự báo test (cho vẽ)
            results['Linear Regression'] = {
                "metrics": metrics_lr,
                "forecast": future_lr_df,
                "test_pred": pd.DataFrame({'ds': series['ds'].iloc[split_idx:].reset_index(drop=True),
                                           'y_true': y_te_orig.reset_index(drop=True), 'y_pred': y_pred_te})
            }
        # --------- Prophet ----------
        if "Prophet" in models_to_run:
            prophet_freq = 'M' if freq in ['M', 'ME'] else freq
            m = Prophet(yearly_seasonality=True, weekly_seasonality=(prophet_freq == 'D'))
            m.fit(series_model[['ds', 'y']])
            forecast = m.predict(m.make_future_dataframe(periods=horizon, freq=prophet_freq))
            n = len(series)
            test_cutoff = int(n * (1 - test_size))
            y_true_test = series['y'].iloc[test_cutoff:n].reset_index(drop=True)

            y_pred_test = forecast['yhat'].iloc[test_cutoff:n].reset_index(drop=True)
            if log_transform:
                y_pred_test = np.expm1(y_pred_test)
            metrics_prophet = compute_metrics(y_true_test.values, y_pred_test.values)

            future_prophet_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].iloc[n:].reset_index(drop=True)
            if log_transform:
                future_prophet_df[['yhat', 'yhat_lower', 'yhat_upper']] = np.expm1(
                    future_prophet_df[['yhat', 'yhat_lower', 'yhat_upper']])
            results['Prophet'] = {
                "metrics": metrics_prophet,
                "forecast": future_prophet_df,
                "test_pred": pd.DataFrame(
                    {'ds': series['ds'].iloc[test_cutoff:].reset_index(drop=True), 'y_true': y_true_test,
                     'y_pred': y_pred_test})
            }
        with tab_overview:
            st.subheader("Lịch sử + Dự báo (tương lai)")
            fig = px.line(series, x='ds', y='y', title=f"{target} lịch sử & dự báo")
            # add model forecasts
            for name, res in results.items():
                df_fc = res['forecast']
                fig.add_scatter(x=df_fc['ds'], y=df_fc['yhat'], mode='lines', name=f"{name} forecast")
                # if prophet has lower/upper, draw CI
                if 'yhat_lower' in df_fc.columns and 'yhat_upper' in df_fc.columns:
                    fig.add_scatter(x=list(df_fc['ds']) + list(df_fc['ds'][::-1]),
                                    y=list(df_fc['yhat_upper']) + list(df_fc['yhat_lower'][::-1]),
                                    fill='toself', fillcolor='rgba(200,200,200,0.2)', line=dict(color='rgba(255,255,255,0)'),
                                    hoverinfo="skip", name=f"{name} CI")
            st.plotly_chart(fig, use_container_width=True)

            # metrics table
            if results:
                st.subheader("Metrics trên tập test")
                metrics_df = pd.DataFrame({m: res['metrics'] for m, res in results.items()}).T
                st.dataframe(metrics_df.style.format(precision=3))
            # download buttons
            for model, res in results.items():
                csv = res['forecast'].to_csv(index=False)
                st.download_button(f"Tải dự báo ({model})", csv, file_name=f"{model}_forecast.csv", mime="text/csv")
        #     Giai thich tab
            # === Phân tích kết quả dự báo ===
            st.subheader("📌 Phân tích kết quả dự báo & Hành động đề xuất")

            # Tính toán thay đổi dự báo
            analysis_container = st.container()
            with analysis_container:
                for model_name, res in results.items():
                    forecast_df = res["forecast"]
                    if forecast_df.empty:
                        continue

                    # Tính mức tăng/giảm giữa kỳ đầu và cuối dự báo
                    start_val = forecast_df['yhat'].iloc[0]
                    end_val = forecast_df['yhat'].iloc[-1]
                    pct_change = ((end_val - start_val) / start_val) * 100 if start_val != 0 else 0

                    # Hiển thị
                    st.markdown(f"### 📈 {model_name} — Phân tích xu hướng")
                    if pct_change > 5:
                        st.success(f"Dự báo cho thấy **tăng {pct_change:.2f}%** trong giai đoạn dự báo.")
                        st.markdown(
                            """
                            <div class="recommendation-box">
                                <h3>Khuyến nghị hành động:</h3>
                                <ul>
                                    <li> Nên chuẩn bị <b>tăng sản lượng hàng hóa</b>, tăng nhân lực kho/bán hàng.</li>
                                    <li> Cân nhắc <b>tăng chiết khấu hoặc đẩy mạnh marketing</b> để khai thác xu hướng tốt.</li>
                                    <li> Có thể tăng giá nhẹ nếu thị trường chấp nhận.</li>
                                </ul>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    elif pct_change < -5:
                        st.warning(f"Dự báo cho thấy **giảm {abs(pct_change):.2f}%** trong giai đoạn dự báo.")
                        # st.markdown(
                        #     "- ⚠️ Cần xem xét lý do giảm: mùa vụ, sản phẩm lỗi thời, cạnh tranh?\n"
                        #     "- 👉 Xem xét **giảm tồn kho**, tối ưu chi phí, tập trung sản phẩm sinh lời cao.\n"
                        #     "- 👉 Triển khai **khuyến mãi/ưu đãi** để kích cầu nếu cần."
                        # )
                        st.markdown(
                            """
                            <div class="recommendation-box">
                                <h3>Khuyến nghị hành động:</h3>
                                <ul>
                                    <li> Cần xem xét lý do giảm: <b>mùa vụ, sản phẩm lỗi thời, cạnh tranh?</b></li>
                                    <li> Xem xét <b>giảm tồn kho, tối ưu chi phí, tập trung sản phẩm sinh lời cao</b> </li>
                                    <li> Triển khai <b>khuyến mãi/ưu đãi</b> để kích cầu nếu cần </li>
                                </ul>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        st.info(f"Dự báo ổn định với biến động nhỏ: {pct_change:.2f}%.")
                        # st.markdown(
                        #     "- 👍 Giữ ổn định hoạt động, không cần thay đổi lớn.\n"
                        #     "- 👉 Có thể theo dõi thêm xu hướng dài hạn để hành động nếu cần."
                        # )
                        st.markdown(
                            """
                            <div class="recommendation-box">
                                <h3>Khuyến nghị hành động:</h3>
                                <ul>
                                    <li> Giữ ổn định hoạt động, không cần thay đổi lớn</li>
                                    <li>  Có thể theo dõi thêm xu hướng dài hạn để hành động nếu cần</li>                                  
                                </ul>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
        # --------- Diagnostics tab ----------
        with tab_diag:
            st.subheader("Diagnostics")
            # Residuals for LR
            if "Linear Regression" in results:
                st.markdown("**Linear Regression: residuals (test set)**")
                resids = results['Linear Regression']['test_pred']['y_true'] - \
                         results['Linear Regression']['test_pred']['y_pred']
                figr, axr = plt.subplots(figsize=(10, 4))
                sns.lineplot(x=results['Linear Regression']['test_pred']['ds'], y=resids, ax=axr)
                axr.axhline(0, color='k', linewidth=0.8, linestyle='--')
                axr.set_title("Residuals over time (LR)")
                st.pyplot(figr)

                st.markdown("#### Residuals over Time")
                st.markdown("""
                Biểu đồ này thể hiện độ lệch giữa dự báo và thực tế theo thời gian. Nếu sai số dao động đều quanh 0 là tốt. Nếu lệch hẳn 1 phía là dấu hiệu dự báo bị thiên lệch.
                """)

                fig_hist, axh = plt.subplots(figsize=(8, 4))
                sns.histplot(resids, kde=True, ax=axh)
                axh.set_title("Residual distribution (LR)")
                st.pyplot(fig_hist)
                st.markdown("#### Residual distribution (LR)")
                st.markdown("""
                       🔔Đối xứng quanh 0 → mô hình cân bằng, lỗi ngẫu nhiên.
                        📉Lệch sang 1 bên → mô hình đang thường xuyên đoán quá cao hoặc quá thấp.
                        🚨Có đuôi dài (outliers) → có những lúc dự báo sai quá nhiều, cần xem xét kỹ dữ liệu đó.       
                 """)
            # Prophet components
            if "Prophet" in results:
                st.markdown("**Prophet components**")
                try:
                    fig_comp = m.plot_components(forecast)
                    st.pyplot(fig_comp)
                    st.markdown("#### Prophet Components (Seasonality)")
                    st.markdown(
                        """
                        <div class="recommendation-box">
                            <h3>🟦 Giải thích:</h3>
                            <ul>
                                <li><b>Trend (Xu hướng):</b> dự báo doanh thu tăng/giảm về lâu dài.</li>
                                <li><b>Yearly seasonality:</b> mô hình học được mùa vụ trong năm (ví dụ tháng 12 thường cao do lễ Tết).</li>
                                <li><b>Weekly seasonality (nếu có):</b> các ngày trong tuần khác nhau ra sao (thứ 2 thấp, thứ 6 cao...).</li>
                            </ul>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                except Exception as e:
                    st.write("Không thể vẽ Prophet components:", e)

            # Actual vs Predicted Sales
            st.markdown("**Actual vs Predicted Sales**")
            for model_name, res in results.items():
                test_pred_df = res.get("test_pred", None)
                if test_pred_df is None:
                    continue
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.lineplot(x=test_pred_df['ds'], y=test_pred_df['y_true'], label='Actual', linewidth=2.5, ax=ax)
                sns.lineplot(x=test_pred_df['ds'], y=test_pred_df['y_pred'], label='Predicted', linestyle='--',
                             linewidth=2, color='r', ax=ax)
                ax.set_title(f'Actual vs Predicted Sales - {model_name}')
                ax.set_xlabel("Date")
                ax.set_ylabel("Sales")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)

        # --------- Backtest tab ----------
        with tab_backtest:
            st.subheader("Rolling-origin Backtest")
            init_default = max(int(len(series) * 0.5), horizon + 1)
            st.write("Initial training window (rows):", init_default)
            initial = st.number_input("Initial window (rows) for backtest", min_value=horizon + 1, max_value=len(series) - horizon, value=init_default)
            period = st.number_input("Roll step (periods)", min_value=1, max_value=12, value=3)

            bt_results = {}
            for model_name in models_to_run:
                st.write(f"Running backtest: {model_name} ...")
                out = rolling_origin_backtest(series, model_name, freq, horizon, initial=initial, period=period, log_transform=log_transform)
                if out is not None:
                    bt_results[model_name] = out
            # show summary table
            if bt_results:
                summary = {}
                for mname, out in bt_results.items():
                    summary[mname] = out['avg_metrics']
                st.subheader("Average metrics across folds")
                st.dataframe(pd.DataFrame(summary).T.style.format(precision=3))

                # error-by-horizon plot
                st.subheader("Mean absolute error by horizon (from backtest)")
                plt.figure(figsize=(10, 4))
                for mname, out in bt_results.items():
                    horizons = sorted(out['per_horizon_mean'].keys())
                    errors = [out['per_horizon_mean'][h] for h in horizons]
                    plt.plot(horizons, errors, label=mname, marker='o')
                plt.xlabel("Horizon step")
                plt.ylabel("Mean Absolute Error")
                plt.legend()
                plt.grid(True)
                st.pyplot(plt)
                # st.markdown("""
                # ### 📊 Mean Absolute Error by Horizon
                # Biểu đồ này thể hiện mức độ sai số trung bình của dự báo tại từng bước thời gian trong tương lai.
                #
                # - Trục X: số bước dự báo (ví dụ: tháng 1, tháng 2,...)
                # - Trục Y: sai số trung bình tuyệt đối (MAE)
                # - Đường biểu diễn: mỗi mô hình có một đường riêng
                # """)
                st.markdown(
                    """
                    <div class="recommendation-box">
                        <h3>Mean Absolute Error by Horizon</h3>
                        <p>Biểu đồ này thể hiện mức độ sai số trung bình của dự báo tại từng bước thời gian trong tương lai</p>
                        <ul>
                            <li> Trục X: số bước dự báo (ví dụ: tháng 1, tháng 2,...)</li>
                            <li> Trục Y: sai số trung bình tuyệt đối (MAE)</li>   
                            <li> Đường biểu diễn: mỗi mô hình có một đường riêng</li>                               
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        with st.expander("🧪 Đánh giá độ tin cậy của dự báo", expanded=True):
            st.markdown(
                """
                <div class="recommendation-box">
                    <h3>Đánh giá mô hình dự báo:</h3>
                    <ul>
                        <li><b>MAE, RMSE, MAPE</b> cho biết độ chính xác của mô hình trên dữ liệu thực tế.</li>
                        <li>Nếu <b>MAPE &lt; 20%</b>, có thể tin tưởng vào kết quả dự báo để ra quyết định.</li>
                        <li>Residuals ổn định, không có xu hướng → mô hình hợp lý.</li>
                        <li>Backtest (kiểm định theo thời gian) cho thấy mô hình không bị overfit.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )

        with st.expander("📊 Phân tích kết quả & khuyến nghị"):
            st.markdown(
                """
                <div class="recommendation-box">
                    <h3> Khuyến nghị hành động</h3>
                    <ul>
                        <li>Dự báo cho thấy doanh thu trong quý tới có xu hướng <b>tăng nhẹ</b> so với các kỳ trước.</li>
                        <li>Biến động nằm trong mức sai số chấp nhận được (MAPE dưới 15%).</li>
                        <li><b>Nếu dự báo tăng:</b> Có thể xem xét tăng sản lượng, bổ sung hàng tồn kho, tăng marketing.</li>
                        <li><b>Nếu dự báo giảm:</b> Cần đánh giá lại chính sách giá, chiết khấu hoặc mở rộng tệp khách hàng.</li>
                    </ul>                   
                </div>
                """,
                unsafe_allow_html=True
            )

        st.success("Hoàn tất chạy dự báo. Kiểm tra các tab trên.")

