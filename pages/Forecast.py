# # pages/Forecast.py
# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# from sklearn.linear_model import Ridge
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# from prophet import Prophet
#
# from components.data_loader import load_data
#
# st.set_page_config(page_title="Forecast", layout="wide")
# st.title("05 - Forecast: Dự báo doanh thu")
#
# # === Controls moved INTO page (not sidebar) ===
# with st.expander("Filter & Forecast controls", expanded=True):
#     col1, col2, col3 = st.columns([1,1,1])
#     with col1:
#         target = st.selectbox("Chọn chỉ số dự báo", ["SALES", "TOTAL_ORDER_VALUE"])
#         freq_label = st.selectbox("Tần suất tổng hợp", ["Daily", "Weekly", "Monthly"], index=2)
#     with col2:
#         horizon = st.number_input("Số kỳ dự báo (periods)", min_value=1, max_value=24, value=6)
#         test_size = st.slider("Tỷ lệ tập test (theo thời gian)", 0.05, 0.4, 0.2)
#     with col3:
#         models_to_run = st.multiselect("Chạy mô hình", ["Linear Regression", "Prophet"],
#                                        default=["Linear Regression", "Prophet"])
#         run = st.button("Chạy dự báo")
#
# # Map freq: use 'ME' for month-end to avoid pandas FutureWarning
# freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "ME"}
# freq = freq_map[freq_label]
#
# # === Utility functions ===
# def aggregate_time_series(df, date_col, value_col, freq):
#     df = df.copy()
#     df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
#     ts = df.set_index(date_col)[value_col].resample(freq).sum().reset_index()
#     ts.columns = ['ds', 'y']
#     idx = pd.date_range(ts['ds'].min(), ts['ds'].max(), freq=freq)
#     ts = ts.set_index('ds').reindex(idx).rename_axis('ds').reset_index()
#     ts['y'] = ts['y'].fillna(0)
#     return ts
#
# def prepare_lr_features(series, freq):
#     df = series.copy().sort_values('ds').reset_index(drop=True)
#     df['t'] = np.arange(len(df))
#     df['t2'] = df['t'] ** 2
#     # month dummies (create m_1..m_11)
#     df['month'] = df['ds'].dt.month
#     month_dummies = pd.get_dummies(df['month'], prefix='m', drop_first=True)
#     df = pd.concat([df, month_dummies], axis=1)
#     period = 12 if freq in ['M','ME'] else 52 if freq == 'W' else 365
#     df['sin1'] = np.sin(2 * np.pi * df['t'] / period)
#     df['cos1'] = np.cos(2 * np.pi * df['t'] / period)
#     # X columns: keep deterministic order
#     drop_cols = ['ds', 'y', 'month']
#     X = df.drop(columns=drop_cols)
#     return X, df['y'], df
#
# def time_train_test_split(X, y, test_size):
#     n = len(X)
#     t = int(n * (1 - test_size))
#     return X.iloc[:t], X.iloc[t:], y.iloc[:t], y.iloc[t:], t
#
# import numpy as np
# from sklearn.metrics import mean_absolute_error, mean_squared_error
#
# def compute_metrics(y_true, y_pred):
#     """
#     Trả về dict: {'MAE', 'RMSE', 'MAPE'}
#     - Tương thích với nhiều phiên bản sklearn (không phụ thuộc vào param squared).
#     - Bỏ các mẫu có y_true == 0 khi tính MAPE (trả về None nếu không có mẫu hợp lệ).
#     """
#     # Chuyển về numpy float arrays
#     y_true_arr = np.asarray(y_true).astype(float)
#     y_pred_arr = np.asarray(y_pred).astype(float)
#
#     if y_true_arr.shape[0] != y_pred_arr.shape[0]:
#         raise ValueError(f"compute_metrics: y_true và y_pred có số mẫu khác nhau: "
#                          f"{y_true_arr.shape[0]} != {y_pred_arr.shape[0]}")
#
#     mae = mean_absolute_error(y_true_arr, y_pred_arr)
#
#     # RMSE: dùng try/except để hỗ trợ các phiên bản sklearn cũ hơn
#     try:
#         # nếu sklearn hỗ trợ param 'squared'
#         rmse = mean_squared_error(y_true_arr, y_pred_arr, squared=False)
#     except TypeError:
#         rmse = np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))
#
#     # MAPE: loại bỏ các y_true == 0 để tránh chia cho 0
#     nonzero_mask = y_true_arr != 0
#     if nonzero_mask.sum() == 0:
#         mape = None
#     else:
#         mape = np.mean(np.abs((y_true_arr[nonzero_mask] - y_pred_arr[nonzero_mask]) /
#                                y_true_arr[nonzero_mask])) * 100.0
#
#     return {"MAE": float(mae), "RMSE": float(rmse), "MAPE": (float(mape) if mape is not None else None)}
#
#
# def build_future_lr_features(df_feat, horizon, freq):
#     # df_feat: the dataframe returned from prepare_lr_features (with ds, t, t2, month,..)
#     last_t = df_feat['t'].iloc[-1]
#     future_t = np.arange(last_t + 1, last_t + 1 + horizon)
#     # compute next ds start: add one period
#     # note: pd.Timedelta with unit=freq doesn't accept 'ME' -> use offset alias mapping
#     freq_for_offset = {'D':'D','W':'W','ME':'M'}[freq]
#     future_ds = pd.date_range(df_feat['ds'].max() + pd.tseries.frequencies.to_offset(freq_for_offset),
#                               periods=horizon, freq=freq)
#     future_df = pd.DataFrame({'ds': future_ds, 't': future_t})
#     future_df['t2'] = future_df['t'] ** 2
#     future_df['month'] = future_df['ds'].dt.month
#     month_dummies = pd.get_dummies(future_df['month'], prefix='m', drop_first=True)
#     # ensure we have the same dummy columns as df_feat
#     existing_month_cols = [c for c in df_feat.columns if c.startswith('m_')]
#     for c in existing_month_cols:
#         if c in month_dummies:
#             future_df[c] = month_dummies[c]
#         else:
#             future_df[c] = 0
#     period = 12 if freq in ['M','ME'] else 52 if freq == 'W' else 365
#     future_df['sin1'] = np.sin(2 * np.pi * future_df['t'] / period)
#     future_df['cos1'] = np.cos(2 * np.pi * future_df['t'] / period)
#     # Return columns in same order as X (drop ds, y, month)
#     X_future = future_df.drop(columns=['month'])
#     # Ensure columns order matches X_train
#     return X_future
#
# # === Run forecast ===
# if run:
#     with st.spinner("Đang tải dữ liệu và chuẩn bị..."):
#         df = load_data()
#         if target not in df.columns:
#             st.error(f"Cột {target} không tồn tại trong dataframe.")
#             st.stop()
#         series = aggregate_time_series(df, 'ORDERDATE', target, freq)
#
#     results = {}
#     # base plot (history)
#     fig = px.line(series, x='ds', y='y', title=f"{target} - Lịch sử và Dự báo ({freq_label})")
#     st.subheader("Dữ liệu lịch sử")
#     st.plotly_chart(fig, use_container_width=True)
#
#     # --- Linear Regression ---
#     if "Linear Regression" in models_to_run:
#         X, y, df_feat = prepare_lr_features(series, freq)
#         X_tr, X_te, y_tr, y_te, split_idx = time_train_test_split(X, y, test_size)
#         lr_model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
#         lr_model.fit(X_tr, y_tr)
#         y_pred_te = lr_model.predict(X_te)
#         metrics_lr = compute_metrics(y_te, y_pred_te)
#
#         # prepare future features with same columns as X
#         X_future_full = build_future_lr_features(df_feat, horizon, freq)
#         # X_future_full might contain 'ds' at front if kept; ensure we pass same columns as training X
#         X_future_only = X_future_full[X.columns]  # safe column alignment
#         y_future_pred = lr_model.predict(X_future_only)
#
#         future_lr_df = pd.DataFrame({'ds': X_future_full['ds'], 'yhat': y_future_pred})
#         results['Linear Regression'] = {"metrics": metrics_lr, "forecast": future_lr_df}
#
#         # add traces
#         fig.add_scatter(x=future_lr_df['ds'], y=future_lr_df['yhat'], mode='lines', name='LR Forecast')
#
#     # --- Prophet ---
#     if "Prophet" in models_to_run:
#         # Prophet expects freq like 'D','W','M' etc. it accepts 'ME' as 'M', but safe to pass freq mapped to 'M' for Prophet
#         prophet_freq = 'M' if freq in ['M','ME'] else freq
#         m = Prophet(yearly_seasonality=True, weekly_seasonality=(prophet_freq == 'D'))
#         m.fit(series[['ds', 'y']])
#         forecast = m.make_future_dataframe(periods=horizon, freq=prophet_freq)
#         forecast = m.predict(forecast)
#
#         # IMPORTANT FIX: align test slice sizes — only take forecast yhat for historical test period, not including future horizon
#         n = len(series)
#         test_cutoff = int(n * (1 - test_size))
#         # y_true_test: last portion of historical series
#         y_true_test = series['y'].iloc[test_cutoff:].reset_index(drop=True)
#         # y_pred_test: corresponding predictions from forecast for the same historical dates
#         # forecast rows 0..n-1 correspond to historical dates; so slice forecast['yhat'][test_cutoff : n]
#         y_pred_test = forecast['yhat'].iloc[test_cutoff:n].reset_index(drop=True)
#
#         # safety check
#         if len(y_true_test) != len(y_pred_test):
#             st.warning("Kích thước y_true và y_pred không khớp — kiểm tra lại. "
#                        f"len(y_true)={len(y_true_test)}, len(y_pred)={len(y_pred_test)}")
#         metrics_prophet = compute_metrics(y_true_test, y_pred_test)
#
#         future_prophet_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].iloc[n:].reset_index(drop=True)
#         results['Prophet'] = {"metrics": metrics_prophet, "forecast": future_prophet_df}
#
#         # add to plot
#         fig.add_scatter(x=future_prophet_df['ds'], y=future_prophet_df['yhat'], mode='lines', name='Prophet Forecast')
#         # optionally add CI as shaded area (Plotly requires trace for lower and upper)
#         fig.add_scatter(x=list(future_prophet_df['ds']) + list(future_prophet_df['ds'][::-1]),
#                         y=list(future_prophet_df['yhat_upper']) + list(future_prophet_df['yhat_lower'][::-1]),
#                         fill='toself', fillcolor='rgba(200,200,200,0.2)', line=dict(color='rgba(255,255,255,0)'),
#                         hoverinfo="skip", name='Prophet CI')
#
#     # === Final plot with forecasts ===
#     st.subheader("Lịch sử + dự báo")
#     st.plotly_chart(fig, use_container_width=True)
#
#     # === Metrics table ===
#     if results:
#         st.subheader("Kết quả đánh giá (trên tập test)")
#         metrics_df = pd.DataFrame({model: res['metrics'] for model, res in results.items()}).T
#         st.dataframe(metrics_df)
#
#         # Download buttons
#         for model, res in results.items():
#             csv = res['forecast'].to_csv(index=False)
#             st.download_button(f"Tải dự báo ({model})", csv, file_name=f"{model}_forecast.csv", mime="text/csv")
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
        with open(file_name) as f:
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
        per_horizon_errors = {h+1: [] for h in range(horizon)}  # horizon step -> list of MAE across folds (or RMSE)
        progress = st.progress(0)
        for i, start in enumerate(starts):
            train = series.iloc[:start].reset_index(drop=True)
            test = series.iloc[start:start + horizon].reset_index(drop=True)

            # Prepare train labels (maybe log)
            if log_transform:
                train_y_for_model = np.log1p(train['y'])
            else:
                train_y_for_model = train['y']

            if model_name == "Linear Regression":
                # Fit on train
                X_train, y_train_dummy, df_feat_train = prepare_lr_features(pd.DataFrame({'ds': train['ds'], 'y': train_y_for_model}), freq)
                lr = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
                lr.fit(X_train, y_train_dummy)
                # Prepare test features: use test ds and align to X_train cols
                X_test_raw, _, df_feat_test = prepare_lr_features(pd.DataFrame({'ds': test['ds'], 'y': test['y']}), freq)
                X_test = align_features_to_ref(X_test_raw, X_train.columns)
                pred = lr.predict(X_test)
                if log_transform:
                    pred = np.expm1(pred)
            else:  # Prophet
                m = Prophet(yearly_seasonality=True)
                df_train_prophet = pd.DataFrame({'ds': train['ds'], 'y': train_y_for_model})
                m.fit(df_train_prophet)
                # create future with length = len(train)+horizon, then take last horizon
                future = m.make_future_dataframe(periods=horizon, freq=('M' if freq in ['M', 'ME'] else freq))
                fc = m.predict(future)
                # predictions for test correspond to last horizon rows
                pred = fc['yhat'].iloc[-horizon:].values
                if log_transform:
                    pred = np.expm1(pred)

            # compute metrics against actual test['y']
            metrics = compute_metrics(test['y'].values, pred)
            all_metrics.append(metrics)
            # collect per-horizon absolute errors
            abs_errs = np.abs(test['y'].values - pred)
            for h_step in range(len(abs_errs)):
                per_horizon_errors[h_step + 1].append(abs_errs[h_step])

            progress.progress((i + 1) / len(starts))

        # aggregate metrics: compute mean across folds for each metric
        metrics_df = pd.DataFrame(all_metrics)
        avg_metrics = metrics_df.mean().to_dict()
        # compute mean error per horizon
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

    # ====== Run ======
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

            # metrics on test (compare to original-scale y)
            metrics_lr = compute_metrics(series['y'].iloc[split_idx:].values, y_pred_te)

            # future forecast
            X_future_full = build_future_lr_features(df_feat, horizon, freq)
            X_future_only = align_features_to_ref(X_future_full.drop(columns=['ds']), X_tr.columns)
            y_future_pred = lr_model.predict(X_future_only)
            if log_transform:
                y_future_pred = np.expm1(y_future_pred)
            future_lr_df = pd.DataFrame({'ds': X_future_full['ds'], 'yhat': y_future_pred})

            results['Linear Regression'] = {"metrics": metrics_lr, "forecast": future_lr_df}

        # --------- Prophet ----------
        if "Prophet" in models_to_run:
            prophet_freq = 'M' if freq in ['M', 'ME'] else freq
            m = Prophet(yearly_seasonality=True, weekly_seasonality=(prophet_freq == 'D'))
            m.fit(series_model[['ds', 'y']])
            forecast = m.make_future_dataframe(periods=horizon, freq=prophet_freq)
            forecast = m.predict(forecast)

            n = len(series)
            test_cutoff = int(n * (1 - test_size))
            y_true_test = series['y'].iloc[test_cutoff:n].reset_index(drop=True)
            y_pred_test = forecast['yhat'].iloc[test_cutoff:n].reset_index(drop=True)
            if log_transform:
                y_pred_test = np.expm1(y_pred_test)
            metrics_prophet = compute_metrics(y_true_test.values, y_pred_test.values)

            future_prophet_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].iloc[n:].reset_index(drop=True)
            if log_transform:
                future_prophet_df[['yhat', 'yhat_lower', 'yhat_upper']] = np.expm1(future_prophet_df[['yhat', 'yhat_lower', 'yhat_upper']])
            results['Prophet'] = {"metrics": metrics_prophet, "forecast": future_prophet_df}

        # --------- Overview tab: plot results and metrics ----------
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

        # --------- Diagnostics tab ----------
        with tab_diag:
            st.subheader("Diagnostics")
            # Residuals for LR
            if "Linear Regression" in results:
                st.markdown("**Linear Regression: residuals (test set)**")
                resids = series['y'].iloc[split_idx:].values - y_pred_te
                figr, axr = plt.subplots(figsize=(10, 4))
                sns.lineplot(x=series['ds'].iloc[split_idx:], y=resids, ax=axr)
                axr.axhline(0, color='k', linewidth=0.8, linestyle='--')
                axr.set_title("Residuals over time (LR)")
                st.pyplot(figr)

                fig_hist, axh = plt.subplots(figsize=(8, 4))
                sns.histplot(resids, kde=True, ax=axh)
                axh.set_title("Residual distribution (LR)")
                st.pyplot(fig_hist)

            # Prophet components
            if "Prophet" in results:
                st.markdown("**Prophet components**")
                try:
                    fig_comp = m.plot_components(forecast)
                    st.pyplot(fig_comp)
                except Exception as e:
                    st.write("Không thể vẽ Prophet components:", e)

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

        st.success("Hoàn tất chạy dự báo. Kiểm tra các tab trên.")

