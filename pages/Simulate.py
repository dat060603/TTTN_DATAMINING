#
# # pages/Simulate.py
# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go
# import plotly.express as px
# from prophet import Prophet
# from datetime import datetime
#
# from components.data_loader import load_data
#
# st.set_page_config(page_title="Simulate - What-if", layout="wide")
# st.title("Simulate: Công cụ What-if (Mô phỏng kịch bản)")
#
# # ---------------- Helpers ----------------
# @st.cache_data(show_spinner=False)
# def load_and_prep():
#     df = load_data()
#     if 'ORDERDATE' in df.columns:
#         df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'], errors='coerce')
#     return df
#
# def aggregate(df, date_col='ORDERDATE', value_col='SALES', freq='M'):
#     df = df.copy()
#     df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
#     s = df.set_index(date_col)[value_col].resample(freq).sum().reset_index()
#     s.columns = ['ds', 'y']
#     idx = pd.date_range(s['ds'].min(), s['ds'].max(), freq=freq)
#     s = s.set_index('ds').reindex(idx).rename_axis('ds').reset_index()
#     s['y'] = s['y'].fillna(0)
#     return s
#
# def build_prophet_forecast(series, periods, freq='M'):
#     m = Prophet(yearly_seasonality=True)
#     m.fit(series[['ds','y']])
#     future = m.make_future_dataframe(periods=periods, freq=freq)
#     fc = m.predict(future)
#     return fc[['ds','yhat','yhat_lower','yhat_upper']], m
#
# def compute_summary(baseline_ts, scenario_ts):
#     dfb = baseline_ts.set_index('ds')
#     dfs = scenario_ts.set_index('ds')
#     idx = dfb.index.union(dfs.index)
#     dfb = dfb.reindex(idx).fillna(0)
#     dfs = dfs.reindex(idx).fillna(0)
#     total_base = dfb['y'].sum()
#     total_scen = dfs['y'].sum()
#     delta_abs = total_scen - total_base
#     delta_pct = (delta_abs / total_base)*100 if total_base != 0 else np.nan
#     return {"base_total": float(total_base), "scenario_total": float(total_scen),
#             "delta_abs": float(delta_abs), "delta_pct": float(delta_pct)}
#
# def run_montecarlo(trans_df, apply_fn, sampler_fn, n_runs=300, freq='M'):
#     sims = []
#     for i in range(n_runs):
#         params = sampler_fn()
#         df_sim = apply_fn(params)
#         agg = aggregate(df_sim, date_col='ORDERDATE', value_col='SALES_sim', freq=freq)
#         sims.append(agg.set_index('ds')['y'])
#     sims_df = pd.concat(sims, axis=1)
#     mean_s = sims_df.mean(axis=1)
#     lower = sims_df.quantile(0.05, axis=1)
#     upper = sims_df.quantile(0.95, axis=1)
#     out = pd.DataFrame({'ds': mean_s.index, 'mean': mean_s.values, 'lower': lower.values, 'upper': upper.values})
#     return out
#
# # ---------------- UI (Tiếng Việt) ----------------
# with st.expander("Cấu hình kịch bản (Scenario controls)", expanded=True):
#     c1, c2, c3 = st.columns(3)
#     with c1:
#         agg_label = st.selectbox("Chu kỳ tổng hợp", ["Monthly", "Weekly", "Daily"], index=0)
#         freq_map = {'Monthly':'M', 'Weekly':'W', 'Daily':'D'}
#         freq = freq_map[agg_label]
#         use_upload = st.checkbox("Dùng baseline forecast từ file (CSV) thay vì Prophet", value=False)
#         uploaded_fc = st.file_uploader("Upload baseline forecast CSV (cột: ds,yhat)", type=['csv']) if use_upload else None
#     with c2:
#         # Nhập thay đổi giá
#         price_change_pct = st.number_input(
#             "Thay đổi giá (%) (ví dụ 10 = +10%)",
#             value=0.0,
#             step=0.5,
#             format="%.2f"
#         )
#
#         # Hiển thị chú thích trạng thái
#         if price_change_pct > 0:
#             st.markdown(f"💹 **{price_change_pct}% (Tăng giá)**", unsafe_allow_html=True)
#         elif price_change_pct < 0:
#             st.markdown(f"📉 **{price_change_pct}% (Giảm giá)**", unsafe_allow_html=True)
#         else:
#             st.markdown(f"⚪ **0% (Không thay đổi)**", unsafe_allow_html=True)
#
#         elasticity = st.number_input("Price elasticity (ví dụ -1.0). Nếu không biết để -1.0", value=-1.0, step=0.1)
#         volume_change_pct = st.number_input("Thay đổi lượng bán (%) trực tiếp (volume)", value=0.0, step=0.5)
#     with c3:
#         promotion = st.checkbox("Áp dụng khuyến mãi (promotion window)", value=False)
#         if promotion:
#             promo_start = st.date_input("Ngày bắt đầu khuyến mãi", value=datetime.today())
#             promo_end = st.date_input("Ngày kết thúc khuyến mãi", value=datetime.today())
#             promo_lift = st.number_input("Tăng lượng bán trong khuyến mãi (%)", value=10.0, step=1.0)
#         margin_pct = st.number_input("Tỷ lệ lợi nhuận (margin %) để ước tính profit", value=20.0, step=1.0)
#
#     st.markdown("---")
#     st.write("Monte Carlo (tùy chọn): mô phỏng bất định cho elasticity / promo lift")
#     mc_enable = st.checkbox("Bật Monte Carlo", value=False)
#     if mc_enable:
#         mc_runs = st.number_input("Số lần chạy MC", min_value=50, max_value=2000, value=300, step=50)
#         elast_sd = st.number_input("Elasticity std dev (ví dụ 0.2)", value=0.2, step=0.05)
#         promo_sd = st.number_input("Promo lift std dev (pct points)", value=2.0, step=0.5)
#
#     # productline selection: safer UX
#     st.markdown("---")
#     st.write("Chọn PRODUCTLINE để áp dụng kịch bản (mặc định: không chọn = KHÔNG áp dụng).")
#     df_tmp = load_and_prep()
#     if 'PRODUCTLINE' in df_tmp.columns:
#         all_pl = sorted(df_tmp['PRODUCTLINE'].dropna().unique().tolist())
#         apply_all_pl = st.checkbox("Apply to ALL PRODUCTLINE (áp dụng cho tất cả)", value=False)
#         if apply_all_pl:
#             selected_pl = st.multiselect("PRODUCTLINE", options=all_pl, default=all_pl)
#         else:
#             selected_pl = st.multiselect("PRODUCTLINE", options=all_pl, default=[])
#     else:
#         selected_pl = None
#         st.info("Không tìm thấy cột PRODUCTLINE trong dữ liệu — kịch bản sẽ áp dụng cho toàn bộ dữ liệu (fallback).")
#
#     run_button = st.button("Chạy mô phỏng (Run Simulation)")
#
# # ---------------- Run simulation ----------------
# if run_button:
#     df = load_and_prep()
#     st.success(f"Đã tải dữ liệu ({len(df)} rows).")
#
#     # Baseline forecast: upload hoặc tự tính bằng Prophet
#     if use_upload and uploaded_fc is not None:
#         try:
#             baseline_fc = pd.read_csv(uploaded_fc, parse_dates=['ds'])
#             baseline_ts = baseline_fc.rename(columns={'yhat':'y'})[['ds','y']]
#             st.info("Đã dùng baseline từ file upload.")
#         except Exception as e:
#             st.error(f"Không đọc được file forecast: {e}")
#             baseline_ts = None
#     else:
#         hist_ts = aggregate(df, date_col='ORDERDATE', value_col='SALES', freq=freq)
#         periods = int(st.number_input("Số kỳ forecast cho baseline (periods)", min_value=1, max_value=36, value=6))
#         with st.spinner("Huấn luyện Prophet cho baseline..."):
#             baseline_fc, _ = build_prophet_forecast(hist_ts, periods=periods, freq=freq)
#             baseline_ts = baseline_fc[['ds','yhat']].rename(columns={'yhat':'y'})
#         st.info("Baseline Prophet đã sẵn sàng.")
#
#     # Build scenario: apply price/volume/promo on transaction-level df
#     df_sim = df.copy()
#
#     # Create mask for selected productlines
#     if selected_pl is None:
#         # No PRODUCTLINE column found -> apply to all
#         mask_pl_all = pd.Series(True, index=df_sim.index)
#     else:
#         if len(selected_pl) == 0:
#             # default: user didn't select any -> DO NOT apply to anyone
#             mask_pl_all = pd.Series(False, index=df_sim.index)
#         else:
#             mask_pl_all = df_sim['PRODUCTLINE'].isin(selected_pl)
#
#     # For debug: show chosen selection and counts
#     st.write("Selected PRODUCTLINE:", (selected_pl if selected_pl is not None else "No PRODUCTLINE column"))
#     st.write("Apply mask count (rows affected):", int(mask_pl_all.sum()))
#
#     # Prepare simulation columns safely (start from original values)
#     # Use .get to avoid KeyError when column missing
#     df_sim['PRICEEACH_sim'] = df_sim.get('PRICEEACH', np.nan)
#     df_sim['QUANTITYORDERED_sim'] = df_sim.get('QUANTITYORDERED', np.nan)
#
#     # Price change applied only to selected productlines (via mask)
#     pct_price = price_change_pct / 100.0
#     if pct_price != 0 and 'PRICEEACH' in df_sim.columns:
#         df_sim.loc[mask_pl_all, 'PRICEEACH_sim'] = df_sim.loc[mask_pl_all, 'PRICEEACH'] * (1 + pct_price)
#
#     # Volume direct change applied only to selected productlines
#     pct_vol = volume_change_pct / 100.0
#     if pct_vol != 0 and 'QUANTITYORDERED' in df_sim.columns:
#         df_sim.loc[mask_pl_all, 'QUANTITYORDERED_sim'] = df_sim.loc[mask_pl_all, 'QUANTITYORDERED'] * (1 + pct_vol)
#
#     # Elasticity effect applied only to selected productlines if price changed
#     if pct_price != 0 and 'QUANTITYORDERED' in df_sim.columns:
#         delta_q_pct = elasticity * pct_price
#         # ensure QUANTITYORDERED_sim exists
#         df_sim['QUANTITYORDERED_sim'] = df_sim['QUANTITYORDERED_sim'].fillna(df_sim['QUANTITYORDERED'])
#         df_sim.loc[mask_pl_all, 'QUANTITYORDERED_sim'] = df_sim.loc[mask_pl_all, 'QUANTITYORDERED_sim'] * (1 + delta_q_pct)
#
#     # Promotion window applied only to selected productlines
#     if promotion:
#         start = pd.to_datetime(promo_start)
#         end = pd.to_datetime(promo_end)
#         mask_date = (pd.to_datetime(df_sim['ORDERDATE']) >= start) & (pd.to_datetime(df_sim['ORDERDATE']) <= end)
#         mask_promo = mask_date & mask_pl_all
#         lift = promo_lift / 100.0
#         # ensure QUANTITYORDERED_sim exists
#         if 'QUANTITYORDERED_sim' not in df_sim.columns:
#             df_sim['QUANTITYORDERED_sim'] = df_sim.get('QUANTITYORDERED', 0)
#         df_sim.loc[mask_promo, 'QUANTITYORDERED_sim'] = df_sim.loc[mask_promo, 'QUANTITYORDERED_sim'] * (1 + lift)
#
#     # Recompute SALES_sim carefully (only from sim columns; fallback to original SALES)
#     if 'PRICEEACH_sim' in df_sim.columns and 'QUANTITYORDERED_sim' in df_sim.columns:
#         df_sim['SALES_sim'] = df_sim['SALES']  # giữ nguyên mặc định
#         df_sim.loc[mask_pl_all, 'SALES_sim'] = df_sim.loc[mask_pl_all, 'PRICEEACH_sim'] * df_sim.loc[
#             mask_pl_all, 'QUANTITYORDERED_sim']
#
#     elif 'PRICEEACH_sim' in df_sim.columns and 'QUANTITYORDERED' in df_sim.columns:
#         df_sim['SALES_sim'] = df_sim['PRICEEACH_sim'] * df_sim['QUANTITYORDERED']
#     elif 'PRICEEACH' in df_sim.columns and 'QUANTITYORDERED_sim' in df_sim.columns:
#         df_sim['SALES_sim'] = df_sim['PRICEEACH'] * df_sim['QUANTITYORDERED_sim']
#     else:
#         df_sim['SALES_sim'] = df_sim.get('SALES', 0)
#
#     # Ensure unaffected rows keep original SALES (no accidental NaN)
#     df_sim['SALES_sim'] = df_sim['SALES_sim'].fillna(df_sim.get('SALES', 0))
#
#     # Aggregate historical baseline and scenario overall
#     hist_agg = aggregate(df, date_col='ORDERDATE', value_col='SALES', freq=freq)
#     scen_agg = aggregate(df_sim, date_col='ORDERDATE', value_col='SALES_sim', freq=freq)
#
#     # Merge historical for overall comparison
#     merged = pd.merge(hist_agg.rename(columns={'y':'baseline'}), scen_agg.rename(columns={'y':'scenario'}), on='ds', how='outer').sort_values('ds').fillna(0)
#
#     # Summary overall
#     summary = compute_summary(merged[['ds','baseline']].rename(columns={'baseline':'y'}), merged[['ds','scenario']].rename(columns={'scenario':'y'}))
#     st.subheader("Tóm tắt (Historical) — Tổng quan")
#     st.markdown(f"- Tổng baseline (historical): **{summary['base_total']:.0f}**")
#     st.markdown(f"- Tổng scenario (historical): **{summary['scenario_total']:.0f}**")
#     st.markdown(f"- Delta (abs): **{summary['delta_abs']:.0f}**, Delta (%): **{summary['delta_pct']:.2f}%**")
#
#     # Plot historical baseline vs scenario (overall)
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=merged['ds'], y=merged['baseline'], mode='lines', name='Baseline (historical)'))
#     fig.add_trace(go.Scatter(x=merged['ds'], y=merged['scenario'], mode='lines', name='Scenario (historical)'))
#     fig.update_layout(title="Baseline vs Scenario (Historical) — Tổng quan", xaxis_title="Ngày", yaxis_title="Doanh thu")
#     st.plotly_chart(fig, use_container_width=True)
#
#     # Breakdown theo PRODUCTLINE (nếu có)
#     if 'PRODUCTLINE' in df.columns:
#         st.subheader("Breakdown theo PRODUCTLINE")
#         base_by_pl = df.groupby('PRODUCTLINE').apply(lambda d: d['SALES'].sum()).rename('baseline_total').reset_index()
#         scen_by_pl = df_sim.groupby('PRODUCTLINE').apply(lambda d: d['SALES_sim'].sum()).rename('scenario_total').reset_index()
#         by_pl = pd.merge(base_by_pl, scen_by_pl, on='PRODUCTLINE', how='outer').fillna(0)
#         by_pl['delta_abs'] = by_pl['scenario_total'] - by_pl['baseline_total']
#         by_pl['delta_pct'] = np.where(by_pl['baseline_total']==0, np.nan, by_pl['delta_abs'] / by_pl['baseline_total'] * 100)
#         # mark affected productlines
#         if selected_pl is None:
#             by_pl['affected'] = True
#         else:
#             by_pl['affected'] = by_pl['PRODUCTLINE'].isin(selected_pl)
#         # sort so affected appear first
#         by_pl = by_pl.sort_values(['affected','delta_abs'], ascending=[False, False]).reset_index(drop=True)
#         st.dataframe(by_pl.style.format({"baseline_total":"{:.0f}", "scenario_total":"{:.0f}", "delta_abs":"{:.0f}", "delta_pct":"{:.2f}%"}))
#
#         # bar chart delta by productline
#         fig_pl = px.bar(by_pl, x='PRODUCTLINE', y='delta_abs', title='Delta doanh thu theo PRODUCTLINE (scenario - baseline)')
#         st.plotly_chart(fig_pl, use_container_width=True)
#     else:
#         st.info("Không có cột PRODUCTLINE trong dữ liệu — bỏ qua breakdown theo PRODUCTLINE.")
#
#     # Project scenario onto baseline forecast tail
#     project_future = st.checkbox("Áp dụng thay đổi scenario lên phần dự báo tương lai của baseline (project to future)", value=True)
#     projected_df = None
#     if project_future and baseline_ts is not None:
#         max_hist = hist_agg['ds'].max()
#         L = st.number_input("Dùng L kỳ cuối để tính % thay đổi (L)", min_value=1, max_value=12, value=3)
#         last_base = merged.tail(L)['baseline'].mean()
#         last_scen = merged.tail(L)['scenario'].mean()
#         pct_change = 0.0 if last_base == 0 else (last_scen - last_base) / last_base
#         st.write(f"Avg % change over last {L} periods: {pct_change*100:.2f}%")
#         baseline_tail = baseline_ts[baseline_ts['ds'] > max_hist].copy()
#         if baseline_tail.empty:
#             st.warning("Không có phần forecast trong baseline (tail) để project.")
#         else:
#             baseline_tail['scenario_proj'] = baseline_tail['y'] * (1 + pct_change)
#             projected_df = baseline_tail[['ds','y','scenario_proj']].rename(columns={'y':'baseline','scenario_proj':'scenario'})
#             fig2 = go.Figure()
#             fig2.add_trace(go.Scatter(x=merged['ds'], y=merged['baseline'], mode='lines', name='Baseline (hist)'))
#             fig2.add_trace(go.Scatter(x=merged['ds'], y=merged['scenario'], mode='lines', name='Scenario (hist)'))
#             fig2.add_trace(go.Scatter(x=baseline_tail['ds'], y=baseline_tail['y'], mode='lines', name='Baseline (forecast tail)', line=dict(dash='dash')))
#             fig2.add_trace(go.Scatter(x=baseline_tail['ds'], y=baseline_tail['y']*(1+pct_change), mode='lines', name='Scenario (projected tail)', line=dict(dash='dash')))
#             fig2.update_layout(title="Historical + Projected Tail", xaxis_title="Ngày", yaxis_title="Doanh thu")
#             st.plotly_chart(fig2, use_container_width=True)
#
#     # Profit simple estimate
#     margin = margin_pct / 100.0
#     profit_base = merged['baseline'].sum() * margin
#     profit_scen = merged['scenario'].sum() * margin
#     st.subheader("Ước tính lợi nhuận (đơn giản)")
#     st.markdown(f"- Base profit (historical) = **{profit_base:.0f}**")
#     st.markdown(f"- Scenario profit (historical) = **{profit_scen:.0f}**")
#     st.markdown(f"- Delta profit = **{(profit_scen - profit_base):.0f}**")
#
#     # Monte Carlo
#     if mc_enable:
#         st.subheader("Monte Carlo simulation")
#         st.info("Sampling elasticity and promo lift; building distribution for scenario revenue.")
#         def sampler():
#             return {'elasticity': np.random.normal(loc=elasticity, scale=elast_sd),
#                     'promo_lift': np.random.normal(loc=promo_lift if promotion else 0.0, scale=promo_sd)}
#         def apply_fn(params):
#             dfi = df.copy()
#             # mask selected productlines
#             if selected_pl is None:
#                 mask_pl = pd.Series(True, index=dfi.index)
#             else:
#                 mask_pl = dfi['PRODUCTLINE'].isin(selected_pl) if len(selected_pl) > 0 else pd.Series(False, index=dfi.index)
#             # apply elasticity-driven quantity change if price change present
#             if pct_price != 0 and 'QUANTITYORDERED' in dfi.columns:
#                 delta_q_pct = params['elasticity'] * pct_price
#                 dfi['QUANTITYORDERED_sim'] = dfi['QUANTITYORDERED']
#                 dfi.loc[mask_pl, 'QUANTITYORDERED_sim'] = dfi.loc[mask_pl, 'QUANTITYORDERED'] * (1 + delta_q_pct)
#             else:
#                 dfi['QUANTITYORDERED_sim'] = dfi.get('QUANTITYORDERED', 0)
#             # promo lift
#             if promotion and params['promo_lift'] != 0:
#                 mask = (pd.to_datetime(dfi['ORDERDATE']) >= pd.to_datetime(promo_start)) & (pd.to_datetime(dfi['ORDERDATE']) <= pd.to_datetime(promo_end))
#                 mask_full = mask & mask_pl
#                 dfi.loc[mask_full, 'QUANTITYORDERED_sim'] = dfi.loc[mask_full, 'QUANTITYORDERED_sim'] * (1 + params['promo_lift']/100.0)
#             # price
#             if 'PRICEEACH' in dfi.columns:
#                 dfi['PRICEEACH_sim'] = dfi['PRICEEACH']
#                 dfi.loc[mask_pl, 'PRICEEACH_sim'] = dfi.loc[mask_pl, 'PRICEEACH'] * (1 + pct_price)
#                 dfi['SALES_sim'] = dfi['PRICEEACH_sim'] * dfi['QUANTITYORDERED_sim']
#             else:
#                 dfi['SALES_sim'] = dfi.get('SALES', 0)
#             return dfi
#
#         with st.spinner("Running Monte Carlo..."):
#             mc_out = run_montecarlo(df, apply_fn, sampler, n_runs=int(mc_runs), freq=freq)
#         st.success("Monte Carlo hoàn tất.")
#         figmc = go.Figure()
#         figmc.add_trace(go.Scatter(x=mc_out['ds'], y=mc_out['mean'], name='MC mean'))
#         figmc.add_trace(go.Scatter(x=list(mc_out['ds']) + list(mc_out['ds'][::-1]),
#                                    y=list(mc_out['upper']) + list(mc_out['lower'][::-1]),
#                                    fill='toself', fillcolor='rgba(200,200,200,0.2)', line=dict(color='rgba(255,255,255,0)'),
#                                    name='90% CI'))
#         figmc.update_layout(title="Monte Carlo: mean & 90% CI", xaxis_title="Ngày", yaxis_title="Doanh thu")
#         st.plotly_chart(figmc, use_container_width=True)
#         st.markdown(f"- MC mean total revenue (summed mean series) = **{mc_out['mean'].sum():.0f}**")
#
#     # Downloads
#     merged_out = merged.rename(columns={'baseline':'baseline','scenario':'scenario'})
#     st.download_button("Tải CSV: So sánh lịch sử baseline vs scenario", merged_out.to_csv(index=False), file_name="simulate_historical_compare.csv", mime="text/csv")
#     if projected_df is not None:
#         st.download_button("Tải CSV: Projected forecast tail", projected_df.to_csv(index=False), file_name="simulate_projected_tail.csv", mime="text/csv")
#
#     st.success("Mô phỏng hoàn tất. Kiểm tra biểu đồ và tải dữ liệu nếu cần.")
#
#
# pages/Simulate.py
# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from components.data_loader import load_data
#
# st.set_page_config(page_title="Simulate & What-if", layout="wide")
# st.title("🔄 Simulate & What-if – Mô phỏng kịch bản kinh doanh")
#
# # Load data
# df = load_data()
# df = df.dropna(subset=['PRICEEACH', 'QUANTITYORDERED', 'MSRP', 'MONTH_ID'])
#
# # 1. MULTISELECT CHỌN NHIỀU PRODUCTLINE
# productlines = st.multiselect("Chọn PRODUCTLINE để mô phỏng", df['PRODUCTLINE'].dropna().unique(), default=["Classic Cars"])
# df_filtered = df[df['PRODUCTLINE'].isin(productlines)].copy()
#
# # 2. CHỌN CHẾ ĐỘ MÔ PHỎNG
# mode = st.radio("🔧 Chọn phương pháp mô phỏng", ["🔹 Rule-based", "🔸 Machine Learning (Linear Regression)"])
#
# # 3. THIẾT LẬP KỊCH BẢN
# st.subheader("⚙️ Thiết lập kịch bản")
# col1, col2, col3 = st.columns(3)
# with col1:
#     price_change_pct = st.slider("Thay đổi giá bán (%)", -50, 50, 0)
# with col2:
#     discount_change_pct = st.slider("Thay đổi chiết khấu (%)", -50, 50, 0)
# with col3:
#     use_custom_cost = st.checkbox("Tùy chỉnh tỷ lệ chi phí theo PRODUCTLINE", value=False)
#
# # Tạo bản sao dữ liệu mô phỏng
# df_sim = df_filtered.copy()
# df_sim['PRICEEACH_NEW'] = df_sim['PRICEEACH'] * (1 + price_change_pct / 100)
# df_sim['DISCOUNT_NEW'] = 0  # Nếu có dữ liệu chiết khấu thật, thay bằng logic tính toán
#
# # 3. TÙY CHỌN TỶ LỆ CHI PHÍ RIÊNG CHO PRODUCTLINE
# if use_custom_cost:
#     st.markdown("### 🧮 Nhập tỷ lệ chi phí cho từng PRODUCTLINE (% trên SALES)")
#     cost_map = {}
#     for pl in productlines:
#         cost_map[pl] = st.slider(f"{pl}", 30, 100, 70, key=f"cost_{pl}")
#     df_sim['COST_PCT'] = df_sim['PRODUCTLINE'].map(cost_map)
# else:
#     default_cost_pct = st.slider("Tỷ lệ chi phí (% trên SALES)", 30, 100, 70)
#     df_sim['COST_PCT'] = default_cost_pct
#
# # 4. MÔ HÌNH HỌC MÁY HOẶC RULE-BASED
# if mode == "🔸 Machine Learning (Linear Regression)":
#     df_encoded = pd.get_dummies(df_sim, columns=['PRODUCTLINE'], drop_first=True)
#     feature_cols = ['PRICEEACH_NEW', 'MSRP', 'MONTH_ID'] + [col for col in df_encoded.columns if col.startswith('PRODUCTLINE_')]
#     X = df_encoded[feature_cols]
#     y = df_encoded['QUANTITYORDERED']
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     model = LinearRegression()
#     model.fit(X_train, y_train)
#     df_sim['QUANTITYORDERED_NEW'] = model.predict(X[feature_cols])
# else:
#     df_sim['QUANTITYORDERED_NEW'] = df_sim['QUANTITYORDERED']
#
# # 5. TÍNH TOÁN MỚI
# df_sim['SALES_NEW'] = df_sim['PRICEEACH_NEW'] * df_sim['QUANTITYORDERED_NEW']
# df_sim['COST_NEW'] = df_sim['SALES_NEW'] * (df_sim['COST_PCT'] / 100)
# df_sim['PROFIT_NEW'] = df_sim['SALES_NEW'] - df_sim['COST_NEW']
#
# # 6. TỔNG HỢP KẾT QUẢ THEO PRODUCTLINE
# st.subheader("📊 Kết quả mô phỏng theo PRODUCTLINE")
#
# summary = []
#
# for pl in productlines:
#     df_orig_pl = df_filtered[df_filtered['PRODUCTLINE'] == pl]
#     df_sim_pl = df_sim[df_sim['PRODUCTLINE'] == pl]
#
#     orig_sales = df_orig_pl['SALES'].sum()
#     orig_profit = orig_sales - orig_sales * (df_sim_pl['COST_PCT'].iloc[0] / 100)
#     orig_qty = df_orig_pl['QUANTITYORDERED'].sum()
#
#     sim_sales = df_sim_pl['SALES_NEW'].sum()
#     sim_profit = df_sim_pl['PROFIT_NEW'].sum()
#     sim_qty = df_sim_pl['QUANTITYORDERED_NEW'].sum()
#
#     summary.append({
#         "PRODUCTLINE": pl,
#         "Doanh thu gốc": orig_sales,
#         "Doanh thu mô phỏng": sim_sales,
#         "Chênh lệch doanh thu": sim_sales - orig_sales,
#         "% thay đổi doanh thu": 100 * (sim_sales - orig_sales) / orig_sales,
#         "Lợi nhuận gốc": orig_profit,
#         "Lợi nhuận mô phỏng": sim_profit,
#         "Chênh lệch lợi nhuận": sim_profit - orig_profit,
#         "% thay đổi lợi nhuận": 100 * (sim_profit - orig_profit) / orig_profit,
#         "Số lượng gốc": orig_qty,
#         "Số lượng mô phỏng": sim_qty,
#         "Chênh lệch SL": sim_qty - orig_qty,
#         "% thay đổi SL": 100 * (sim_qty - orig_qty) / orig_qty if orig_qty else 0
#     })
#
# summary_df = pd.DataFrame(summary)
#
# st.dataframe(summary_df.style.format({
#     "Doanh thu gốc": "${:,.2f}",
#     "Doanh thu mô phỏng": "${:,.2f}",
#     "Chênh lệch doanh thu": "${:,.2f}",
#     "% thay đổi doanh thu": "{:.2f}%",
#     "Lợi nhuận gốc": "${:,.2f}",
#     "Lợi nhuận mô phỏng": "${:,.2f}",
#     "Chênh lệch lợi nhuận": "${:,.2f}",
#     "% thay đổi lợi nhuận": "{:.2f}%",
#     "Số lượng gốc": "{:,.0f}",
#     "Số lượng mô phỏng": "{:,.0f}",
#     "Chênh lệch SL": "{:,.0f}",
#     "% thay đổi SL": "{:.2f}%"
# }), use_container_width=True)
#
# # Biểu đồ tổng hợp
# st.subheader("📈 So sánh doanh thu & lợi nhuận mô phỏng")
# fig = go.Figure()
# fig.add_trace(go.Bar(name="Doanh thu gốc", x=summary_df['PRODUCTLINE'], y=summary_df['Doanh thu gốc']))
# fig.add_trace(go.Bar(name="Doanh thu mô phỏng", x=summary_df['PRODUCTLINE'], y=summary_df['Doanh thu mô phỏng']))
# fig.add_trace(go.Bar(name="Lợi nhuận mô phỏng", x=summary_df['PRODUCTLINE'], y=summary_df['Lợi nhuận mô phỏng']))
# fig.update_layout(barmode='group', xaxis_title="Product Line", yaxis_title="Giá trị ($)")
# st.plotly_chart(fig, use_container_width=True)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from components.data_loader import load_data
def app():
    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    local_css("style.css")
    st.set_page_config(page_title="Simulate & What-if", layout="wide")
    st.title("🔄 Simulate & What-if – Mô phỏng kịch bản kinh doanh")

    # Load data
    df = load_data()
    df = df.dropna(subset=['PRICEEACH', 'QUANTITYORDERED', 'MSRP', 'MONTH_ID'])

    # --- 1. Thiết lập kịch bản mô phỏng ---
    st.header("⚙️ 1. Thiết lập kịch bản")
    productlines = st.multiselect("Chọn PRODUCTLINE để mô phỏng", df['PRODUCTLINE'].dropna().unique(), default=["Classic Cars"])
    df_filtered = df[df['PRODUCTLINE'].isin(productlines)].copy()

    col1, col2, col3 = st.columns(3)
    with col1:
        price_change_pct = st.slider("Thay đổi giá bán (%)", -50, 50, 0)
    with col2:
        discount_change_pct = st.slider("Thay đổi chiết khấu (%)", -50, 50, 0)
    with col3:
        use_custom_cost = st.checkbox("Tùy chỉnh tỷ lệ chi phí theo PRODUCTLINE", value=False)

    # Tạo bản sao dữ liệu mô phỏng
    df_sim = df_filtered.copy()
    df_sim['PRICEEACH_NEW'] = df_sim['PRICEEACH'] * (1 + price_change_pct / 100) * (1 - discount_change_pct / 100)
    df_sim['DISCOUNT_NEW'] = discount_change_pct

    if use_custom_cost:
        st.markdown("### 🧮 Nhập tỷ lệ chi phí cho từng PRODUCTLINE (% trên SALES)")
        cost_map = {}
        for pl in productlines:
            cost_map[pl] = st.slider(f"{pl}", 30, 100, 70, key=f"cost_{pl}")
        df_sim['COST_PCT'] = df_sim['PRODUCTLINE'].map(cost_map)
    else:
        default_cost_pct = st.slider("Tỷ lệ chi phí (% trên SALES)", 30, 100, 70)
        df_sim['COST_PCT'] = default_cost_pct

    # --- 2. Cấu hình mô hình học máy ---
    st.header("🧠 2. Cấu hình mô hình học máy (tùy chọn)")
    mode = st.radio("🔧 Chọn phương pháp mô phỏng", ["🔹 Rule-based", "🔸 Machine Learning"], horizontal=True)

    if mode == "🔸 Machine Learning":
        ml_model_choice = st.selectbox("Chọn mô hình học máy", ["Linear Regression", "Random Forest", "XGBoost"])
        compare_models = st.checkbox("📊 So sánh các mô hình")

        df_encoded = pd.get_dummies(df_sim, columns=['PRODUCTLINE'], drop_first=True)
        feature_cols_all = ['PRICEEACH_NEW', 'MSRP', 'MONTH_ID'] + [col for col in df_encoded.columns if col.startswith('PRODUCTLINE_')]
        selected_features = st.multiselect("Chọn biến đầu vào (biến giải thích)", feature_cols_all, default=['PRICEEACH_NEW', 'MSRP', 'MONTH_ID'])

        X = df_encoded[selected_features]
        y = df_encoded['QUANTITYORDERED']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "XGBoost": XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
        }

        if compare_models:
            st.subheader("📊 So sánh các mô hình dự đoán")
            results = []
            for name, model in models.items():
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                results.append({
                    "Mô hình": name,
                    "R2 Score": r2_score(y_test, preds),
                    "MAE": mean_absolute_error(y_test, preds),
                    "RMSE": np.sqrt(mean_squared_error(y_test, preds))
                })

            results_df = pd.DataFrame(results)
            st.dataframe(results_df.style.format({"R2 Score": "{:.3f}", "MAE": "{:.2f}", "RMSE": "{:.2f}"}))

            st.markdown("### ✍️ Nhận xét về mô hình")
            best_model = results_df.sort_values(by="R2 Score", ascending=False).iloc[0]
            st.info(f"✅ Mô hình tốt nhất là **{best_model['Mô hình']}** với R2 = {best_model['R2 Score']:.3f}, RMSE = {best_model['RMSE']:.2f}")

        model = models[ml_model_choice]
        model.fit(X_train, y_train)
        df_sim['QUANTITYORDERED_NEW'] = model.predict(X[selected_features])
    else:
        df_sim['QUANTITYORDERED_NEW'] = df_sim['QUANTITYORDERED']

    # --- 3. Tính toán mô phỏng ---
    st.header("📊 3. Kết quả mô phỏng theo PRODUCTLINE")
    df_sim['SALES_NEW'] = df_sim['PRICEEACH_NEW'] * df_sim['QUANTITYORDERED_NEW']
    df_sim['COST_NEW'] = df_sim['SALES_NEW'] * (df_sim['COST_PCT'] / 100)
    df_sim['PROFIT_NEW'] = df_sim['SALES_NEW'] - df_sim['COST_NEW']

    summary = []

    for pl in productlines:
        df_orig_pl = df_filtered[df_filtered['PRODUCTLINE'] == pl]
        df_sim_pl = df_sim[df_sim['PRODUCTLINE'] == pl]

        orig_sales = df_orig_pl['SALES'].sum()
        orig_profit = orig_sales - orig_sales * (df_sim_pl['COST_PCT'].iloc[0] / 100)
        orig_qty = df_orig_pl['QUANTITYORDERED'].sum()

        sim_sales = df_sim_pl['SALES_NEW'].sum()
        sim_profit = df_sim_pl['PROFIT_NEW'].sum()
        sim_qty = df_sim_pl['QUANTITYORDERED_NEW'].sum()

        summary.append({
            "PRODUCTLINE": pl,
            "Doanh thu gốc": orig_sales,
            "Doanh thu mô phỏng": sim_sales,
            "Chênh lệch doanh thu": sim_sales - orig_sales,
            "% thay đổi doanh thu": 100 * (sim_sales - orig_sales) / orig_sales,
            "Lợi nhuận gốc": orig_profit,
            "Lợi nhuận mô phỏng": sim_profit,
            "Chênh lệch lợi nhuận": sim_profit - orig_profit,
            "% thay đổi lợi nhuận": 100 * (sim_profit - orig_profit) / orig_profit,
            "Số lượng gốc": orig_qty,
            "Số lượng mô phỏng": sim_qty,
            "Chênh lệch SL": sim_qty - orig_qty,
            "% thay đổi SL": 100 * (sim_qty - orig_qty) / orig_qty if orig_qty else 0
        })

    summary_df = pd.DataFrame(summary)

    st.dataframe(summary_df.style.format({
        "Doanh thu gốc": "${:,.2f}",
        "Doanh thu mô phỏng": "${:,.2f}",
        "Chênh lệch doanh thu": "${:,.2f}",
        "% thay đổi doanh thu": "{:.2f}%",
        "Lợi nhuận gốc": "${:,.2f}",
        "Lợi nhuận mô phỏng": "${:,.2f}",
        "Chênh lệch lợi nhuận": "${:,.2f}",
        "% thay đổi lợi nhuận": "{:.2f}%",
        "Số lượng gốc": "{:,.0f}",
        "Số lượng mô phỏng": "{:,.0f}",
        "Chênh lệch SL": "{:,.0f}",
        "% thay đổi SL": "{:.2f}%"
    }), use_container_width=True)

    # --- 4. Biểu đồ ---
    st.header("📈 4. So sánh doanh thu & lợi nhuận mô phỏng")
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Doanh thu gốc", x=summary_df['PRODUCTLINE'], y=summary_df['Doanh thu gốc']))
    fig.add_trace(go.Bar(name="Doanh thu mô phỏng", x=summary_df['PRODUCTLINE'], y=summary_df['Doanh thu mô phỏng']))
    fig.add_trace(go.Bar(name="Lợi nhuận mô phỏng", x=summary_df['PRODUCTLINE'], y=summary_df['Lợi nhuận mô phỏng']))
    fig.update_layout(barmode='group', xaxis_title="Product Line", yaxis_title="Giá trị ($)")
    st.plotly_chart(fig, use_container_width=True)

