# pages/Optimize.py
"""
Optimize.py — Product Portfolio Optimization (Revenue & Profit)
- Pareto / Top-N selection by chosen metric
- Auto-fill COST (AVG_PRICE * 0.5) and manual edit via st.data_editor
- Breakdown charts compare Baseline vs Selected-set (if requested)
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import pyplot as plt
import pulp
from pulp import LpProblem, LpVariable, lpSum, LpMinimize, LpBinary, PULP_CBC_CMD
def app():
        # optional data loader from components
        try:
            from components.data_loader import load_data
        except Exception:
            load_data = None
        def local_css(file_name):
            with open(file_name, "r", encoding="utf-8") as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        local_css("style.css")
        # page config (should be one of the first streamlit calls)
        st.set_page_config(page_title="04_Optimize — Product Portfolio", layout="wide")
        st.title("🧩 Tối ưu hóa danh mục sản phẩm — Revenue & Profit")

        # -------------------------
        # Helpers
        # -------------------------
        # Tạo các tab con Optimize
        tab1, tab2, tab3 = st.tabs(["Doanh thu & Lợi nhuận", "Chi phí", "Vận chuyển"])
        with tab1:
            @st.cache_data
            def load_df(path: str = "cleaned_sales_data_final.csv") -> pd.DataFrame:
                if load_data is not None:
                    try:
                        return load_data(path)
                    except Exception:
                        pass
                df0 = pd.read_csv(path, encoding='ISO-8859-1', low_memory=False)
                df0.columns = df0.columns.str.upper().str.strip()
                if 'ORDERDATE' in df0.columns and not pd.api.types.is_datetime64_any_dtype(df0['ORDERDATE']):
                    df0['ORDERDATE'] = pd.to_datetime(df0['ORDERDATE'], errors='coerce', dayfirst=True)
                return df0
            def summarize_products(df: pd.DataFrame, product_col: str) -> pd.DataFrame:
                # ensure numeric
                for c in ['SALES','QUANTITYORDERED','PRICEEACH','MSRP']:
                    if c in df.columns:
                        df[c] = pd.to_numeric(df[c], errors='coerce')
                agg = {}
                if 'SALES' in df.columns:
                    agg['REVENUE'] = ('SALES','sum')
                else:
                    # fallback: sum PRICEEACH (not ideal but safe)
                    agg['REVENUE'] = ('PRICEEACH','sum')
                if 'QUANTITYORDERED' in df.columns:
                    agg['QTY'] = ('QUANTITYORDERED','sum')
                else:
                    agg['QTY'] = ('REVENUE','count')
                if 'PRICEEACH' in df.columns:
                    agg['AVG_PRICE'] = ('PRICEEACH','mean')
                else:
                    agg['AVG_PRICE'] = ('REVENUE','mean')
                if 'ORDERNUMBER' in df.columns:
                    agg['ORDERS'] = ('ORDERNUMBER','nunique')
                else:
                    agg['ORDERS'] = ('REVENUE','count')
                grp = df.groupby(product_col).agg(**agg).reset_index().rename(columns={product_col:'PRODUCT'})
                grp = grp.fillna({'REVENUE':0.0,'QTY':0.0,'AVG_PRICE':0.0,'ORDERS':0})
                grp = grp.sort_values('REVENUE', ascending=False).reset_index(drop=True)
                total_rev = grp['REVENUE'].sum() if grp['REVENUE'].sum() != 0 else 1.0
                grp['REV_SHARE'] = 100.0 * grp['REVENUE'] / total_rev
                grp['CUM_REVENUE'] = grp['REVENUE'].cumsum()
                grp['CUM_PERC'] = 100.0 * grp['CUM_REVENUE'] / total_rev
                return grp

            def greedy_select(df_items: pd.DataFrame, value_col: str, cost_col: str, budget: float) -> pd.DataFrame:
                df = df_items.copy()
                if cost_col not in df.columns or value_col not in df.columns:
                    return pd.DataFrame(columns=list(df.columns)+['ratio'])
                df = df[df[cost_col] > 0].copy()
                if df.empty:
                    return pd.DataFrame(columns=list(df.columns)+['ratio'])
                df['ratio'] = df[value_col] / df[cost_col]
                df = df.sort_values('ratio', ascending=False)
                selected = []
                total_cost = 0.0
                for _, r in df.iterrows():
                    if total_cost + float(r[cost_col]) <= budget:
                        selected.append(r)
                        total_cost += float(r[cost_col])
                if not selected:
                    return pd.DataFrame(columns=list(df.columns)+['ratio'])
                return pd.DataFrame(selected).reset_index(drop=True)

            df = load_df()
            df.columns = df.columns.str.upper().str.strip()
            # initialize session-state objects
            if 'selected_df' not in st.session_state:
                st.session_state['selected_df'] = pd.DataFrame()
            if 'product_cost_overrides' not in st.session_state:
                st.session_state['product_cost_overrides'] = {}
            # choose product column
            product_candidates = [c for c in ['PRODUCTLINE','PRODUCTNAME','PRODUCT','ITEM','ITEMNAME'] if c in df.columns]
            if not product_candidates:
                st.error("Không tìm thấy cột sản phẩm (PRODUCTLINE/PRODUCTNAME/...). Kiểm tra file.")
                st.stop()
            product_col = product_candidates[0]

            with st.expander("📂 Filters", expanded=True):
                metric_choice = st.selectbox("Metric to analyze / Pareto", ["Revenue", "Profit"])

                # --- Bộ lọc năm ---
                years = sorted(df['YEAR_ID'].dropna().unique()) if 'YEAR_ID' in df.columns else []
                year_options = ["Tất cả các năm"] + [str(int(y)) for y in years]  # Thêm tùy chọn tất cả các năm
                selected_year = None
                if years:
                    selected_year = st.selectbox("Select Year", options=year_options, index=0)  # mặc định là "Tất cả các năm"
                # --- Bộ lọc quốc gia ---
                countries = sorted(df['COUNTRY'].dropna().unique()) if 'COUNTRY' in df.columns else []
                selected_countries = st.multiselect("Country (filter)", options=countries, default=countries)
                # --- Bộ lọc DealSize ---
                deals = sorted(df['DEALSIZE'].dropna().unique()) if 'DEALSIZE' in df.columns else []
                selected_deals = st.multiselect("DealSize (filter)", options=deals, default=deals)
            # ===== Apply filters =====
            df_f = df.copy()
            # Lọc theo năm nếu không chọn "Tất cả các năm"
            if selected_year != "Tất cả các năm" and selected_year is not None and 'YEAR_ID' in df_f.columns:
                df_f = df_f[df_f['YEAR_ID'] == int(selected_year)]
            # Lọc quốc gia và DealSize
            if selected_countries and 'COUNTRY' in df_f.columns:
                df_f = df_f[df_f['COUNTRY'].isin(selected_countries)]
            if selected_deals and 'DEALSIZE' in df_f.columns:
                df_f = df_f[df_f['DEALSIZE'].isin(selected_deals)]

            st.markdown(f"**Records after filters:** {len(df_f):,}")

            summary = summarize_products(df_f, product_col=product_col)
            # derive COST mapping if raw df has COST
            if 'COST' in df_f.columns:
                try:
                    cost_map = df_f.groupby(product_col)['COST'].mean().to_dict()
                except Exception:
                    cost_map = {}
            else:
                cost_map = {}

            summary['COST'] = summary['PRODUCT'].map(lambda p: cost_map.get(p, np.nan))

            # Auto-fill missing COST = AVG_PRICE * 0.5
            summary['COST'] = summary.apply(lambda r: (r['AVG_PRICE'] * 0.5) if (pd.isna(r['COST']) or r['COST']==0) else r['COST'], axis=1)

            # apply any overrides from session
            for p,v in st.session_state['product_cost_overrides'].items():
                summary.loc[summary['PRODUCT']==p, 'COST'] = v
            # compute PROFIT = (AVG_PRICE - COST) * QTY
            # ensure numeric
            for c in ['AVG_PRICE','COST','QTY']:
                if c in summary.columns:
                    summary[c] = pd.to_numeric(summary[c], errors='coerce').fillna(0)
            summary['PROFIT'] = (summary['AVG_PRICE'] - summary['COST']) * summary['QTY']
            # share columns
            total_rev = summary['REVENUE'].sum() if summary['REVENUE'].sum()!=0 else 1.0
            total_profit = summary['PROFIT'].sum() if summary['PROFIT'].sum()!=0 else 1.0
            summary['REV_SHARE'] = 100.0 * summary['REVENUE'] / total_rev
            summary['PROFIT_SHARE'] = 100.0 * summary['PROFIT'] / total_profit
            summary = summary.sort_values('REVENUE', ascending=False).reset_index(drop=True)

            left, right = st.columns([2,1])

            with right:
                st.subheader("Selection (Rule-based)")
                sel_mode = st.selectbox("Rule", ["Top N by metric", "Pareto cumulative %"])
                if sel_mode == "Top N by metric":
                    default_top = min(10, len(summary)) if len(summary)>0 else 1
                    top_n = st.number_input("Top N", min_value=1, max_value=max(1,len(summary)), value=default_top, step=1)
                else:
                    default_pct = 80
                    pct = st.slider("Pareto cumulative threshold (%)", min_value=10, max_value=100, value=default_pct, step=5)

                st.markdown("**Edit COST** (click cell to change). After editing, press 'Save COST' to apply.")
                # ensure 'COST' & 'PRODUCT' exist
                editable = summary[['PRODUCT','COST']].set_index('PRODUCT') if 'PRODUCT' in summary.columns and 'COST' in summary.columns else pd.DataFrame()
                edited = st.data_editor(editable, num_rows="dynamic", use_container_width=True) if not editable.empty else pd.DataFrame()

                if st.button("Save COST"):
                    try:
                        ed = edited.reset_index() if not edited.empty else pd.DataFrame()
                        if not ed.empty:
                            # normalize column names: expect ['PRODUCT','COST']
                            if 'PRODUCT' not in ed.columns and ed.columns[0] == 'index':
                                ed = ed.rename(columns={'index':'PRODUCT'})
                            for _, r in ed.iterrows():
                                prod = r.get('PRODUCT', None)
                                val = r.get('COST', None)
                                try:
                                    valf = float(val) if pd.notna(val) else np.nan
                                except Exception:
                                    valf = np.nan
                                if prod is not None:
                                    st.session_state['product_cost_overrides'][prod] = valf
                            # re-apply overrides to summary
                            for p, v in st.session_state['product_cost_overrides'].items():
                                summary.loc[summary['PRODUCT']==p, 'COST'] = v
                            # recompute profit/shares
                            summary['PROFIT'] = (summary['AVG_PRICE'] - summary['COST']) * summary['QTY']
                            total_profit = summary['PROFIT'].sum() if summary['PROFIT'].sum() != 0 else 1.0
                            summary['PROFIT_SHARE'] = 100.0 * summary['PROFIT'] / total_profit
                            st.success("Saved COST to session and updated Profit.")
                        else:
                            st.info("No edits to save.")
                    except Exception as e:
                        st.error(f"Error saving COST edits: {e}")

                if st.button("Download COST mapping"):
                    cm = summary[['PRODUCT','COST']].copy()
                    st.download_button("Download COST CSV", data=cm.to_csv(index=False).encode('utf-8'), file_name='product_cost_mapping.csv', mime='text/csv')

                st.markdown("---")
                st.subheader("▶ Run selection")
                if st.button("Run selection"):
                    # compute selection and store to session
                    if sel_mode == "Top N by metric":
                        if metric_choice == "Revenue":
                            st.session_state['selected_df'] = summary.sort_values('REVENUE', ascending=False).head(top_n).copy()
                        else:
                            st.session_state['selected_df'] = summary.sort_values('PROFIT', ascending=False).head(top_n).copy()
                    else:
                        if metric_choice == "Revenue":
                            tmp = summary.sort_values('REVENUE', ascending=False).reset_index(drop=True)
                            tmp['CUM_PERC_METRIC'] = 100.0 * tmp['REVENUE'].cumsum() / max(tmp['REVENUE'].sum(), 1.0)
                            st.session_state['selected_df'] = tmp[tmp['CUM_PERC_METRIC'] <= pct].copy()
                        else:
                            tmp = summary.sort_values('PROFIT', ascending=False).reset_index(drop=True)
                            tmp['CUM_PERC_METRIC'] = 100.0 * tmp['PROFIT'].cumsum() / max(tmp['PROFIT'].sum(), 1.0)
                            st.session_state['selected_df'] = tmp[tmp['CUM_PERC_METRIC'] <= pct].copy()
                    st.success("Selection updated.")

                if st.button("Clear selection"):
                    st.session_state['selected_df'] = pd.DataFrame()
                    st.info("Selection cleared.")

            with left:
                st.subheader("📑 Selection result (rule-based)")
                sel_df = st.session_state.get('selected_df', pd.DataFrame())
                if sel_df is None or sel_df.empty:
                    st.info("No products selected yet. Press **Run selection** to see results.")
                else:
                    metric_col = 'REVENUE' if metric_choice == 'Revenue' else 'PROFIT'
                    sel_count = len(sel_df)
                    sel_metric_sum = sel_df[metric_col].sum() if metric_col in sel_df.columns else 0.0
                    overall_metric_sum = max(summary[metric_col].sum(), 1.0) if metric_col in summary.columns else 1.0
                    sel_pct = 100.0 * sel_metric_sum / overall_metric_sum

                    st.markdown(f"**Rule:** {sel_mode} — **Metric:** {metric_choice}")
                    st.markdown(f"- Selected products: **{sel_count}**")
                    st.markdown(f"- Sum {metric_choice}: **{sel_metric_sum:,.2f}** which is **{sel_pct:.2f}%** of total {metric_choice}.")

                    show_cols = ['PRODUCT', 'REVENUE', 'PROFIT', 'QTY', 'AVG_PRICE', 'COST']
                    show_cols = [c for c in show_cols if c in sel_df.columns]
                    try:
                        st.dataframe(
                            sel_df[show_cols].style.format({
                                'REVENUE': '{:,.2f}',
                                'PROFIT': '{:,.2f}',
                                'COST': '{:,.2f}',
                                'AVG_PRICE': '{:,.2f}'
                            }), use_container_width=True
                        )
                    except Exception:
                        # fallback plain dataframe
                        st.dataframe(sel_df[show_cols], use_container_width=True)

                    st.download_button(
                        "📥 Download selected CSV",
                        data=sel_df.to_csv(index=False).encode('utf-8'),
                        file_name='selected_products.csv',
                        mime='text/csv'
                    )
            st.markdown("---")
            st.subheader("Pareto & Treemap (by selected metric)")

            if metric_choice == "Revenue":
                pareto_df = summary.sort_values('REVENUE', ascending=False).reset_index(drop=True)
                pareto_df['CUM_PERC_METRIC'] = 100.0 * pareto_df['REVENUE'].cumsum() / (pareto_df['REVENUE'].sum() if pareto_df['REVENUE'].sum()!=0 else 1.0)
                y_col = 'REVENUE'
            else:
                pareto_df = summary.sort_values('PROFIT', ascending=False).reset_index(drop=True)
                pareto_df['CUM_PERC_METRIC'] = 100.0 * pareto_df['PROFIT'].cumsum() / (pareto_df['PROFIT'].sum() if pareto_df['PROFIT'].sum()!=0 else 1.0)
                y_col = 'PROFIT'
            try:
                fig = go.Figure()
                fig.add_trace(go.Bar(x=pareto_df['PRODUCT'], y=pareto_df[y_col], name=y_col))
                fig.add_trace(go.Scatter(x=pareto_df['PRODUCT'], y=pareto_df['CUM_PERC_METRIC'],
                                         name='Cumulative %', yaxis='y2', mode='lines+markers'))
                fig.update_layout(xaxis_tickangle=-45, yaxis=dict(title=y_col),
                                  yaxis2=dict(title='Cumulative %', overlaying='y', side='right', range=[0,100]), height=480)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("""
                📌 **Biểu đồ Pareto** cho thấy sản phẩm nào đóng góp nhiều nhất vào doanh thu/lợi nhuận. 
                Đường màu xanh thể hiện phần trăm tích lũy. Ví dụ: 20% sản phẩm đầu tiên có thể tạo ra tới 80% doanh thu.
                """)
            except Exception as e:
                st.info("Cannot draw Pareto chart: " + str(e))
            top_n_treem = min(50, len(pareto_df))
            if top_n_treem < 1:
                top_n_treem = 1
            top_n_treem = st.slider("Top N for treemap", min_value=1, max_value=max(1,top_n_treem), value=min(15, top_n_treem))
            try:
                treedf = pareto_df.head(top_n_treem).copy()
                color_col = 'REV_SHARE' if metric_choice=='Revenue' else 'PROFIT_SHARE'
                fig_t = px.treemap(treedf, path=['PRODUCT'], values=y_col, color=color_col,
                                   color_continuous_scale='Blues', title=f"Treemap (Top {top_n_treem} by {metric_choice})")
                st.plotly_chart(fig_t, use_container_width=True)
            except Exception as e:
                st.info("Cannot draw treemap: " + str(e))
            st.markdown("""
            📌 **Treemap** là bản đồ khối thể hiện quy mô doanh thu/lợi nhuận từng sản phẩm. 
            Sản phẩm càng lớn, màu càng đậm thì đóng góp càng cao.
            """)

            st.markdown("---")
            st.subheader("Breakdowns: compare metric across groups")

            available_groups = [c for c in ['PRODUCTLINE','COUNTRY','DEALSIZE'] if c in df_f.columns]
            if not available_groups:
                st.info("No grouping columns (PRODUCTLINE/COUNTRY/DEALSIZE) available in data.")
            else:
                group_by = st.selectbox("Compare by", options=available_groups, index=0)
                compare_with_selected = st.checkbox("Compare baseline (all filtered data) vs selected set", value=True)

                # Baseline metric aggregation (safe handling)
                if metric_choice == "Revenue":
                    if 'SALES' in df_f.columns and df_f['SALES'].notna().any():
                        baseline_grp = df_f.groupby(group_by)['SALES'].sum().reset_index().rename(columns={'SALES':'METRIC'})
                    elif 'PRICEEACH' in df_f.columns and 'QUANTITYORDERED' in df_f.columns:
                        df_f['__SALES__'] = pd.to_numeric(df_f['PRICEEACH'], errors='coerce').fillna(0) * pd.to_numeric(df_f['QUANTITYORDERED'], errors='coerce').fillna(0)
                        baseline_grp = df_f.groupby(group_by)['__SALES__'].sum().reset_index().rename(columns={'__SALES__':'METRIC'})
                    else:
                        # fallback to counts
                        baseline_grp = df_f.groupby(group_by).size().reset_index().rename(columns={0:'METRIC'})
                else:
                    # Profit baseline: compute record-level profit using product-level cost mapping
                    prod_cost = dict(zip(summary['PRODUCT'], summary['COST']))
                    rec = df_f.copy()
                    rec_prod_col = product_col if product_col in rec.columns else None
                    if rec_prod_col is None:
                        # create a dummy product column to avoid crash
                        rec['__PRODUCT_KEY__'] = 'ALL'
                    else:
                        rec['__PRODUCT_KEY__'] = rec[rec_prod_col].astype(str)
                    rec['PRICEEACH'] = pd.to_numeric(rec['PRICEEACH'], errors='coerce').fillna(0) if 'PRICEEACH' in rec.columns else 0
                    rec['QUANTITYORDERED'] = pd.to_numeric(rec['QUANTITYORDERED'], errors='coerce').fillna(0) if 'QUANTITYORDERED' in rec.columns else 0
                    rec['__COST__'] = rec['__PRODUCT_KEY__'].map(lambda x: prod_cost.get(x, np.nan))
                    rec['__COST__'] = rec.apply(lambda r: (r['PRICEEACH'] * 0.5) if (pd.isna(r['__COST__']) or r['__COST__']==0) else r['__COST__'], axis=1)
                    rec['RECORD_PROFIT'] = (rec['PRICEEACH'] - rec['__COST__']) * rec['QUANTITYORDERED']
                    if group_by in rec.columns:
                        baseline_grp = rec.groupby(group_by)['RECORD_PROFIT'].sum().reset_index().rename(columns={'RECORD_PROFIT':'METRIC'})
                    else:
                        baseline_grp = rec.groupby(rec.index)['RECORD_PROFIT'].sum().reset_index().rename(columns={'RECORD_PROFIT':'METRIC'})
                baseline_grp = baseline_grp.sort_values('METRIC', ascending=False).reset_index(drop=True)
                baseline_top = baseline_grp.head(30)
                # Compare with selected set if requested and selection exists
                sel_df = st.session_state.get('selected_df', pd.DataFrame())
                if compare_with_selected and (sel_df is not None) and (not sel_df.empty):
                    selected_products = sel_df['PRODUCT'].astype(str).tolist() if 'PRODUCT' in sel_df.columns else []
                    # compute selected-set grouped metric
                    rec_sel = df_f.copy()
                    if product_col not in rec_sel.columns:
                        rec_sel['__PRODUCT_KEY__'] = 'ALL'
                    else:
                        rec_sel['__PRODUCT_KEY__'] = rec_sel[product_col].astype(str)
                    rec_sel = rec_sel[rec_sel['__PRODUCT_KEY__'].isin(selected_products)].copy()
                    if rec_sel.empty:
                        sel_grp = pd.DataFrame(columns=[group_by, 'METRIC'])
                    else:
                        if metric_choice == "Revenue":
                            if 'SALES' in rec_sel.columns and rec_sel['SALES'].notna().any():
                                sel_grp = rec_sel.groupby(group_by)['SALES'].sum().reset_index().rename(columns={'SALES':'METRIC'})
                            elif 'PRICEEACH' in rec_sel.columns and 'QUANTITYORDERED' in rec_sel.columns:
                                rec_sel['__SALES__'] = pd.to_numeric(rec_sel['PRICEEACH'], errors='coerce').fillna(0) * pd.to_numeric(rec_sel['QUANTITYORDERED'], errors='coerce').fillna(0)
                                sel_grp = rec_sel.groupby(group_by)['__SALES__'].sum().reset_index().rename(columns={'__SALES__':'METRIC'})
                            else:
                                sel_grp = rec_sel.groupby(group_by).size().reset_index().rename(columns={0:'METRIC'})
                        else:
                            prod_cost = dict(zip(summary['PRODUCT'], summary['COST']))
                            rec_sel['__COST__'] = rec_sel['__PRODUCT_KEY__'].map(lambda x: prod_cost.get(x, np.nan))
                            rec_sel['PRICEEACH'] = pd.to_numeric(rec_sel['PRICEEACH'], errors='coerce').fillna(0) if 'PRICEEACH' in rec_sel.columns else 0
                            rec_sel['QUANTITYORDERED'] = pd.to_numeric(rec_sel['QUANTITYORDERED'], errors='coerce').fillna(0) if 'QUANTITYORDERED' in rec_sel.columns else 0
                            rec_sel['__COST__'] = rec_sel.apply(lambda r: (r['PRICEEACH'] * 0.5) if (pd.isna(r['__COST__']) or r['__COST__']==0) else r['__COST__'], axis=1)
                            rec_sel['RECORD_PROFIT'] = (rec_sel['PRICEEACH'] - rec_sel['__COST__']) * rec_sel['QUANTITYORDERED']
                            sel_grp = rec_sel.groupby(group_by)['RECORD_PROFIT'].sum().reset_index().rename(columns={'RECORD_PROFIT':'METRIC'})
                    # merge baseline and selection for comparison
                    cmp = baseline_grp.merge(sel_grp, on=group_by, how='left', suffixes=('_BASE','_SEL')).fillna(0)
                    cmp = cmp.sort_values('METRIC_BASE', ascending=False).head(30)
                    try:
                        fig_cmp = go.Figure()
                        fig_cmp.add_trace(go.Bar(x=cmp[group_by], y=cmp['METRIC_BASE'], name='Baseline (all filtered)'))
                        fig_cmp.add_trace(go.Bar(x=cmp[group_by], y=cmp['METRIC_SEL'], name='Selected products'))
                        fig_cmp.update_layout(barmode='group', xaxis_tickangle=-45, title=f"Compare {metric_choice}: Baseline vs Selected (by {group_by})", height=480)
                        st.plotly_chart(fig_cmp, use_container_width=True)
                    except Exception as e:
                        st.info("Cannot draw comparison chart: " + str(e))
                else:
                    # just show baseline
                    try:
                        fig_base = px.bar(baseline_top, x=group_by, y='METRIC', title=f"{metric_choice} by {group_by} (baseline)", labels={'METRIC':metric_choice})
                        st.plotly_chart(fig_base, use_container_width=True)
                    except Exception as e:
                        st.info("Cannot draw baseline bar chart: " + str(e))
                st.markdown("""
                📌 **Biểu đồ so sánh nhóm** giúp doanh nghiệp biết nhóm khách hàng / sản phẩm nào đang mang lại hiệu quả cao nhất. 
                Từ đó có thể tái phân bổ ngân sách hoặc ưu tiên bán hàng.
                """)

            st.markdown("---")
            st.markdown(
                """
                <div class="recommendation-box">
                    <h3>Khuyến nghị:</h3>
                    <ul>
                        <li>Nếu mục tiêu là <b>Doanh thu</b>: dùng Pareto theo Revenue để lấy tập sản phẩm chiếm ~80% doanh thu — tập trung marketing & tồn kho cho chúng.</li>
                        <li>Nếu mục tiêu là <b>Lợi nhuận</b>: chuyển sang metric = Profit. Các sản phẩm top theo Profit có thể khác top theo Revenue (vì COST khác nhau).</li>
                        <li>Có thể <b>sửa COST thực tế</b> (nếu có) để kết quả profit chính xác.</li>
                        <li>Dùng tính năng <b>Compare Baseline vs Selected</b> để xem việc tập trung vào selected set thay đổi phân bố doanh thu/lợi nhuận theo Country/DealSize như thế nào.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )

        with tab2:
            st.subheader("🚚 So sánh tối ưu chi phí: 1 kho vs 3 kho")

            df = load_data()
            df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'])

            # Hàm tối ưu chi phí
            def optimize_shipping_cost(df, warehouses, capacity):
                options = []
                for _, row in df.iterrows():
                    for wh, wh_loc in warehouses.items():
                        distance = np.sqrt((row['cust_x'] - wh_loc[0]) ** 2 + (row['cust_y'] - wh_loc[1]) ** 2)
                        for day_offset in range(3):
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

                # MILP model
                orders = cost_df[['order_id', 'order_line']].drop_duplicates().to_records(index=False)
                x = LpVariable.dicts("x", cost_dict.keys(), cat=LpBinary)
                model = LpProblem("Minimize_Shipping_Cost", LpMinimize)
                model += lpSum(cost_dict[k] * x[k] for k in cost_dict.keys())

                # Ràng buộc: mỗi order line 1 phương án
                for oid, line in orders:
                    model += lpSum(
                        x[(oid, line, wh, date)]
                        for (oi, li, wh, date) in cost_dict.keys()
                        if oi == oid and li == line
                    ) == 1

                # Ràng buộc capacity
                for wh in warehouses.keys():
                    for d in cost_df['delivery_date'].unique():
                        model += lpSum(
                            x[(oid, line, w, dd)]
                            for (oid, line, w, dd) in cost_dict.keys()
                            if w == wh and dd == d
                        ) <= capacity

                model.solve(PULP_CBC_CMD(msg=False))

                # Lấy kết quả
                assignments = [
                    (oid, line, wh, date, cost_dict[(oid, line, wh, date)])
                    for (oid, line, wh, date) in cost_dict.keys()
                    if pulp.value(x[(oid, line, wh, date)]) > 0.5
                ]

                result_df = pd.DataFrame(assignments, columns=['order_id', 'order_line', 'warehouse', 'delivery_date',
                                                               'shipping_cost'])
                return result_df

            #

            # 1 kho vs 3 kho
            warehouses_1 = {"WH_ONLY": (0, 0)}
            warehouses_3 = {"WH_A": (0, 0), "WH_B": (50, 0), "WH_C": (25, 43)}
            '''
            res1= optimize_shipping_cost(df, warehouses_1, capacity=18)
            res3 = optimize_shipping_cost(df, warehouses_3, capacity=13)
            '''
            res1 = pd.read_csv("shipping_result_1kho.csv")
            res3 = pd.read_csv("shipping_result_3kho.csv")

            # Plot tổng chi phí
            total_costs = [res1['shipping_cost'].sum(), res3['shipping_cost'].sum()]
            fig1 = px.bar(
                x=["1 kho", "3 kho"],
                y=total_costs,
                labels={'x': "Kịch bản", 'y': "Tổng chi phí"},
                text=total_costs
            )
            st.plotly_chart(fig1, use_container_width=True)

            # Histogram chi phí từng đơn
            fig2 = go.Figure()
            fig2.add_trace(go.Histogram(x=res1['shipping_cost'], name="1 kho", opacity=0.6))
            fig2.add_trace(go.Histogram(x=res3['shipping_cost'], name="3 kho", opacity=0.6))
            fig2.update_layout(
                barmode='overlay',
                xaxis_title="Chi phí đơn hàng",
                yaxis_title="Số lượng",
                title="Phân phối chi phí từng đơn"
            )
            st.plotly_chart(fig2, use_container_width=True)
            # --- Kịch bản 1 kho ---
            warehouses_1 = {"WH_ONLY": (0, 0)}
            total_cost1 = res1['shipping_cost'].sum()
            st.subheader("📦 Kết quả kịch bản 1 kho")
            st.dataframe(res1.head(20))
            st.success(f"Tổng chi phí: {total_cost1:,.2f}")

            # --- Kịch bản 3 kho ---
            warehouses_3 = {"WH_A": (0, 0), "WH_B": (50, 0), "WH_C": (25, 43)}
            total_cost3 = res3['shipping_cost'].sum()
            st.subheader("📦 Kết quả kịch bản 3 kho")
            st.dataframe(res3.head(20))
            st.success(f"Tổng chi phí: {total_cost3:,.2f}")
            df_raw = pd.read_csv("cleaned_sales_data_final.csv", encoding='ISO-8859-1')

            # Xuất res1 đúng thứ tự gốc
            '''
            res1_export = df_raw[['ORDERNUMBER', 'ORDERLINENUMBER']].merge(
                res1,
                left_on=['ORDERNUMBER', 'ORDERLINENUMBER'],
                right_on=['order_id', 'order_line'],
                how='left'
            )

            res1_export.to_csv("shipping_result_1kho.csv", index=False)

            # Xuất res3 đúng thứ tự gốc
            res3_export = df_raw[['ORDERNUMBER', 'ORDERLINENUMBER']].merge(
                res3,
                left_on=['ORDERNUMBER', 'ORDERLINENUMBER'],
                right_on=['order_id', 'order_line'],
                how='left'
            )

            res3_export.to_csv("shipping_result_3kho.csv", index=False)
            '''
        with tab3:
            st.subheader("🚚 So sánh mô phỏng giao hàng ")
            # --- Load dữ liệu ---
            df = load_data()
            df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'])
            df = df.sort_values('ORDERDATE').reset_index(drop=True)
            st.write("📦 Tổng số đơn:", len(df))

            # --- Tham số tối ưu ---
            capacity = 18
            min_delay_days = 3
            max_extend_days = 7

            if st.button("▶️ Chạy mô phỏng tối ưu vận chuyển"):
                st.info("⏳ Đang khởi tạo và giải mô hình tối ưu...")

                # Khởi tạo mô hình LP
                model = LpProblem("Simulate_Shipping", LpMinimize)
                assign = {}

                for i in df.index:
                    start_date = df.loc[i, 'ORDERDATE'] + pd.Timedelta(days=min_delay_days)
                    end_date = df.loc[i, 'ORDERDATE'] + pd.Timedelta(days=max_extend_days)
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
                model.solve(PULP_CBC_CMD(msg=False, timeLimit=60))
                st.success("✅ Giải xong!")

                # Lấy kết quả SIM_SHIP_DATE sau tối ưu
                sim_ship_dates = []
                for i in df.index:
                    ship_date = None
                    start_date = df.loc[i, 'ORDERDATE'] + pd.Timedelta(days=min_delay_days)
                    end_date = df.loc[i, 'ORDERDATE'] + pd.Timedelta(days=max_extend_days)
                    for d in pd.date_range(start_date, end_date):
                        if assign[(i, d)].value() == 1:
                            ship_date = d
                            break
                    sim_ship_dates.append(ship_date)

                df["SIM_SHIP_DATE_CAP18"] = sim_ship_dates
                df["SHIP_DAYS_CAP18"] = (df["SIM_SHIP_DATE_CAP18"] - df["ORDERDATE"]).dt.days

                # Hiển thị bảng dữ liệu so sánh
                st.subheader("🔍 Xem trước dữ liệu")
                st.dataframe(df[[
                    "ORDERNUMBER", "ORDERDATE", "SIM_SHIP_DATE", "SIM_SHIP_DATE_CAP18",
                    "SHIP_DAYS_CAP_OLD", "SHIP_DAYS_CAP18"
                ]].head(20))

                # --- Thống kê số lượng đơn theo số ngày giao ---
                st.subheader("📋 Thống kê số lượng đơn theo số ngày giao hàng")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Gốc**")
                    df_old_counts = df['SHIP_DAYS_CAP_OLD'].value_counts().sort_index().reset_index()
                    df_old_counts.columns = ['Số ngày giao hàng', 'Số đơn']
                    fig_old = px.bar(
                        df_old_counts,
                        x='Số ngày giao hàng',
                        y='Số đơn',
                        title="Phân phối số ngày giao hàng - Gốc",
                        text='Số đơn'
                    )
                    st.plotly_chart(fig_old, use_container_width=True)

                with col2:
                    st.markdown("**Capacity=18**")
                    df_new_counts = df['SHIP_DAYS_CAP18'].value_counts().sort_index().reset_index()
                    df_new_counts.columns = ['Số ngày giao hàng', 'Số đơn']
                    fig_new = px.bar(
                        df_new_counts,
                        x='Số ngày giao hàng',
                        y='Số đơn',
                        title="Phân phối số ngày giao hàng - Sau tối ưu",
                        text='Số đơn'
                    )
                    st.plotly_chart(fig_new, use_container_width=True)

                # --- Biểu đồ histogram so sánh ---
                st.subheader("📊 So sánh phân phối số ngày giao hàng")
                hist_df = pd.DataFrame({
                    "Gốc": df["SHIP_DAYS_CAP_OLD"],
                    "Capacity=18": df["SHIP_DAYS_CAP18"]
                })
                fig_hist = px.histogram(
                    hist_df.melt(value_vars=["Gốc", "Capacity=18"], var_name="Loại", value_name="Số ngày giao hàng"),
                    x="Số ngày giao hàng", color="Loại", barmode="overlay",
                    title="So sánh số ngày giao hàng: Gốc vs Capacity=18"
                )
                st.plotly_chart(fig_hist, use_container_width=True)
