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
def app():
    # def local_css(file_name):
    #     with open(file_name) as f:
    #         st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    # local_css("style.css")
# try import data_loader
    try:
        from components.data_loader import load_data
    except Exception:
        load_data = None

    st.set_page_config(page_title="04_Optimize — Product Portfolio", layout="wide")
    st.title("🧩 Product Portfolio Optimization — Revenue & Profit")

    # -------------------------
    # Helpers
    # -------------------------
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
        df = df[df[cost_col] > 0].copy()
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

    # -------------------------
    # Load data and basic preprocess
    # -------------------------
    df = load_df()
    df.columns = df.columns.str.upper().str.strip()

    # choose product column
    product_candidates = [c for c in ['PRODUCTLINE','PRODUCTNAME','PRODUCT','ITEM','ITEMNAME'] if c in df.columns]
    if not product_candidates:
        st.error("Không tìm thấy cột sản phẩm (PRODUCTLINE/PRODUCTNAME/...). Kiểm tra file.")
        st.stop()
    product_col = product_candidates[0]

    # -------------------------
    # Sidebar: filters and metric
    # -------------------------
    # Hiển thị thông tin nguồn dữ liệu
    with st.expander("📂 Filters", expanded=True):

        # Chọn metric
        metric_choice = st.selectbox("Metric to analyze / Pareto", ["Revenue", "Profit"])

        # Chọn năm
        years = sorted(df['YEAR_ID'].dropna().unique()) if 'YEAR_ID' in df.columns else []
        selected_year = None
        if years:
            selected_year = st.selectbox("Select Year", options=years, index=len(years)-1)

        # Chọn quốc gia
        countries = sorted(df['COUNTRY'].dropna().unique()) if 'COUNTRY' in df.columns else []
        selected_countries = st.multiselect("Country (filter)", options=countries, default=countries)

        # Chọn deal size
        deals = sorted(df['DEALSIZE'].dropna().unique()) if 'DEALSIZE' in df.columns else []
        selected_deals = st.multiselect("DealSize (filter)", options=deals, default=deals)
    # apply filters
    df_f = df.copy()
    if selected_year is not None:
        df_f = df_f[df_f['YEAR_ID'] == selected_year]
    if selected_countries and 'COUNTRY' in df_f.columns:
        df_f = df_f[df_f['COUNTRY'].isin(selected_countries)]
    if selected_deals and 'DEALSIZE' in df_f.columns:
        df_f = df_f[df_f['DEALSIZE'].isin(selected_deals)]

    st.markdown(f"**Records after filters:** {len(df_f):,}")

    # -------------------------
    # Summary & COST handling
    # -------------------------
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
    missing_before = summary['COST'].isna().sum()
    summary['COST'] = summary.apply(lambda r: (r['AVG_PRICE'] * 0.5) if (pd.isna(r['COST']) or r['COST']==0) else r['COST'], axis=1)
    missing_after = summary['COST'].isna().sum()

    # session overrides
    if 'product_cost_overrides' not in st.session_state:
        st.session_state['product_cost_overrides'] = {}

    for p,v in st.session_state['product_cost_overrides'].items():
        summary.loc[summary['PRODUCT']==p, 'COST'] = v

    # compute PROFIT = (AVG_PRICE - COST) * QTY
    summary['PROFIT'] = (summary['AVG_PRICE'] - summary['COST']) * summary['QTY']
    # share columns
    total_rev = summary['REVENUE'].sum() if summary['REVENUE'].sum()!=0 else 1.0
    total_profit = summary['PROFIT'].sum() if summary['PROFIT'].sum()!=0 else 1.0
    summary['REV_SHARE'] = 100.0 * summary['REVENUE'] / total_rev
    summary['PROFIT_SHARE'] = 100.0 * summary['PROFIT'] / total_profit
    summary = summary.sort_values('REVENUE', ascending=False).reset_index(drop=True)

    # -------------------------
    # Selection UI (rule-based)
    # -------------------------
    left, right = st.columns([2,1])

    with right:
        st.subheader("Selection (Rule-based)")
        sel_mode = st.selectbox("Rule", ["Top N by metric", "Pareto cumulative %"])
        if sel_mode == "Top N by metric":
            default_top = min(10, len(summary))
            top_n = st.number_input("Top N", min_value=1, max_value=len(summary), value=default_top, step=1)
        else:
            default_pct = 80
            pct = st.slider("Pareto cumulative threshold (%)", min_value=10, max_value=100, value=default_pct, step=5)

        st.markdown("**Edit COST** (click cell to change). After editing, press 'Save COST' to apply.")
        editable = summary[['PRODUCT','COST']].set_index('PRODUCT')
        edited = st.data_editor(editable, num_rows="dynamic", use_container_width=True)

        if st.button("Save COST"):
            ed = edited.reset_index().rename(columns={'index':'PRODUCT'})
            for _, r in ed.iterrows():
                prod = r['PRODUCT']
                val = r['COST']
                try:
                    valf = float(val) if (val is not None and not (isinstance(val, float) and np.isnan(val))) else np.nan
                except Exception:
                    valf = np.nan
                st.session_state['product_cost_overrides'][prod] = valf
            for p,v in st.session_state['product_cost_overrides'].items():
                summary.loc[summary['PRODUCT']==p, 'COST'] = v
            summary['PROFIT'] = (summary['AVG_PRICE'] - summary['COST']) * summary['QTY']
            total_profit = summary['PROFIT'].sum() if summary['PROFIT'].sum()!=0 else 1.0
            summary['PROFIT_SHARE'] = 100.0 * summary['PROFIT'] / total_profit
            st.success("Saved COST to session and updated Profit.")

        if st.button("Download COST mapping"):
            cm = summary[['PRODUCT','COST']].copy()
            st.download_button("Download COST CSV", data=cm.to_csv(index=False).encode('utf-8'), file_name='product_cost_mapping.csv', mime='text/csv')

        st.markdown("---")
        st.subheader("Run selection")
        if st.button("Run selection"):
            st.session_state['run_selection'] = True

    # compute selected set based on current rule / metric
    if 'run_selection' not in st.session_state:
        st.session_state['run_selection'] = False

    # Always create selected_df dynamically so we can use it for breakdowns
    if sel_mode == "Top N by metric":
        if metric_choice == "Revenue":
            selected_df = summary.sort_values('REVENUE', ascending=False).head(top_n).copy()
        else:
            selected_df = summary.sort_values('PROFIT', ascending=False).head(top_n).copy()
    else:
        if metric_choice == "Revenue":
            tmp = summary.sort_values('REVENUE', ascending=False).reset_index(drop=True)
            tmp['CUM_PERC_METRIC'] = 100.0 * tmp['REVENUE'].cumsum() / (tmp['REVENUE'].sum() if tmp['REVENUE'].sum()!=0 else 1.0)
            selected_df = tmp[tmp['CUM_PERC_METRIC'] <= pct].copy()
        else:
            tmp = summary.sort_values('PROFIT', ascending=False).reset_index(drop=True)
            tmp['CUM_PERC_METRIC'] = 100.0 * tmp['PROFIT'].cumsum() / (tmp['PROFIT'].sum() if tmp['PROFIT'].sum()!=0 else 1.0)
            selected_df = tmp[tmp['CUM_PERC_METRIC'] <= pct].copy()

    # show selection summary
    with left:
        st.subheader("Selection result (rule-based)")
        if selected_df.empty:
            st.info("No products selected by the current rule. Try adjusting Top N or Pareto %.")
        else:
            metric_col = 'REVENUE' if metric_choice=='Revenue' else 'PROFIT'
            sel_count = len(selected_df)
            sel_metric_sum = selected_df[metric_col].sum()
            overall_metric_sum = summary[metric_col].sum() if summary[metric_col].sum()!=0 else 1.0
            sel_pct = 100.0 * sel_metric_sum / overall_metric_sum
            st.markdown(f"**Rule:** {sel_mode} — **Metric:** {metric_choice}")
            st.markdown(f"- Selected products: **{sel_count}**")
            st.markdown(f"- Sum {metric_choice}: **{sel_metric_sum:,.2f}** which is **{sel_pct:.2f}%** of total {metric_choice}.")
            show_cols = ['PRODUCT','REVENUE','PROFIT','QTY','AVG_PRICE','COST']
            show_cols = [c for c in show_cols if c in selected_df.columns]
            st.dataframe(selected_df[show_cols].style.format({'REVENUE':'{:,.2f}','PROFIT':'{:,.2f}','COST':'{:,.2f}','AVG_PRICE':'{:,.2f}'}))
            if st.button("Download selected CSV"):
                st.download_button("Download selected CSV", data=selected_df.to_csv(index=False).encode('utf-8'), file_name='selected_products.csv', mime='text/csv')

    # -------------------------
    # Pareto chart & Treemap (visual)
    # -------------------------
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

    fig = go.Figure()
    fig.add_trace(go.Bar(x=pareto_df['PRODUCT'], y=pareto_df[y_col], name=y_col))
    fig.add_trace(go.Scatter(x=pareto_df['PRODUCT'], y=pareto_df['CUM_PERC_METRIC'], name='Cumulative %', yaxis='y2', mode='lines+markers'))
    fig.update_layout(xaxis_tickangle=-45, yaxis=dict(title=y_col), yaxis2=dict(title='Cumulative %', overlaying='y', side='right', range=[0,100]), height=480)
    st.plotly_chart(fig, use_container_width=True)

    top_n_treem = min(50, len(pareto_df))
    top_n_treem = st.slider("Top N for treemap", min_value=1, max_value=top_n_treem, value=min(15, top_n_treem))
    try:
        treedf = pareto_df.head(top_n_treem).copy()
        color_col = 'REV_SHARE' if metric_choice=='Revenue' else 'PROFIT_SHARE'
        fig_t = px.treemap(treedf, path=['PRODUCT'], values=y_col, color=color_col, color_continuous_scale='Blues', title=f"Treemap (Top {top_n_treem} by {metric_choice})")
        st.plotly_chart(fig_t, use_container_width=True)
    except Exception as e:
        st.info("Cannot draw treemap: " + str(e))

    # -------------------------
    # BREAKDOWNS: compare metric across groups
    # -------------------------
    st.markdown("---")
    st.subheader("Breakdowns: compare metric across groups")

    # choose group
    available_groups = [c for c in ['PRODUCTLINE','COUNTRY','DEALSIZE'] if c in df_f.columns]
    if not available_groups:
        st.info("No grouping columns (PRODUCTLINE/COUNTRY/DEALSIZE) available in data.")
    else:
        group_by = st.selectbox("Compare by", options=available_groups, index=0)
        compare_with_selected = st.checkbox("Compare baseline (all filtered data) vs selected set", value=True)

        # baseline aggregation (all filtered df_f but grouped by group_by and product membership)
        # Baseline: sum metric for all records in df_f but aggregated by group_by
        if metric_choice == "Revenue":
            baseline_grp = df_f.groupby(group_by)['SALES'].sum().reset_index().rename(columns={'SALES':'METRIC'})
        else:
            # For profit baseline: approximate using product-level cost mapping and record-level qty
            # Build product->COST mapping from summary
            prod_cost = dict(zip(summary['PRODUCT'], summary['COST']))
            # Map product to records
            rec = df_f.copy()
            # determine product column in df_f (product_col)
            rec_prod_col = product_col
            if rec_prod_col not in rec.columns:
                st.error("Product column missing in raw data.")
                rec['PRODUCT_TMP'] = 'ALL'
                rec_prod_col = 'PRODUCT_TMP'
            rec['__PRODUCT_KEY__'] = rec[rec_prod_col].astype(str)
            # map cost and price
            rec['__COST__'] = rec['__PRODUCT_KEY__'].map(lambda x: prod_cost.get(x, np.nan))
            rec['PRICEEACH'] = pd.to_numeric(rec['PRICEEACH'], errors='coerce')
            rec['QUANTITYORDERED'] = pd.to_numeric(rec['QUANTITYORDERED'], errors='coerce').fillna(0)
            # if COST missing, fill with avg price*0.5 fallback
            rec['__COST__'] = rec.apply(lambda r: (r['PRICEEACH'] * 0.5) if pd.isna(r['__COST__']) or r['__COST__']==0 else r['__COST__'], axis=1)
            rec['RECORD_PROFIT'] = (rec['PRICEEACH'] - rec['__COST__']) * rec['QUANTITYORDERED']
            baseline_grp = rec.groupby(group_by)['RECORD_PROFIT'].sum().reset_index().rename(columns={'RECORD_PROFIT':'METRIC'})

        baseline_grp = baseline_grp.sort_values('METRIC', ascending=False).reset_index(drop=True)
        baseline_top = baseline_grp.head(30)

        if compare_with_selected and not selected_df.empty:
            # compute selected-set grouped metric: sum metric for only records that belong to selected products
            selected_products = selected_df['PRODUCT'].tolist()
            rec_sel = df_f.copy()
            # product key
            rec_sel['__PRODUCT_KEY__'] = rec_sel[product_col].astype(str)
            rec_sel = rec_sel[rec_sel['__PRODUCT_KEY__'].isin(selected_products)].copy()
            if metric_choice == "Revenue":
                sel_grp = rec_sel.groupby(group_by)['SALES'].sum().reset_index().rename(columns={'SALES':'METRIC'})
            else:
                # profit per record
                prod_cost = dict(zip(summary['PRODUCT'], summary['COST']))
                rec_sel['__COST__'] = rec_sel['__PRODUCT_KEY__'].map(lambda x: prod_cost.get(x, np.nan))
                rec_sel['PRICEEACH'] = pd.to_numeric(rec_sel['PRICEEACH'], errors='coerce')
                rec_sel['QUANTITYORDERED'] = pd.to_numeric(rec_sel['QUANTITYORDERED'], errors='coerce').fillna(0)
                rec_sel['__COST__'] = rec_sel.apply(lambda r: (r['PRICEEACH'] * 0.5) if pd.isna(r['__COST__']) or r['__COST__']==0 else r['__COST__'], axis=1)
                rec_sel['RECORD_PROFIT'] = (rec_sel['PRICEEACH'] - rec_sel['__COST__']) * rec_sel['QUANTITYORDERED']
                sel_grp = rec_sel.groupby(group_by)['RECORD_PROFIT'].sum().reset_index().rename(columns={'RECORD_PROFIT':'METRIC'})

            # merge baseline and selection for comparison
            cmp = baseline_grp.merge(sel_grp, on=group_by, how='left', suffixes=('_BASE','_SEL')).fillna(0)
            cmp = cmp.sort_values('METRIC_BASE', ascending=False).head(30)
            # prepare plotly grouped bar
            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Bar(x=cmp[group_by], y=cmp['METRIC_BASE'], name='Baseline (all filtered)'))
            fig_cmp.add_trace(go.Bar(x=cmp[group_by], y=cmp['METRIC_SEL'], name='Selected products'))
            fig_cmp.update_layout(barmode='group', xaxis_tickangle=-45, title=f"Compare {metric_choice}: Baseline vs Selected (by {group_by})", height=480)
            st.plotly_chart(fig_cmp, use_container_width=True)
        else:
            # just show baseline
            fig_base = px.bar(baseline_top, x=group_by, y='METRIC', title=f"{metric_choice} by {group_by} (baseline)", labels={'METRIC':metric_choice})
            st.plotly_chart(fig_base, use_container_width=True)

    st.markdown("---")
    st.subheader("Recommendations")
    st.write("""
    - Nếu mục tiêu là **Doanh thu**: dùng Pareto theo Revenue để lấy tập sản phẩm chiếm ~80% doanh thu — tập trung marketing & tồn kho cho chúng.
    - Nếu mục tiêu là **Lợi nhuận**: chuyển sang metric = Profit. Các sản phẩm top theo Profit có thể khác top theo Revenue (vì COST khác nhau).
    - Luôn **sửa COST thực tế** (nếu có) để kết quả profit chính xác. Auto-fill chỉ để demo.
    - Dùng tính năng Compare Baseline vs Selected để xem việc tập trung vào selected set thay đổi phân bố doanh thu/lợi nhuận theo Country/DealSize như thế nào.
    """)

