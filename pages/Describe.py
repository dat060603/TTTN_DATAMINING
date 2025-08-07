import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from components.data_loader import load_data

def app():
    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    local_css("style.css")
    st.title("📊 Describe — Thống kê mô tả dữ liệu")

    # Load dữ liệu
    df = load_data()

    # ---------- Bộ lọc dữ liệu ----------
    with st.expander("🔍 Bộ lọc dữ liệu", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            years = sorted(df['YEAR_ID'].dropna().unique()) if 'YEAR_ID' in df.columns else []
            selected_year = st.selectbox("Chọn năm", options=years, index=(len(years)-1) if years else 0)
        with col2:
            countries = sorted(df['COUNTRY'].dropna().unique())
            selected_country = st.multiselect("Chọn quốc gia", options=countries, default=countries)
        with col3:
            product_lines = sorted(df['PRODUCTLINE'].dropna().unique())
            selected_product = st.multiselect("Chọn dòng sản phẩm", options=product_lines, default=product_lines)

    # Áp bộ lọc
    _df = df.copy()
    if selected_year: _df = _df[_df['YEAR_ID'] == selected_year]
    if selected_country: _df = _df[_df['COUNTRY'].isin(selected_country)]
    if selected_product: _df = _df[_df['PRODUCTLINE'].isin(selected_product)]

    # Ép kiểu numeric
    for c in ['SALES', 'PRICEEACH', 'QUANTITYORDERED', 'MSRP']:
        if c in _df.columns:
            _df[c] = pd.to_numeric(_df[c], errors='coerce')

    # ---------- KPI tổng quan ----------
    st.markdown("### 📌 Thống kê nhanh")
    k1, k2, k3 = st.columns(3)
    k1.metric("💰 Tổng doanh thu", f"${_df['SALES'].sum():,.2f}")
    k2.metric("🧾 Số đơn hàng", f"{_df['ORDERNUMBER'].nunique():,}")
    k3.metric("📊 Doanh thu TB", f"${_df['SALES'].mean():,.2f}")

    # ---------- Tabs ----------
    tab1, tab2, tab3 = st.tabs(["📈 Doanh thu", "📦 Sản phẩm", "👥 Khách hàng"])

    # ---------- TAB 1: Doanh thu ----------
    with tab1:
        st.subheader("Phân phối doanh thu")
        if 'SALES' in _df.columns:
            col1, col2 = st.columns([2, 1])
            with col1:
                fig_hist = px.histogram(_df, x='SALES', nbins=40, marginal='box', title="Histogram: Doanh thu")
                st.plotly_chart(fig_hist, use_container_width=True)
            with col2:
                fig_box = px.box(_df, y='SALES', title="Box Plot: Doanh thu")
                st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.info("Không có cột SALES để hiển thị.")

    # ---------- TAB 2: Sản phẩm ----------
    with tab2:
        st.subheader("Tổng quan sản phẩm")
        c1, c2 = st.columns(2)
        c1.metric("🔢 Tổng số lượng sản phẩm", f"{_df['QUANTITYORDERED'].sum():,.0f}")
        c2.metric("📊 Số lượng TB theo đơn", f"{_df['QUANTITYORDERED'].mean():.2f}")

        if 'QUANTITYORDERED' in _df.columns:
            fig_qty = px.histogram(_df, x='QUANTITYORDERED', nbins=30, title='Phân phối số lượng đặt hàng')
            st.plotly_chart(fig_qty, use_container_width=True)

        if 'PRICEEACH' in _df.columns:
            fig_price = px.histogram(_df, x='PRICEEACH', nbins=40, title='Phân phối đơn giá (PRICEEACH)')
            st.plotly_chart(fig_price, use_container_width=True)

        if 'PRODUCTLINE' in _df.columns and 'SALES' in _df.columns:
            st.subheader("Doanh thu theo dòng sản phẩm")
            prod_sales = _df.groupby('PRODUCTLINE')['SALES'].sum().reset_index().sort_values('SALES', ascending=False)
            fig_prod = px.bar(prod_sales, x='PRODUCTLINE', y='SALES', title="Bar Chart: Doanh thu theo dòng sản phẩm")
            st.plotly_chart(fig_prod, use_container_width=True)

            # Pareto chart
            prod_sales['cum_sum'] = prod_sales['SALES'].cumsum()
            prod_sales['cum_perc'] = 100 * prod_sales['cum_sum'] / prod_sales['SALES'].sum()
            fig_p = go.Figure()
            fig_p.add_trace(go.Bar(x=prod_sales['PRODUCTLINE'], y=prod_sales['SALES'], name='Doanh thu'))
            fig_p.add_trace(go.Scatter(x=prod_sales['PRODUCTLINE'], y=prod_sales['cum_perc'],
                                       name='Tỷ lệ lũy kế (%)', yaxis='y2', mode='lines+markers'))
            fig_p.update_layout(title='Pareto Chart: Doanh thu theo PRODUCTLINE',
                                yaxis=dict(title='Doanh thu'),
                                yaxis2=dict(title='Tỷ lệ lũy kế (%)', overlaying='y', side='right', range=[0,100]))
            st.plotly_chart(fig_p, use_container_width=True)
        else:
            st.info("Thiếu cột PRODUCTLINE hoặc SALES.")

    # ---------- TAB 3: Khách hàng ----------
    with tab3:
        st.subheader("Doanh thu theo khách hàng (Top 50)")
        if 'CUSTOMERNAME' in _df.columns and 'SALES' in _df.columns:
            cust_rev = _df.groupby('CUSTOMERNAME')['SALES'].sum().reset_index().sort_values('SALES', ascending=False).head(50)
            try:
                fig_tree = px.treemap(cust_rev, path=['CUSTOMERNAME'], values='SALES')
                st.plotly_chart(fig_tree, use_container_width=True)
            except:
                st.dataframe(cust_rev)

        if 'COUNTRY' in _df.columns and 'CUSTOMERNAME' in _df.columns:
            st.subheader("Số lượng khách hàng theo quốc gia (Top 20)")
            country_cnt = _df.groupby('COUNTRY')['CUSTOMERNAME'].nunique().reset_index(name='Số KH')
            country_cnt = country_cnt.sort_values('Số KH', ascending=False)
            st.bar_chart(country_cnt.set_index('COUNTRY').head(20))

    # ---------- Dữ liệu và mô tả ----------
    with st.expander("📊 Thống kê mô tả"):
        desc_cols = ['SALES', 'PRICEEACH', 'QUANTITYORDERED']
        st.dataframe(_df[desc_cols].describe().T)
    with st.expander("📄 Xem dữ liệu đã lọc"):
        st.dataframe(_df.head(300))
    with st.expander("📥 Tải dữ liệu CSV"):
        csv = _df.to_csv(index=False).encode('utf-8')
        st.download_button(label="Tải xuống CSV", data=csv, file_name="describe_filtered.csv", mime="text/csv")
