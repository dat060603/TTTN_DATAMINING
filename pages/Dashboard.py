# pages/Dashboard.py
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from components.data_loader import load_data

def app():
    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    local_css("style.css")
    st.title("📈 Dashboard — Visualize")

    # Load data
    df = load_data()

    # Style: CSS để tăng tính chuyên nghiệp
    # st.markdown("""
    #         <style>
    #         .block-container {
    #             padding-top: 2rem;
    #         }
    #         .metric {
    #             text-align: center !important;
    #         }
    #         .stMetric > div {
    #             background-color: #f0f2f6;
    #             padding: 10px;
    #             border-radius: 10px;
    #             border: 1px solid #ccc;
    #         }
    #         </style>
    #     """, unsafe_allow_html=True)

    # KPI (Tổng quan toàn bộ data)
    st.subheader("📊 Tổng quan dữ liệu")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("💰 Tổng doanh thu", f"${df['SALES'].sum():,.2f}")
    with col2:
        st.metric("📉 Doanh thu nhỏ nhất", f"${df['SALES'].min():,.2f}")
    with col3:
        st.metric("🏆 Doanh thu cao nhất", f"${df['SALES'].max():,.2f}")
    with col4:
        st.metric("📦 Trung bình theo đơn", f"${df['SALES'].mean():,.2f}")
    with col5:
        st.metric("📈 Độ lệch chuẩn", f"{df['SALES'].std():,.2f}")

    st.markdown("---")
    # st.markdown('<div class="fade-in"><h3>🚀 Chào mừng bạn đến với ứng dụng phân tích bán hàng!</h3></div>',
    #             unsafe_allow_html=True)
    # Bộ lọc
    with st.expander("🔍 Bộ lọc dữ liệu", expanded=True):
        colf1, colf2, colf3 = st.columns(3)
        with colf1:
            years = sorted(df['YEAR_ID'].dropna().unique())
            selected_year = st.selectbox("Chọn năm", years, index=len(years) - 1)
        with colf2:
            countries = sorted(df['COUNTRY'].dropna().unique())
            selected_country = st.multiselect("Chọn quốc gia", countries, default=countries)
        with colf3:
            product_lines = sorted(df['PRODUCTLINE'].dropna().unique())
            selected_product = st.multiselect("Chọn dòng sản phẩm", product_lines, default=product_lines)

    # Áp dụng bộ lọc
    df_filtered = df[
        (df['YEAR_ID'] == selected_year) &
        (df['COUNTRY'].isin(selected_country)) &
        (df['PRODUCTLINE'].isin(selected_product))
        ]

    # KPIs sau khi lọc
    st.subheader("📌 Thống kê sau khi lọc")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("💰 Doanh thu (lọc)", f"${df_filtered['SALES'].sum():,.2f}")
    with col2:
        st.metric("🧾 Số đơn hàng", f"{df_filtered['ORDERNUMBER'].nunique():,}")
    with col3:
        st.metric("📊 Doanh thu trung bình", f"${df_filtered['SALES'].mean():,.2f}")
    with col4:
        st.metric("💵 Giá trung bình", f"${df_filtered['PRICEEACH'].mean():,.2f}")

    st.markdown("---")
    # Top products & customers
    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.subheader("Top 5 sản phẩm theo doanh thu")
        if 'PRODUCTLINE' in df_filtered.columns and 'SALES' in df_filtered.columns:
            prod_top = (df_filtered.groupby('PRODUCTLINE')['SALES']
                        .sum()
                        .reset_index()
                        .sort_values('SALES', ascending=False)
                        .head(5))
            fig_prod = px.bar(prod_top, x='SALES', y='PRODUCTLINE', orientation='h',
                              title='Top 5 sản phẩm', labels={'SALES': 'Doanh thu', 'PRODUCTLINE': 'Dòng sản phẩm'})
            st.plotly_chart(fig_prod, use_container_width=True)
        else:
            st.info("Không có dữ liệu PRODUCTLINE hoặc SALES để hiển thị Top sản phẩm.")

    with col_right:
        st.subheader("Top 5 khách hàng theo doanh thu")
        if 'CUSTOMERNAME' in df_filtered.columns and 'SALES' in df_filtered.columns:
            cust_top = (df_filtered.groupby('CUSTOMERNAME')['SALES']
                        .sum()
                        .reset_index()
                        .sort_values('SALES', ascending=False)
                        .head(5))
            fig_cust = px.bar(cust_top, x='SALES', y='CUSTOMERNAME', orientation='h',
                              title='Top 5 khách hàng', labels={'SALES': 'Doanh thu', 'CUSTOMERNAME': 'Khách hàng'})
            st.plotly_chart(fig_cust, use_container_width=True)
        else:
            st.info("Không có dữ liệu CUSTOMERNAME hoặc SALES để hiển thị Top khách hàng.")
    st.markdown("---")

    # Tabs hiển thị đồ họa
    st.subheader("📊 Biểu đồ minh họa")
    tab1, tab2, tab3 = st.tabs([
        "🔁 Doanh thu theo tháng",
        "🌍 Doanh thu theo quốc gia",
        "📅 Doanh thu theo thời gian"
    ])

    with tab1:
        if 'MONTH_ID' in df_filtered.columns:
            df_month = df_filtered.groupby('MONTH_ID')['SALES'].sum().reset_index()
            fig = px.bar(df_month, x='MONTH_ID', y='SALES',
                         labels={'MONTH_ID': 'Tháng', 'SALES': 'Doanh thu'},
                         title="Doanh thu theo tháng")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("📊 Doanh thu theo quốc gia")

        if 'COUNTRY' in df_filtered.columns and 'SALES' in df_filtered.columns:
            country_sales = df_filtered.groupby('COUNTRY')['SALES'].sum().reset_index().sort_values('SALES',
                                                                                                    ascending=False)

            col1, col2 = st.columns(2)

            with col1:
                try:
                    fig_chor = px.choropleth(
                        country_sales,
                        locations='COUNTRY',
                        locationmode='country names',
                        color='SALES',
                        color_continuous_scale='Blues',
                        title='Doanh thu theo quốc gia',
                        labels={'SALES': 'Doanh thu'}
                    )
                    st.plotly_chart(fig_chor, use_container_width=True)
                except Exception:
                    st.warning("Không thể hiển thị bản đồ, chuyển sang biểu đồ cột.")
                    fig_bar_country = px.bar(
                        country_sales.head(20),
                        x='COUNTRY',
                        y='SALES',
                        title='Doanh thu theo quốc gia (Bar)',
                        labels={'SALES': 'Doanh thu', 'COUNTRY': 'Quốc gia'}
                    )
                    st.plotly_chart(fig_bar_country, use_container_width=True)

            with col2:
                fig_pie = px.pie(
                    country_sales.head(10),
                    names='COUNTRY',
                    values='SALES',
                    title='Top 10 quốc gia theo doanh thu'
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            st.markdown("---")
            with st.expander("📄 Xem dữ liệu đã lọc"):
                st.dataframe(df_filtered[['COUNTRY', 'SALES']].dropna().head(200))

        else:
            st.info("Không có cột 'COUNTRY' hoặc 'SALES' để hiển thị biểu đồ.")
    with tab3:
        st.subheader("📅 Doanh thu theo thời gian (theo tháng)")
        if ('ORDERDATE' in df_filtered.columns) and ('SALES' in df_filtered.columns):
            ts = df_filtered[['ORDERDATE', 'SALES']].dropna(subset=['ORDERDATE']).copy()

            # Chuyển ORDERDATE sang datetime nếu chưa
            if not pd.api.types.is_datetime64_any_dtype(ts['ORDERDATE']):
                ts['ORDERDATE'] = pd.to_datetime(ts['ORDERDATE'])

            ts = ts.groupby('ORDERDATE')['SALES'].sum().reset_index().sort_values('ORDERDATE')
            ts = ts.set_index('ORDERDATE').resample('ME').sum().reset_index()
            ts['month_str'] = ts['ORDERDATE'].dt.strftime('%Y-%m')
            ts['rolling_3m'] = ts['SALES'].rolling(window=3, min_periods=1).mean()

            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(x=ts['ORDERDATE'], y=ts['SALES'],
                                        mode='lines+markers',
                                        name='Doanh thu (tháng)'))
            fig_ts.add_trace(go.Scatter(x=ts['ORDERDATE'], y=ts['rolling_3m'],
                                        mode='lines',
                                        name='Rolling mean (3 tháng)',
                                        line=dict(dash='dash')))
            fig_ts.update_layout(title="📅 Doanh thu theo tháng với Rolling Mean (3 tháng)",
                                 xaxis_title="Thời gian", yaxis_title="Doanh thu (USD)",
                                 template='plotly_white')
            st.plotly_chart(fig_ts, use_container_width=True)
        else:
            st.info("❗ Cần có cột ORDERDATE và SALES để hiển thị đồ thị thời gian.")

    st.markdown("---")

    # -----------------------
    # Time series
    # st.subheader("Doanh thu theo thời gian (theo tháng)")
    # if ('ORDERDATE' in df_filtered.columns) and ('SALES' in df_filtered.columns):
    #     ts = df_filtered[['ORDERDATE','SALES']].dropna(subset=['ORDERDATE'])
    #     ts = ts.groupby('ORDERDATE')['SALES'].sum().reset_index().sort_values('ORDERDATE')
    #     ts = ts.set_index('ORDERDATE').resample('M').sum().reset_index()
    #     ts['month_str'] = ts['ORDERDATE'].dt.strftime('%Y-%m')
    #     ts['rolling_3m'] = ts['SALES'].rolling(window=3, min_periods=1).mean()
    #
    #     fig_ts = go.Figure()
    #     fig_ts.add_trace(go.Scatter(x=ts['ORDERDATE'], y=ts['SALES'], mode='lines+markers', name='Doanh thu (tháng)'))
    #     fig_ts.add_trace(go.Scatter(x=ts['ORDERDATE'], y=ts['rolling_3m'], mode='lines', name='Rolling mean (3m)', line=dict(dash='dash')))
    #     fig_ts.update_layout(title="Doanh thu theo tháng với Rolling Mean (3 tháng)",
    #                          xaxis_title="Thời gian", yaxis_title="Doanh thu (USD)", template='plotly_white')
    #     st.plotly_chart(fig_ts, use_container_width=True)
    # else:
    #     st.info("Cần cột ORDERDATE và SALES để hiển thị đồ thị thời gian.")
    #
    # # -----------------------


    # -----------------------
    # Choropleth / fallback bar by country
    # st.subheader("Doanh thu theo quốc gia (Choropleth)")
    # if 'COUNTRY' in df_filtered.columns and 'SALES' in df_filtered.columns:
    #     country_sales = (df_filtered.groupby('COUNTRY')['SALES'].sum().reset_index().sort_values('SALES', ascending=False))
    #     try:
    #         fig_chor = px.choropleth(country_sales, locations='COUNTRY', locationmode='country names',
    #                                  color='SALES', color_continuous_scale='Blues',
    #                                  title='Doanh thu theo quốc gia', labels={'SALES':'Doanh thu'})
    #         st.plotly_chart(fig_chor, use_container_width=True)
    #     except Exception:
    #         fig_bar_country = px.bar(country_sales.head(20), x='COUNTRY', y='SALES',
    #                                  title='Doanh thu theo quốc gia (Bar)', labels={'SALES':'Doanh thu', 'COUNTRY':'Quốc gia'})
    #         st.plotly_chart(fig_bar_country, use_container_width=True)
    # else:
    #     st.info("Không có cột COUNTRY hoặc SALES để hiển thị bản đồ/quốc gia.")
    #
    # st.markdown("---")
    # with st.expander("📄 Xem dữ liệu (Filtered)"):
    #     st.dataframe(df_filtered.head(200))
