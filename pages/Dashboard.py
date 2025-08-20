# pages/Dashboard.py
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from components.data_loader import load_data
import time
def app():
    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    local_css("style.css")
    st.title("📈 Dashboard")

    # Load data
    df = load_data()
    # --- KPI (Tổng quan toàn bộ data) ---
    st.subheader("📊 Theo dõi KPI")

    # CSS hiển thị rõ ràng
    st.markdown("""
        <style>
        .metric-big {
            font-size: 20px;
            font-weight: bold;
            text-align: center;
        }
        .metric-label {
            text-align: center;
            font-size: 12px;
            color: gray;
        }
        </style>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4,col5  = st.columns(5)
    with col1:
        st.markdown(f"<div class='metric-big'>💰 ${df['SALES'].sum():,.2f}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Tổng doanh thu</div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-big'>🧾 {df['ORDERNUMBER'].nunique():,}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Số đơn hàng</div>", unsafe_allow_html=True)
        # st.markdown(f"<div class='metric-big'>🏆 ${df['SALES'].max():,.2f}</div>", unsafe_allow_html=True)
        # st.markdown("<div class='metric-label'>Doanh thu cao nhất</div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-big'>💵 ${df['PRICEEACH'].mean():,.2f}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Giá trung bình</div>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<div class='metric-big'>📦 ${df['SALES'].mean():,.2f}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Trung bình theo đơn</div>", unsafe_allow_html=True)
    with col5:
        st.markdown(f"<div class='metric-big'>📈 {df['SALES'].std():,.2f}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Độ lệch chuẩn</div>", unsafe_allow_html=True)

    st.markdown("---")

    # --- Bộ lọc dữ liệu ---
    with st.expander("🔍 Bộ lọc dữ liệu", expanded=True):
        colf1, colf2, colf3 = st.columns(3)
        with colf1:
            years = sorted(df['YEAR_ID'].dropna().unique())
            year_options = ["Tất cả các năm"] + [str(int(y)) for y in years]
            selected_year = st.selectbox("Chọn năm", year_options, index=0)
        with colf2:
            countries = sorted(df['COUNTRY'].dropna().unique())
            selected_country = st.multiselect("Chọn quốc gia", countries, default=countries)
        with colf3:
            product_lines = sorted(df['PRODUCTLINE'].dropna().unique())
            selected_product = st.multiselect("Chọn dòng sản phẩm", product_lines, default=product_lines)

    # --- Áp dụng bộ lọc ---
    df_filtered = df.copy()

    # Lọc theo năm nếu không phải "Tất cả các năm"
    if selected_year != "Tất cả các năm":
        df_filtered = df_filtered[df_filtered['YEAR_ID'] == int(selected_year)]

    # Lọc theo quốc gia và dòng sản phẩm
    df_filtered = df_filtered[
        (df_filtered['COUNTRY'].isin(selected_country)) &
        (df_filtered['PRODUCTLINE'].isin(selected_product))
        ]

    # --- KPI sau khi lọc ---
    st.subheader("📌 Thống kê số liệu kinh doanh")

    # --- Hàng 1: Doanh thu & Đơn hàng & Giá trung bình ---
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"<div class='metric-big'>💰 ${df_filtered['SALES'].sum():,.2f}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Doanh thu</div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-big'>📊 ${df_filtered['SALES'].mean():,.2f}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Doanh thu trung bình</div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-big'>🏆 ${df['SALES'].max():,.2f}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Doanh thu cao nhất</div>", unsafe_allow_html=True)

    with col4:
        st.markdown(f"<div class='metric-big'>📉 ${df['SALES'].min():,.2f}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Doanh thu nhỏ nhất</div>", unsafe_allow_html=True)


    st.markdown("---")

    # --- Hàng 2: COST ---
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        total_cost = (df_filtered['COST'] * df_filtered['QUANTITYORDERED']).sum()
        st.markdown(f"<div class='metric-big'>💵 ${total_cost:,.2f}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Tổng COST</div>", unsafe_allow_html=True)
    with col2:
        avg_cost = (df_filtered['COST'] * df_filtered['QUANTITYORDERED']).mean()
        st.markdown(f"<div class='metric-big'>💰 ${avg_cost:,.2f}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>COST trung bình</div>", unsafe_allow_html=True)
    with col3:
        max_cost = (df_filtered['COST'] * df_filtered['QUANTITYORDERED']).max()
        st.markdown(f"<div class='metric-big'>📈 ${max_cost:,.2f}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>COST cao nhất</div>", unsafe_allow_html=True)
    with col4:
        min_cost = (df_filtered['COST'] * df_filtered['QUANTITYORDERED']).min()
        st.markdown(f"<div class='metric-big'>📉 ${min_cost:,.2f}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>COST thấp nhất</div>", unsafe_allow_html=True)

    st.markdown("---")

    # --- Hàng 3: LỢI NHUẬN ---
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        df_filtered['PROFIT'] = df_filtered['SALES'] - df_filtered['COST'] * df_filtered['QUANTITYORDERED']
        st.markdown(f"<div class='metric-big'>💰 ${df_filtered['PROFIT'].sum():,.2f}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Tổng PROFIT</div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-big'>📊 ${df_filtered['PROFIT'].mean():,.2f}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Profit trung bình</div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-big'>📈 ${df_filtered['PROFIT'].max():,.2f}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Profit cao nhất</div>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<div class='metric-big'>📉 ${df_filtered['PROFIT'].min():,.2f}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Profit thấp nhất</div>", unsafe_allow_html=True)

    st.markdown("---")

    # Top products & customers
    st.subheader("📊 Top doanh thu theo nhiều chiều")
    with st.container():
        col1, col2 = st.columns(2)

        # ==== COL1: Sản phẩm ====
        with col1:
            with st.spinner("Đang phân tích dòng sản phẩm..."):
                time.sleep(0.5)
                if 'PRODUCTLINE' in df_filtered.columns and 'SALES' in df_filtered.columns:
                    prod_top = (
                        df_filtered.groupby('PRODUCTLINE')['SALES']
                        .sum()
                        .reset_index()
                        .sort_values('SALES', ascending=False)
                        .head(5)
                    )
                    fig_prod = px.bar(
                        prod_top, x='SALES', y='PRODUCTLINE', orientation='h',
                        title='Top 5 dòng sản phẩm',

                        labels={'SALES': 'Doanh thu (USD)', 'PRODUCTLINE': 'Dòng sản phẩm'},
                        text='SALES'
                    )
                    fig_prod.update_traces(
                        marker_color='rgb(63, 81, 181)',
                        texttemplate='%{text:,.0f}',
                        textposition='outside'
                    )
                    fig_prod.update_layout(
                        xaxis_title='Doanh thu',
                        yaxis_title='',
                        title_x=0,
                        plot_bgcolor='rgba(0,0,0,0)',
                        transition={'duration': 500},
                        hoverlabel=dict(bgcolor="white", font_size=13)
                    )
                    st.plotly_chart(fig_prod, use_container_width=True)
                else:
                    st.info("Không có dữ liệu PRODUCTLINE hoặc SALES.")

            with st.spinner("Đang phân tích mã sản phẩm..."):
                time.sleep(0.5)
                if 'PRODUCTCODE' in df_filtered.columns and 'SALES' in df_filtered.columns:
                    prodcode_top = (
                        df_filtered.groupby('PRODUCTCODE')['SALES']
                        .sum()
                        .reset_index()
                        .sort_values('SALES', ascending=False)
                        .head(5)
                    )
                    fig_prodcode = px.bar(
                        prodcode_top, x='SALES', y='PRODUCTCODE', orientation='h',
                        title='Top 5 mã sản phẩm',
                        labels={'SALES': 'Doanh thu (USD)', 'PRODUCTCODE': 'Mã sản phẩm'},
                        text='SALES'
                    )
                    fig_prodcode.update_traces(
                        marker_color='rgb(0, 153, 153)',
                        texttemplate='%{text:,.0f}',
                        textposition='outside'
                    )
                    fig_prodcode.update_layout(
                        xaxis_title='Doanh thu',
                        yaxis_title='',
                        title_x=0,
                        plot_bgcolor='rgba(0,0,0,0)',
                        transition={'duration': 500},
                        hoverlabel=dict(bgcolor="white", font_size=13)
                    )
                    st.plotly_chart(fig_prodcode, use_container_width=True)
                else:
                    st.info("Không có dữ liệu PRODUCTCODE hoặc SALES.")

        # ==== COL2: Khách hàng & Quốc gia ====
        with col2:
            with st.spinner("Đang phân tích khách hàng..."):
                time.sleep(0.5)
                if 'CUSTOMERNAME' in df_filtered.columns and 'SALES' in df_filtered.columns:
                    cust_top = (
                        df_filtered.groupby('CUSTOMERNAME')['SALES']
                        .sum()
                        .reset_index()
                        .sort_values('SALES', ascending=False)
                        .head(5)
                    )
                    fig_cust = px.bar(
                        cust_top, x='SALES', y='CUSTOMERNAME', orientation='h',
                        title='Top 5 khách hàng',
                        labels={'SALES': 'Doanh thu (USD)', 'CUSTOMERNAME': 'Khách hàng'},
                        text='SALES'
                    )
                    fig_cust.update_traces(
                        marker_color='rgb(255, 111, 97)',  # màu khác để dễ phân biệt
                        texttemplate='%{text:,.0f}',
                        textposition='outside'
                    )
                    fig_cust.update_layout(
                        xaxis_title='Doanh thu',
                        yaxis_title='',
                        title_x=0,
                        plot_bgcolor='rgba(0,0,0,0)',
                        transition={'duration': 500},
                        hoverlabel=dict(bgcolor="white", font_size=13)
                    )
                    st.plotly_chart(fig_cust, use_container_width=True)
                else:
                    st.info("Không có dữ liệu CUSTOMERNAME hoặc SALES.")

            with st.spinner("Đang phân tích quốc gia..."):
                time.sleep(0.5)
                if 'COUNTRY' in df_filtered.columns and 'SALES' in df_filtered.columns:
                    country_top = (
                        df_filtered.groupby('COUNTRY')['SALES']
                        .sum()
                        .reset_index()
                        .sort_values('SALES', ascending=False)
                        .head(5)
                    )
                    fig_country = px.bar(
                        country_top, x='SALES', y='COUNTRY', orientation='h',
                        title='Top 5 quốc gia',
                        labels={'SALES': 'Doanh thu (USD)', 'COUNTRY': 'Quốc gia'},
                        text='SALES'
                    )
                    fig_country.update_traces(
                        marker_color='rgb(255, 184, 108)',  # màu riêng để dễ phân biệt
                        texttemplate='%{text:,.0f}',
                        textposition='outside'
                    )
                    fig_country.update_layout(
                        xaxis_title='Doanh thu',
                        yaxis_title='',
                        title_x=0,
                        plot_bgcolor='rgba(0,0,0,0)',
                        transition={'duration': 500},
                        hoverlabel=dict(bgcolor="white", font_size=13)
                    )
                    st.plotly_chart(fig_country, use_container_width=True)
                else:
                    st.info("Không có dữ liệu COUNTRY hoặc SALES.")

    st.markdown("---")
    # Tabs hiển thị đồ họa
    st.subheader("📊 Doanh thu")
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔁 Nhóm theo tháng",
        "🌍 Nhóm theo quốc gia",
        "📅 Nhóm theo thời gian",
        "📅 Nhóm theo đơn hàng",
        "📅 Phân phối doanh thu"
    ])
    with tab1:
        # Kiểm tra nếu có dữ liệu tháng và năm
        if 'MONTH_ID' in df_filtered.columns:
            df_month = df_filtered.copy()

            # Nếu có cột YEAR_ID thì tạo animation theo năm
            if 'YEAR_ID' in df_month.columns:
                df_grouped = df_month.groupby(['YEAR_ID', 'MONTH_ID'])['SALES'].sum().reset_index()
                df_grouped = df_grouped.sort_values(['YEAR_ID', 'MONTH_ID'])

                with st.spinner("⏳ Đang tạo biểu đồ doanh thu theo tháng..."):
                    time.sleep(1)  # tạo cảm giác mượt
                    fig = px.bar(
                        df_grouped,
                        x='MONTH_ID',
                        y='SALES',
                        animation_frame='YEAR_ID',
                        range_y=[0, df_grouped['SALES'].max() * 1.1],
                        labels={'MONTH_ID': 'Tháng', 'SALES': 'Doanh thu', 'YEAR_ID': 'Năm'},
                        title="📊 Doanh thu theo tháng "
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(
                        "📌 Biểu đồ thể hiện sự thay đổi doanh thu theo tháng và từng năm. Giúp xác định mùa cao điểm/thấp điểm qua các năm.")
            else:
                # Nếu không có YEAR_ID → biểu đồ tĩnh
                df_grouped = df_month.groupby('MONTH_ID')['SALES'].sum().reset_index()
                with st.spinner("⏳ Đang tạo biểu đồ doanh thu theo tháng..."):
                    time.sleep(1)
                    fig = px.bar(
                        df_grouped,
                        x='MONTH_ID',
                        y='SALES',
                        labels={'MONTH_ID': 'Tháng', 'SALES': 'Doanh thu'},
                        title="📊 Doanh thu theo tháng"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(
                        "📌 Biểu đồ thể hiện sự thay đổi doanh thu theo từng tháng. Giúp xác định mùa cao điểm/thấp điểm.")
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
                    st.caption("📌 Bản đồ doanh thu theo quốc gia. Cho biết thị trường nào đang đóng góp lớn nhất.")
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
                st.caption("📌 Biểu đồ tròn giúp xác định 10 quốc gia hàng đầu theo doanh thu.")
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
            st.caption("📌 Biểu đồ đường doanh thu theo thời gian giúp phát hiện xu hướng tăng/giảm và biến động.")
        else:
            st.info("❗ Cần có cột ORDERDATE và SALES để hiển thị đồ thị thời gian.")
    with tab4:
        st.subheader("📦 Phân tích theo trạng thái đơn hàng")
        status_df = df.groupby('STATUS').agg(
            OrderCount=('ORDERNUMBER', 'nunique'),
            TotalSales=('SALES', 'sum')
        ).reset_index()

        col1, col2 = st.columns(2)

        with col1:
            fig_status_count = px.bar(
                status_df,
                x='STATUS',
                y='OrderCount',
                title='Số lượng đơn hàng theo trạng thái',
                text='OrderCount',
                color='STATUS'
            )
            st.plotly_chart(fig_status_count, use_container_width=True)
            st.caption("📌 Số lượng đơn hàng theo trạng thái cho thấy tiến độ và tình trạng xử lý đơn hàng.")
        with col2:
            fig_status_sales = px.pie(
                status_df,
                names='STATUS',
                values='TotalSales',
                title='Tỷ trọng doanh thu theo trạng thái đơn hàng'
            )
            st.plotly_chart(fig_status_sales, use_container_width=True)
            st.caption("📌 Tỷ trọng doanh thu theo trạng thái đơn hàng cho thấy mức độ ảnh hưởng của mỗi trạng thái.")
    with tab5:
        st.subheader("Phân phối doanh thu")
        if 'SALES' in df_filtered.columns:
            col1, col2 = st.columns([2, 1])
            with col1:
                fig_hist = px.histogram(df_filtered, x='SALES', nbins=40, marginal='box', title="Histogram: Doanh thu")
                st.plotly_chart(fig_hist, use_container_width=True)
            with col2:
                fig_box = px.box(df_filtered, y='SALES', title="Box Plot: Doanh thu")
                st.plotly_chart(fig_box, use_container_width=True)
            st.markdown("""
                                    **🧠 Khuyến nghị**
                                    - Xây dựng chiến lược giá: Nhắm vào khoảng giá có nhiều đơn hàng.
                                    - Chăm sóc khách hàng: Ưu tiên nhóm tạo ra doanh thu cao hoặc ổn định.
                                    - Phân loại khách hàng: Dựa theo giá trị đơn hàng để phân nhóm.
                                    - Tối ưu danh mục sản phẩm: Nhận diện sản phẩm có đơn hàng cao/bất thường.
                                    """)

        else:
            st.info("Không có cột SALES để hiển thị.")
    st.markdown("---")
    # Tabs hiển thị đồ họa
    st.subheader("📊 Lợi nhuận - Chi phí")
    tab1, tab2, tab3, tab4= st.tabs([
        "📊 Pareto: Doanh thu & Cost",
        "💵 Scatter: Doanh thu vs Profit",
        "📉 Phân phối Profit & Cost",
        "🗺️ Treemap: Doanh thu & Profit"
    ])

    with tab1:
        numeric_cols = ['SALES', 'COST']
        prod_summary = df_filtered.groupby('PRODUCTLINE')[numeric_cols].sum().reset_index().sort_values('SALES',
                                                                                                        ascending=False)
        prod_summary['cum_sales_perc'] = prod_summary['SALES'].cumsum() / prod_summary['SALES'].sum() * 100

        fig = go.Figure()
        fig.add_trace(go.Bar(x=prod_summary['PRODUCTLINE'], y=prod_summary['SALES'], name='Doanh thu'))
        fig.add_trace(go.Bar(x=prod_summary['PRODUCTLINE'], y=prod_summary['COST'], name='COST'))
        fig.add_trace(go.Scatter(x=prod_summary['PRODUCTLINE'], y=prod_summary['cum_sales_perc'],
                                 name='Tỷ lệ lũy kế (%)', yaxis='y2', mode='lines+markers'))

        fig.update_layout(title='Pareto: Doanh thu & COST theo dòng sản phẩm',
                          yaxis=dict(title='USD'),
                          yaxis2=dict(title='Tỷ lệ lũy kế (%)', overlaying='y', side='right', range=[0, 100]),
                          barmode='group')
        st.plotly_chart(fig, use_container_width=True)

    # ==== Tab7: Scatter: Doanh thu vs Lợi nhuận ====
    with tab2:
        if 'COST' in df_filtered.columns and 'QUANTITYORDERED' in df_filtered.columns:
            df_filtered['PROFIT'] = df_filtered['SALES'] - df_filtered['COST'] * df_filtered['QUANTITYORDERED']
        else:
            st.warning("❗ Thiếu cột COST hoặc QUANTITYORDERED để tính PROFIT.")

        numeric_cols = ['SALES', 'PROFIT', 'QUANTITYORDERED']
        df_grouped = df_filtered.groupby('PRODUCTCODE')[numeric_cols].sum().reset_index()

        fig = px.scatter(df_grouped,
                         x='SALES', y='PROFIT', size='QUANTITYORDERED',
                         hover_name='PRODUCTCODE',
                         title='Scatter: Doanh thu vs Lợi nhuận',
                         labels={'SALES': 'Doanh thu (USD)', 'PROFIT': 'Lợi nhuận (USD)'},
                         color='PRODUCTCODE')  # hoặc 'PRODUCTLINE' nếu muốn theo dòng sản phẩm

        st.plotly_chart(fig, use_container_width=True)

    # ==== Tab8: Histogram: Profit & COST ====
    with tab3:
        fig = px.histogram(df_filtered, x='PROFIT', nbins=50, title='Phân phối lợi nhuận')
        st.plotly_chart(fig, use_container_width=True)
        fig2 = px.histogram(df_filtered, x='COST', nbins=50, title='Phân phối COST')
        st.plotly_chart(fig2, use_container_width=True)
    # ==== Tab9: Treemap: Doanh thu & Profit theo dòng sản phẩm ====
    with tab4:
        numeric_cols = ['SALES', 'PROFIT']
        prod_summary = df_filtered.groupby('PRODUCTLINE')[numeric_cols].sum().reset_index()

        fig = px.treemap(prod_summary,
                         path=['PRODUCTLINE'], values='SALES',
                         color='PROFIT', color_continuous_scale='RdYlGn',
                         title='Treemap: Doanh thu & Profit theo dòng sản phẩm')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
                        **🧠 Khuyến nghị**
                        - Dòng sản phẩm tạo doanh thu lớn và lợi nhuận cao → nên ưu tiên phát triển.
                        - Doanh thu cao nhưng lợi nhuận thấp → cần xem lại chi phí, định giá.
                        - Doanh thu thấp nhưng lợi nhuận tốt → có thể là sản phẩm ngách tiềm năng.
                        - Doanh thu và lợi nhuận thấp → cân nhắc loại bỏ để tiết kiệm nguồn lực.
                         """)
    st.markdown("---")
    #
    # Tabs hiển thị đồ họa
    st.subheader("📊 Đơn hàng & Sản Phẩm")
    c1, c2 = st.columns(2)
    c1.metric("🔢 Tổng số lượng sản phẩm", f"{df_filtered['QUANTITYORDERED'].sum():,.0f}")
    c2.metric("📊 Số lượng TB theo đơn", f"{df_filtered['QUANTITYORDERED'].mean():.2f}")

    tab1, tab2, tab3= st.tabs([
        "🔁 Số lượng đặt hàng",
        "🌍 Đơn giá",
        "📅 Doanh thu theo dòng sản phẩm",
    ])
    with tab1:
        if 'QUANTITYORDERED' in df_filtered.columns:
            fig_qty = px.histogram(df_filtered, x='QUANTITYORDERED', nbins=30, title='Phân phối số lượng đặt hàng')
            st.plotly_chart(fig_qty, use_container_width=True)
    with tab2:
        if 'PRICEEACH' in df_filtered.columns:
            fig_price = px.histogram(df_filtered, x='PRICEEACH', nbins=40, title='Phân phối đơn giá (PRICEEACH)')
            st.plotly_chart(fig_price, use_container_width=True)
    with tab3:
        if 'PRODUCTLINE' in df_filtered.columns and 'SALES' in df_filtered.columns:
            st.subheader("Doanh thu theo dòng sản phẩm")
            prod_sales = df_filtered.groupby('PRODUCTLINE')['SALES'].sum().reset_index().sort_values('SALES', ascending=False)
            # Treemap doanh thu theo dòng sản phẩm
            prod_line_sales = df_filtered.groupby(['PRODUCTLINE', 'PRODUCTCODE'])['SALES'].sum().reset_index()
            fig_tree = px.treemap(prod_line_sales, path=['PRODUCTLINE', 'PRODUCTCODE'], values='SALES',
                                  title='Treemap')
            st.plotly_chart(fig_tree, use_container_width=True)

            # Pareto chart
            prod_sales['cum_sum'] = prod_sales['SALES'].cumsum()
            prod_sales['cum_perc'] = 100 * prod_sales['cum_sum'] / prod_sales['SALES'].sum()
            fig_p = go.Figure()
            fig_p.add_trace(go.Bar(x=prod_sales['PRODUCTLINE'], y=prod_sales['SALES'], name='Doanh thu'))
            fig_p.add_trace(go.Scatter(x=prod_sales['PRODUCTLINE'], y=prod_sales['cum_perc'],
                                       name='Tỷ lệ lũy kế (%)', yaxis='y2', mode='lines+markers'))
            fig_p.update_layout(title='Pareto Chart',
                                yaxis=dict(title='Doanh thu'),
                                yaxis2=dict(title='Tỷ lệ lũy kế (%)', overlaying='y', side='right', range=[0, 100]))
            st.plotly_chart(fig_p, use_container_width=True)
        else:
            st.info("Thiếu cột PRODUCTLINE hoặc SALES.")

    st.markdown("---")
    st.subheader("📊 Khách hàng ")
    tab1, tab2 = st.tabs([
        "🔁 Top 50 Khách hàng",
        "🌍 Top 20 Khách hàng theo quốc gia",
    ])
    with tab1:
        st.subheader("Doanh thu theo khách hàng (Top 50)")
        if 'CUSTOMERNAME' in df_filtered.columns and 'SALES' in df_filtered.columns:
            cust_rev = df_filtered.groupby('CUSTOMERNAME')['SALES'].sum().reset_index().sort_values('SALES',
                                                                                                    ascending=False).head(
                50)
            try:
                fig_tree = px.treemap(cust_rev, path=['CUSTOMERNAME'], values='SALES')
                st.plotly_chart(fig_tree, use_container_width=True)
            except:
                st.dataframe(cust_rev)
    with tab2:
        if 'COUNTRY' in df_filtered.columns and 'CUSTOMERNAME' in df_filtered.columns:
            st.subheader("Số lượng khách hàng theo quốc gia (Top 20)")
            country_cnt = df_filtered.groupby('COUNTRY')['CUSTOMERNAME'].nunique().reset_index(name='Số KH')
            country_cnt = country_cnt.sort_values('Số KH', ascending=False)
            st.bar_chart(country_cnt.set_index('COUNTRY').head(20))

    st.markdown("---")

    # end
    # ---------- Dữ liệu và mô tả ----------
    with st.expander("📊 Thống kê mô tả"):
        desc_cols = ['SALES', 'PRICEEACH', 'QUANTITYORDERED']
        st.dataframe(df_filtered[desc_cols].describe().T)
    with st.expander("📄 Xem dữ liệu đã lọc"):
        st.dataframe(df_filtered.head(300))
    with st.expander("📥 Tải dữ liệu CSV"):
        csv = df_filtered.to_csv(index=False).encode('utf-8')
        st.download_button(label="Tải xuống CSV", data=csv, file_name="describe_filtered.csv", mime="text/csv")


