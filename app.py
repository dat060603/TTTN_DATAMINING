
import streamlit as st
from streamlit_option_menu import option_menu
from components.data_loader import load_data
# def local_css(file_name):
#     with open(file_name) as f:
#         st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
# local_css("style.css")
# Cấu hình trang
st.set_page_config(page_title="App phân tích bán hàng", page_icon="📊", layout="wide")

# Sidebar custom menu
with st.sidebar:
    selected = option_menu(
        menu_title="",
        options=[
            "🏠 Trang chủ",
            "📈 Dashboard",
            "🔍 Data Reason",
            "💡 Optimize",
            "📅 Forecast",
            "🎯 Simulate",
            "Tối ưu vận chuyển",
            "Tối ưu chi phi"
        ],
        icons=["house", "bar-chart", "file-text", "search", "lightbulb", "calendar", "rocket"],
        default_index=0
    )

# Điều hướng đến các trang tương ứng
if selected == "🏠 Trang chủ":
    import streamlit as st
    from components.data_loader import load_data

    # Cấu hình trang
    st.set_page_config(
        page_title="App phân tích bán hàng",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title("📊 Ứng dụng Phân tích và Khai thác Dữ liệu Doanh nghiệp")
    st.markdown('<div class="fade-in"><h3>🚀 Chào mừng bạn đến với ứng dụng phân tích bán hàng!</h3></div>',
                unsafe_allow_html=True)
    st.markdown("""
    🔽 **Các chức năng chính:**
    - **Dashboard (Visualize):** Hiển thị trực quan doanh thu theo tháng, quốc gia, khách hàng,...
    - **Describe:** Mô tả đặc điểm cơ bản của dữ liệu.
    - **Data Reason:** Tìm hiểu nguyên nhân doanh thu thấp hoặc bất thường.
    - **Optimize:** Tối ưu danh mục sản phẩm theo doanh thu và lợi nhuận.
    - **Forecast:** Dự báo doanh thu theo thời gian.
    - **Simulate:** Mô phỏng What-if để hỗ trợ ra quyết định.
    """)
    # Tải và hiển thị dữ liệu mẫu
    df = load_data()
    with st.expander("📄 Xem trước dữ liệu"):
        st.dataframe(df.head())
    st.markdown("📌 Chọn từng mục trên menu để khám phá các chức năng!")
elif selected == "📈 Dashboard":
    import pages.Dashboard as dashboard
    dashboard.app()

# elif selected == "📝 Describe":
#     import pages.Describe as describe
#     describe.app()

elif selected == "🔍 Data Reason":
    import pages.Data_Reason as data_reason
    data_reason.app()

elif selected == "💡 Optimize":
    import pages.Optimize as optimize
    optimize.app()

elif selected == "📅 Forecast":
    import pages.Forecast as forecast
    forecast.app()

elif selected == "🎯 Simulate":
    import pages.Simulate as simulate
    simulate.app()
elif selected =="Tối ưu vận chuyển":
    import pages.toiuugiaohang as giaohang
    giaohang.app()
elif selected =="Tối ưu chi phi":
    import pages.toiuuchiphi as chiphi
    chiphi.app()