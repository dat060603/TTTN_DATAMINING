
import streamlit as st
from streamlit_option_menu import option_menu
from components.data_loader import load_data
# def local_css(file_name):
#     with open(file_name) as f:
#         st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
# local_css("style.css")
# Cấu hình trang
st.set_page_config(page_title="App phân tích bán hàng", page_icon="📊", layout="wide")


def local_css(file_name):
    with open(file_name, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
local_css("style.css")
# Sidebar custom menu
with st.sidebar:
    selected = option_menu(
        menu_title="",
        options=[
            "🏠 Home",
            "📈 Dashboard",
            "🔍 Data Reason",
            "💡 Optimize",
            "📅 Forecast",
            "🎯 Simulate",
            # "Tối ưu vận chuyển",
            # "Tối ưu chi phi"
        ],
        icons=["house", "bar-chart", "file-text", "search", "lightbulb", "calendar", "rocket"],
        default_index=0
    )

# Điều hướng đến các trang tương ứng
if selected == "🏠 Home":
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
    st.markdown(
        """
        <div class="recommendation-box">
            <h3>🔽 Các chức năng chính:</h3>
            <ul>
                <li><b>Dashboard (Visualize):</b> Hiển thị trực quan doanh thu theo tháng, quốc gia, khách hàng,...</li>
                <li><b>Describe:</b> Mô tả đặc điểm cơ bản của dữ liệu.</li>
                <li><b>Data Reason:</b> Tìm hiểu nguyên nhân doanh thu thấp hoặc bất thường.</li>
                <li><b>Optimize:</b> Tối ưu danh mục sản phẩm theo doanh thu và lợi nhuận.</li>
                <li><b>Forecast:</b> Dự báo doanh thu theo thời gian.</li>
                <li><b>Simulate:</b> Mô phỏng What-if để hỗ trợ ra quyết định.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

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