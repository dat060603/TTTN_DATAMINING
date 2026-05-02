# 📊 TTTN_DATAMINING - Business Data Analysis & Mining App

## 📌 Giới thiệu

Dự án xây dựng một ứng dụng phân tích dữ liệu kinh doanh sử dụng **Streamlit**, hỗ trợ:

* 📈 Dashboard trực quan hóa dữ liệu
* 📊 Phân tích mô tả (Descriptive Analytics)
* 🔍 Phân tích mối quan hệ dữ liệu (Data Reasoning)
* ⚙️ Tối ưu hóa (Optimization)
* 🔮 Dự báo (Forecasting)
* 🧪 Mô phỏng & What-if Analysis

Ứng dụng giúp doanh nghiệp đưa ra quyết định dựa trên dữ liệu.

---

## 🛠️ Công nghệ sử dụng

* Python
* Streamlit
* Pandas, NumPy
* Scikit-learn
* Prophet
* Plotly

---

## 📁 Cấu trúc project

```
TTTN_DATAMINING/
│
├── app.py / main.py        # File chạy chính
├── pages/                 # Các module chức năng
│   ├── Dashboard.py
│   ├── Describe.py
│   ├── Data_Reason.py
│   ├── Optimize.py
│   ├── Forecast.py
│   └── Simulate.py
│
├── data/                  # Dataset
├── assets/                # CSS, hình ảnh
├── requirements.txt
└── README.md
```

---

## ⚙️ Cài đặt môi trường

### 1. Clone repository

```bash
git clone https://github.com/dat060603/TTTN_DATAMINING.git
cd TTTN_DATAMINING
```

---

### 2. Tạo môi trường ảo (khuyến nghị)

```bash
python -m venv venv
```

Kích hoạt:

* Windows:

```bash
venv\Scripts\activate
```

* Mac/Linux:

```bash
source venv/bin/activate
```

---

### 3. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

---

## ▶️ Chạy ứng dụng

```bash
streamlit run app.py
```

hoặc:

```bash
python -m streamlit run app.py
```

---

## 🌐 Truy cập ứng dụng

Sau khi chạy thành công, mở trình duyệt:

```
http://localhost:8501
```

---

## ⚠️ Lưu ý

* Đảm bảo đã cài Python >= 3.8
* Nếu lỗi thiếu thư viện → chạy lại:

```bash
pip install -r requirements.txt
```

* Nếu port bị trùng:

```bash
streamlit run app.py --server.port 8502
```

---

## 📊 Dataset

Dự án sử dụng dataset:

* `sales_data_sample.csv`

Bao gồm các thông tin:

* Doanh thu (SALES)
* Sản phẩm (PRODUCTLINE)
* Khách hàng (CUSTOMERNAME)
* Thời gian (ORDERDATE)
* ...

---

## 🚀 Tính năng nổi bật

* Dashboard tương tác (Plotly)
* Phân tích tương quan & hồi quy
* Dự báo doanh thu (Linear Regression, Prophet)
* Tối ưu danh mục sản phẩm
* Mô phỏng kịch bản kinh doanh

---

## 👨‍💻 Tác giả

* Sinh viên thực tập tốt nghiệp ngành CNTT Triệu Quốc Đạt - N21DCCN016 
* Chủ đề: Data Mining & Business Intelligence
Tháng 8/2025 
---

## 📌 Hướng phát triển

* Tích hợp AI Chatbot hỗ trợ phân tích
* Deploy lên cloud (Streamlit Cloud / AWS)
* Kết nối database realtime

---
