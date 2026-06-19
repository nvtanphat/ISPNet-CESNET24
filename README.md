<div align="center">

# 🌐 ISPNet-CESNET24: Học Sâu Dự Báo Lưu Lượng Mạng Cấp ISP (ISP-Level Network Traffic Forecasting)

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-yellowgreen.svg)](https://opensource.org/licenses/Apache-2.0)
[![Jupyter Notebooks](https://img.shields.io/badge/Jupyter-Notebooks-orange.svg)](https://jupyter.org/)

</div>

Repository này chứa mã nguồn triển khai **ISPNet** — một mô hình học sâu (deep learning framework) cho bài toán dự báo lưu lượng băng thông mạng diện rộng của Nhà cung cấp dịch vụ Internet (ISP), được thử nghiệm trên bộ dữ liệu thực tế **CESNET-TimeSeries24**.

Dự án thực hiện mô hình hóa và dự báo lưu lượng băng thông theo giờ (tính bằng bytes) trên **283 tổ chức và mạng con (subnets)** cho cả hai khoảng thời gian dự báo (horizons): **24 giờ** và **168 giờ (1 tuần)**.

---

## 🚀 Các đóng góp chính

* **Giảm từ 30% đến 37% chỉ số SMAPE** so với các mô hình baseline trung bình trong bài báo gốc của bộ dữ liệu.
* **Kết quả SMAPE thấp hơn so với 7 mô hình baseline học sâu**: LSTM, GRU, TCN, DLinear, Transformer, PatchTST, và iTransformer.
* **Xử lý khoảng khuyết telemetry**: Tích hợp cơ chế tăng cường nhiễu có quan sát mặt nạ (mask-aware noise augmentation) và hàm mất mát Masked Huber Loss cải tiến để khắc phục ảnh hưởng từ các khoảng trống mất dữ liệu telemetry (tỷ lệ khuyết 1.1%) và các đỉnh lưu lượng đột biến (traffic spikes).

---

## 🧠 Kiến trúc Mô hình Đề xuất: ISPNet

Dự báo lưu lượng trên nhiều thực thể mạng khác nhau (từ các mạng con nghiên cứu quy mô nhỏ đến các trường đại học quốc gia lớn) gặp khó khăn do sự chênh lệch lớn về quy mô và mức nền lưu lượng (baseline traffic). Phương pháp **ISPNet** được xây dựng dựa trên cấu trúc **GRU Encoder-Decoder** (2 lớp, ẩn số 64, dropout 0.4) và được thiết kế theo các nguyên lý sau:

1. **Chuẩn hóa động theo cửa sổ có xét mặt nạ (Mask-Aware Reversible Instance Normalization - RevIN)**:
   Để chống lại sự trôi dịch phân phối (distribution shift) và biến động mức nền thời gian, thống kê trung bình ($\mu$) và phương sai ($\sigma^2$) được tính động theo từng cửa sổ đầu vào $X = [x_1, x_2, \dots, x_L]$ nhưng **chỉ dựa trên các vị trí có dữ liệu telemetry thực tế** (nơi mặt nạ quan sát $m_t = 1$):
   $$\mu = \frac{\sum_{t=1}^L m_t x_t}{\sum_{t=1}^L m_t + \epsilon}, \quad \sigma^2 = \frac{\sum_{t=1}^L m_t (x_t - \mu)^2}{\sum_{t=1}^L m_t + \epsilon}$$
   Với $\epsilon = 10^{-5}$ để tránh chia cho 0. Dữ liệu sau đó được chuẩn hóa thành $\tilde{x}_t = \frac{x_t - \mu}{\sqrt{\sigma^2 + \epsilon}}$ trước khi đưa vào mô hình GRU:
   $$h_t = \text{GRU}([\tilde{x}_t, c_t, m_t], h_{t-1})$$
   (trong đó $c_t$ là các đặc trưng lịch thời gian).

2. **Thành phần bù độ lệch theo từng thực thể (Per-Entity Output Bias - $b_{e,h}$)**:
   Để bù đắp sự chênh lệch quy mô lưu lượng giữa các thực thể (lên tới $8.26 \times 10^7$ lần), một lớp nhúng (`nn.Embedding` kích thước $N_{entities} \times H$, khởi tạo bằng 0) được sử dụng để ánh xạ ID của thực thể mạng thành một độ lệch cụ thể cho từng bước dự báo $h$, cộng trực tiếp vào đầu ra sau khi đã được giải chuẩn hóa về thang logarit:
   $$\hat{y}_{t+h}^{\log} = \hat{z}_{t+h}\sqrt{\sigma^2 + \epsilon} + \mu + b_{e,h}$$
   Cơ chế này giữ cấu trúc tham số backbone GRU ổn định ở mức 88.009 tham số (ở cấp tổ chức) nhưng vẫn cho phép điều chỉnh dự báo theo từng thực thể.

3. **Hàm mất mát Masked Huber Loss tối ưu cho ISP**:
   Hàm Huber Loss được cấu hình với $\delta = 1.0$ (tương ứng với 1 IQR của dữ liệu lưu lượng sau chuẩn hóa), hoạt động bình phương cho sai số nhỏ và tuyến tính cho các đỉnh lưu lượng đột biến (traffic spikes). Hàm mất mát chỉ được tính toán tại các thời điểm có telemetry thực sự (tránh rò rỉ hoặc bị kéo đạo hàm bởi các điểm khuyết telemetry):
   $$\mathcal{L} = \frac{\sum_{h=1}^H m_{t+h} H_\delta(r_{t+h})}{\sum_{h=1}^H m_{t+h} + \epsilon}$$

---

## 📊 Kết quả Thực nghiệm

Dưới đây là kết quả đối sánh toàn diện của mô hình đề xuất **ISPNet** với các baseline cổ điển, baseline học sâu và nghiên cứu loại bỏ (Ablation Study) dựa trên số liệu thực tế được ghi nhận trong báo cáo đồ án môn học.

### 1. Đối chiếu tổng quan với Paper Benchmark (Global Weighted Metrics)
*(So sánh gộp toàn cục giữa mô hình cơ sở tốt nhất của bài báo gốc, Local DL Baseline tốt nhất và **ISPNet**)*

| Tập dữ liệu | Khoảng dự báo | Mô hình | SMAPE (Test) | $R^2$ (Test) | SMAPE (Outage) | $R^2$ (Outage) | Mức cải thiện SMAPE |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **Institutions** | **24h (Daily)** | Paper Best (Mean)<br>Local Best (Transformer)<br>**ISPNet (Ours)** | 97.46%<br>62.23%<br>**60.84%** | 0.027<br>-3.673<br>**0.251** | -<br>-<br>**68.09%** | -<br>-<br>**0.228** | **-37.58% so với Paper**<br>**-2.23% so với Local Best** |
| **Institutions** | **168h (Weekly)** | Paper Best (Mean)<br>Local Best (Transformer)<br>**ISPNet (Ours)** | 100.49%<br>71.36%<br>**67.75%** | -0.008<br>-0.205<br>**0.128** | -<br>-<br>**82.97%** | -<br>-<br>**-0.112** | **-32.58% so với Paper**<br>**-5.06% so với Local Best** |
| **Subnets** | **24h (Daily)** | Paper Best (Mean)<br>Local Best (GRU)<br>**ISPNet (Ours)** | 93.03%<br>61.85%<br>**60.15%** | 0.047<br>-1949.02<br>**-4621.07** | -<br>-<br>**67.11%** | -<br>-<br>**0.138** | **-35.34% so với Paper**<br>**-2.74% so với Local Best** |
| **Subnets** | **168h (Weekly)** | Paper Best (Mean)<br>Local Best (Transformer)<br>**ISPNet (Ours)** | 97.93%<br>71.09%<br>**67.84%** | -0.001<br>-1481685.49<br>**-24894.48** | -<br>-<br>**83.87%** | -<br>-<br>**-0.380** | **-30.73% so với Paper**<br>**-4.58% so với Local Best** |

### 2. Bảng đối sánh chi tiết theo thực thể (Per-Entity Median over 3 Seeds)
*(Chỉ số Trung vị - Median của các thực thể mạng, ký hiệu ± biểu diễn Độ lệch chuẩn - standard deviation qua 3 seed chạy ngẫu nhiên)*

#### A. Dự báo ngày (Horizon 24h)
* **Institutions (Daily)**
  | Mô hình | SMAPE median ↓ (±std) | $R^2$ median ↑ (±std) | RMSE ↓ (±std) |
  | :--- | :---: | :---: | :---: |
  | LSTM | 63.63 ± 0.20 | 0.2300 ± 0.0044 | 0.0614 ± 0.0003 |
  | GRU | 63.13 ± 0.19 | 0.2416 ± 0.0119 | 0.0605 ± 0.0005 |
  | TCN | 64.32 ± 0.35 | 0.1856 ± 0.0033 | 0.0637 ± 0.0008 |
  | DLinear | 67.05 ± 0.09 | 0.1944 ± 0.0053 | 0.0623 ± 0.0006 |
  | Transformer | 62.87 ± 0.15 | 0.2246 ± 0.0168 | 0.0602 ± 0.0008 |
  | PatchTST | 65.56 ± 0.50 | 0.1949 ± 0.0044 | 0.0631 ± 0.0005 |
  | iTransformer | 65.45 ± 0.42 | 0.2069 ± 0.0027 | 0.0617 ± 0.0001 |
  | **ISPNet (Ours)** | **62.39 ± 0.33** | **0.2643 ± 0.0138** | **0.0591 ± 0.0005** |

* **Subnets (Daily)**
  | Mô hình | SMAPE median ↓ (±std) | $R^2$ median ↑ (±std) | RMSE ↓ (±std) |
  | :--- | :---: | :---: | :---: |
  | LSTM | 61.70 ± 0.55 | 0.0972 ± 0.0083 | 0.0676 ± 0.0006 |
  | GRU | 60.62 ± 0.08 | 0.0954 ± 0.0103 | 0.0669 ± 0.0006 |
  | TCN | 61.70 ± 0.40 | 0.0672 ± 0.0164 | 0.0700 ± 0.0009 |
  | DLinear | 63.20 ± 0.18 | 0.0796 ± 0.0007 | 0.0688 ± 0.0001 |
  | Transformer | 60.49 ± 0.12 | 0.0947 ± 0.0136 | 0.0671 ± 0.0011 |
  | PatchTST | 62.53 ± 0.18 | 0.0685 ± 0.0046 | 0.0688 ± 0.0004 |
  | iTransformer | 61.76 ± 0.25 | 0.0912 ± 0.0062 | 0.0675 ± 0.0002 |
  | **ISPNet (Ours)** | **58.81 ± 0.21** | **0.1146 ± 0.0023** | **0.0653 ± 0.0001** |

#### B. Dự báo tuần (Horizon 168h)
* **Institutions (Weekly)**
  | Mô hình | SMAPE median ↓ (±std) | $R^2$ median ↑ (±std) | RMSE ↓ (±std) |
  | :--- | :---: | :---: | :---: |
  | LSTM | 72.36 ± 0.43 | 0.0984 ± 0.0427 | 0.0718 ± 0.0019 |
  | GRU | 72.19 ± 1.38 | 0.1249 ± 0.0357 | 0.0717 ± 0.0018 |
  | TCN | 72.42 ± 1.24 | 0.1253 ± 0.0095 | 0.0696 ± 0.0009 |
  | DLinear | 73.54 ± 0.83 | 0.0392 ± 0.0017 | 0.0732 ± 0.0005 |
  | Transformer | 70.91 ± 0.27 | 0.1372 ± 0.0195 | 0.0693 ± 0.0011 |
  | PatchTST | 78.56 ± 1.35 | -0.0128 ± 0.0114 | 0.0780 ± 0.0009 |
  | iTransformer | 74.76 ± 0.31 | 0.0254 ± 0.0044 | 0.0725 ± 0.0005 |
  | **ISPNet (Ours)** | **70.35 ± 0.92** | **0.1455 ± 0.0187** | **0.0667 ± 0.0008** |

* **Subnets (Weekly)**
  | Mô hình | SMAPE median ↓ (±std) | $R^2$ median ↑ (±std) | RMSE ↓ (±std) |
  | :--- | :---: | :---: | :---: |
  | LSTM | 67.59 ± 0.27 | 0.0131 ± 0.0040 | 0.0789 ± 0.0005 |
  | GRU | 67.69 ± 0.42 | 0.0102 ± 0.0045 | 0.0794 ± 0.0007 |
  | TCN | 67.50 ± 0.46 | 0.0172 ± 0.0032 | 0.0776 ± 0.0005 |
  | DLinear | 68.46 ± 1.11 | 0.0019 ± 0.0079 | 0.0763 ± 0.0018 |
  | Transformer | 67.60 ± 1.39 | 0.0167 ± 0.0085 | 0.0767 ± 0.0018 |
  | PatchTST | 74.77 ± 0.60 | -0.0217 ± 0.0055 | 0.0830 ± 0.0009 |
  | iTransformer | 69.66 ± 0.27 | -0.0075 ± 0.0014 | 0.0769 ± 0.0003 |
  | **ISPNet (Ours)** | **65.85 ± 0.32** | **0.0224 ± 0.0054** | **0.0733 ± 0.0004** |

### 3. Nghiên cứu loại bỏ (Ablation Study)
*(Thực nghiệm khảo sát sự đóng góp của các thành phần trong ISPNet trên track Tổ chức - Horizon 24h)*

| Biến thể | Thiết lập kiểm chứng | SMAPE median ↓ (±std) | $R^2$ median ↑ (±std) |
| :--- | :---: | :---: | :---: |
| **A - GRU** | Không sử dụng chuẩn hóa RevIN | 63.13 ± 0.19 | 0.2416 ± 0.0119 |
| **B - GRU + Standard RevIN** | Chuẩn hóa động nhưng không xét mặt nạ (non-mask-aware) | 62.84 ± 0.64 | **0.2704 ± 0.0102** |
| **C - ISPNet** | Đề xuất: Mask-Aware RevIN + GRU | **62.39 ± 0.33** | 0.2643 ± 0.0138 |
| **D - ISPNet + FourierKAN** | Mask-Aware RevIN + Fourier KAN projection head | 62.80 ± 0.22 | 0.2569 ± 0.0052 |
| **E - ISPNet + ChebyKAN** | Mask-Aware RevIN + Chebyshev KAN projection head | 62.59 ± 0.10 | 0.2539 ± 0.0008 |

### 4. Đánh giá khả năng chống chịu sự cốTelemetry (Outage Analysis)
*(So sánh sai số giữa thời gian hoạt động bình thường và thời gian xảy ra sự cố khuyết dữ liệu viễn thông)*

| Tập dữ liệu | Khoảng dự báo | SMAPE Normal | SMAPE Outage | Mức tăng lệch điểm % | Nhận định |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Institutions** | **24h (Daily)** | 62.39% | 70.26% | +7.87% | Hoạt động ổn định, sai số tăng nhẹ dưới 8 điểm % |
| **Institutions** | **168h (Weekly)** | 70.35% | 85.43% | +15.08% | Tăng sai số do khuyết dữ liệu kéo dài (1 tuần) |
| **Subnets** | **24h (Daily)** | 58.81% | 66.29% | +7.49% | Chống nhiễu tốt ở các nhánh mạng con |
| **Subnets** | **168h (Weekly)** | 65.85% | 82.49% | +16.64% | Sai số tăng trong tầm kiểm soát ở phân khúc phức tạp |

---

## 📂 Cấu trúc Thư mục

Dự án được tổ chức theo cấu trúc tiêu chuẩn cho các dự án khoa học dữ liệu chuyên nghiệp:

```
ISPNet-CESNET24/
  ├── BaoCaoDoAn_TimeSeries.docx        # Báo cáo đồ án môn học (Bản DOCX)
  ├── BaoCaoDoAn_TimeSeries.pdf         # Báo cáo đồ án môn học (Bản PDF)
  ├── requirements.txt                  # Các thư viện Python cần cài đặt
  ├── data/                             # Thư mục chứa dữ liệu
  │     ├── raw/                        # Dữ liệu thô ban đầu (được bỏ qua trong git)
  │     └── preprocessed/               # Dữ liệu dạng Parquet sau tiền xử lý (được bỏ qua trong git)
  ├── notebooks/                        # Các file Jupyter Notebooks chạy theo luồng
  │     ├── 00_raw_eda.ipynb            # Phân tích khám phá dữ liệu (EDA) trên dữ liệu thô
  │     ├── 01_preprocessing_pipeline.ipynb # Tiền xử lý, chuẩn hóa & trích xuất đặc trưng
  │     ├── 02_processed_eda.ipynb      # EDA trên tập dữ liệu đã qua xử lý
  │     ├── 03_classical_baselines.ipynb # Các mô hình baseline cổ điển (Mean, Median, MA)
  │     ├── 04_sarima_baseline.ipynb    # Dự báo bằng mô hình SARIMA
  │     ├── 05_dl_baselines_daily.ipynb # Học sâu baseline (Dự báo ngày cho Institutions)
  │     ├── 05_dl_baselines_daily_subnets.ipynb # Học sâu baseline (Dự báo ngày cho Subnets)
  │     ├── 05_dl_baselines_weekly.ipynb # Học sâu baseline (Dự báo tuần cho Institutions)
  │     ├── 05_dl_baselines_weekly_subnets.ipynb # Học sâu baseline (Dự báo tuần cho Subnets)
  │     ├── 06_proposed_ispnet_daily.ipynb # Mô hình đề xuất ISPNet (Dự báo ngày cho Institutions)
  │     ├── 06_proposed_ispnet_daily_subnets.ipynb # Mô hình đề xuất ISPNet (Dự báo ngày cho Subnets)
  │     ├── 06_proposed_ispnet_weekly.ipynb # Mô hình đề xuất ISPNet (Dự báo tuần cho Institutions)
  │     ├── 06_proposed_ispnet_weekly_subnets.ipynb # Mô hình đề xuất ISPNet (Dự báo tuần cho Subnets)
  │     └── 07_ablation_ispnet_daily.ipynb # Ablation study khảo sát các thành phần của ISPNet
  └── results/                          # Thư mục lưu kết quả đầu ra
        ├── baselines/                  # Kết quả và biểu đồ của baseline cổ điển
        ├── eda_figures/                # Các biểu đồ trực quan hóa từ các file EDA
        ├── extra_analysis/             # Các phân tích bổ sung và kiểm định thống kê
        ├── daily/                      # Kết quả dự báo hàng ngày (file CSV & biểu đồ)
        │     ├── baselines/            # Metrics của các mô hình học sâu baseline
        │     ├── baselines_subnets/    # Metrics của các subnet baseline
        │     ├── proposed/             # Metrics của ISPNet Daily đề xuất
        │     └── proposed_subnets/     # Metrics của ISPNet Daily Subnets đề xuất
        └── weekly/                     # Kết quả dự báo hàng tuần (file CSV & biểu đồ)
              ├── baselines/            # Metrics của các mô hình học sâu baseline
              ├── baselines_subnets/    # Metrics của các subnet baseline
              ├── proposed/             # Metrics của ISPNet Weekly đề xuất
              └── proposed_subnets/     # Metrics của ISPNet Weekly Subnets đề xuất
```

---

## 🛠️ Hướng dẫn Cài đặt & Sử dụng

### 1. Cài đặt môi trường
Sao chép repository này về máy và cài đặt các thư viện cần thiết:
```bash
git clone https://github.com/nvtanphat/ISPNet-CESNET24.git
cd ISPNet-CESNET24
pip install -r requirements.txt
```

### 2. Thiết lập tập dữ liệu
Bạn có thể lựa chọn 1 trong 2 cách sau:

* **Cách A: Sử dụng dữ liệu đã tiền xử lý sẵn (Khuyên dùng & Nhanh hơn)**:
  Tải bộ dữ liệu đã được tiền xử lý tại [Kaggle: CESNET-TimeSeries24 Preprocessed](https://www.kaggle.com/datasets/nguynvntnpht/cesnet-timeseries24-preprocessed) và giải nén toàn bộ nội dung vào thư mục `data/preprocessed/`. Bạn sẽ không cần chạy notebook tiền xử lý nữa mà có thể bắt đầu huấn luyện mô hình ngay.

* **Cách B: Sử dụng dữ liệu thô ban đầu**:
  1. Tải bộ dữ liệu thô **CESNET-TimeSeries24** chính thức từ liên kết trong bài báo [Nature Scientific Data Paper](https://www.nature.com/articles/s41597-025-04603-x).
  2. Giải nén dữ liệu thô vào thư mục `data/raw/`.

### 3. Quy trình thực hiện
Mở và chạy tuần tự các notebook trong thư mục `notebooks/`:
1. **Tiền xử lý (Preprocessing)**: (Bỏ qua nếu chọn Cách A) Chạy file [01_preprocessing_pipeline.ipynb](notebooks/01_preprocessing_pipeline.ipynb) để xử lý các điểm khuyết telemetry và tạo các tệp Parquet đặc trưng.
2. **Classical Baselines**: Chạy [03_classical_baselines.ipynb](notebooks/03_classical_baselines.ipynb) và [04_sarima_baseline.ipynb](notebooks/04_sarima_baseline.ipynb).
3. **Deep Learning Baselines**: Chạy các file bắt đầu bằng `05_` để huấn luyện và đánh giá các mô hình LSTM, GRU, TCN, PatchTST và iTransformer.
4. **ISPNet (Mô hình Đề xuất)**: Chạy các file bắt đầu bằng `06_` để huấn luyện mô hình ISPNet và xuất bảng so sánh kết quả thực nghiệm.

---

## 📚 Tài liệu Tham khảo

* **Bài báo chính thức của bộ dữ liệu**: ["CESNET-TimeSeries24: A dataset of network traffic time series at the institution and subnet level"](https://www.nature.com/articles/s41597-025-04603-x) (Scientific Data, 2025).
* **Bài báo benchmark học sâu**: ["Comparative Analysis of Deep Learning Models for Real-World ISP Network Traffic Forecasting"](https://arxiv.org/abs/2503.17410) (arXiv:2503.17410, 2025).

---

## 📜 Trích dẫn (Citation)

Nếu bạn sử dụng repository này hoặc các kết quả từ báo cáo đồ án môn học này phục vụ cho nghiên cứu của mình, vui lòng trích dẫn theo định dạng sau:

```bibtex
@thesis{phatvunguyen2026deep,
  author       = {Nguyễn Văn Tấn Phát and Đoàn Anh Vũ and Nguyễn Hồng Nguyên},
  title        = {Xây dựng và đánh giá mô hình học sâu cho dự báo chuỗi thời gian lưu lượng mạng ISP},
  school       = {Phân hiệu Trường Đại học Thủy lợi},
  year         = {2026},
  type         = {Đồ án môn học},
  url          = {https://github.com/nvtanphat/ISPNet-CESNET24}
}
```

## ✉️ Liên hệ

Mọi thắc mắc hoặc nhu cầu hợp tác nghiên cứu vui lòng liên hệ các tác giả của đồ án:
* **Nguyễn Văn Tấn Phát** - GitHub: [@nvtanphat](https://github.com/nvtanphat)
* **Đoàn Anh Vũ**
* **Nguyễn Hồng Nguyên**
* **Giảng viên hướng dẫn**: TS. Hoàng Văn Quý
