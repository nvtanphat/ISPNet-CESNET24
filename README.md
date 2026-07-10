<div align="center">

# 🌐 ISPNet-CESNET24: Deep Learning for ISP-Level Network Traffic Forecasting

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-yellowgreen.svg)](https://opensource.org/licenses/Apache-2.0)
[![Jupyter Notebooks](https://img.shields.io/badge/Jupyter-Notebooks-orange.svg)](https://jupyter.org/)

</div>

This repository contains the official implementation of **ISPNet**, a deep learning framework designed for wide-area network traffic forecasting at the Internet Service Provider (ISP) level, evaluated on the real-world **CESNET-TimeSeries24** dataset.

The project models and predicts hourly network traffic (in bytes) across **283 distinct organizations (institutions) and subnets** for two forecasting horizons: **24 hours (daily)** and **168 hours (weekly)**.

---

## 🚀 Key Contributions

* **Mask-aware SMAPE Improvement**: Under our observed-telemetry evaluation, ISPNet reduces SMAPE by **26.61% to 36.00%** relative to the Mean baseline, and by **2.23% to 5.06%** relative to the strongest local deep-learning baseline across the four evaluated configurations.
* **Improvement over the Published CESNET Benchmark**: Against the released results of the deep-learning benchmark paper, ISPNet lowers paper-style normalized RMSE by **14.18% to 16.64%** across the four institution/subnet and 24h/168h configurations.
* **Paper-style Benchmark Metrics**: Under the CESNET benchmark-style normalized evaluation, ISPNet also achieves the lowest RMSE and Harmonic score among the local deep-learning baselines in all four configurations.
* **Superiority over Local DL Baselines**: Outperforms 7 local deep learning baselines: LSTM, GRU, TCN, DLinear, Transformer, PatchTST, and iTransformer.
* **Telemetry Outage Handling**: Integrates a mask-aware noise augmentation mechanism and an optimized Masked Huber Loss to mitigate the impact of missing telemetry data (1.1% missing rate) and abrupt traffic spikes.

---

## ⚙️ Proposed Preprocessing Pipeline

To address key characteristics and anomalies identified in the real-world telemetry data, we implement a custom preprocessing pipeline:

* **Log1p Transformation**: To handle the heavy-tailed distribution of traffic volume (`n_bytes`), a $\log(x + 1)$ transformation is applied to normalize scale and stabilize variance.
* **Per-Entity Robust Scaling**: Each network entity is scaled independently using a `RobustScaler` (based on median and IQR statistics) to handle scale disparities without sensitivity to traffic spikes.
* **Mask-Aware Data Loading**: Telemetry gaps (1.1% overall missing rate) represent sensor inactivity rather than zero traffic. We avoid zero-filling by executing a left-join with the complete hourly time spine, generating an `observed_mask` (1 for observed telemetry, 0 for missing telemetry) to guide both normalization and loss computation.
* **Cyclical Calendar Features**: Hour of day and day of week are encoded using sine and cosine transformations to preserve cyclical proximity (e.g., mapping hour 23 close to hour 0).
* **Telemetry Outage and Gap Features**: An `is_outage` flag identifies the sensor failure window (May 20 – Jun 10, 2024) for robust evaluation, and a `gap_since_last_obs` feature explicitly models telemetry gap duration.

---

## 🧠 Proposed Architecture: ISPNet


Predicting traffic across highly heterogeneous network entities (ranging from small research subnets to national universities) is challenging due to the massive scale variance and differing baseline traffic levels (up to $8.26 \times 10^7$ times difference). **ISPNet** is built on a **GRU Encoder-Decoder** backbone (2 layers, 64 hidden units, 0.4 dropout) and incorporates three core design principles:

1. **Mask-Aware Reversible Instance Normalization (Mask-Aware RevIN)**:
   To combat distribution shift and temporal baseline drifts, the mean ($\mu$) and variance ($\sigma^2$) are dynamically calculated per input window $X = [x_1, x_2, \dots, x_L]$ using **only observed telemetry data** (where the mask $m_t = 1$):
   $$\mu = \frac{\sum_{t=1}^L m_t x_t}{\sum_{t=1}^L m_t + \epsilon}, \quad \sigma^2 = \frac{\sum_{t=1}^L m_t (x_t - \mu)^2}{\sum_{t=1}^L m_t + \epsilon}$$
   where $\epsilon = 10^{-5}$ prevents division by zero. The inputs are then normalized as $\tilde{x}_t = \frac{x_t - \mu}{\sqrt{\sigma^2 + \epsilon}}$ before being fed to the GRU:
   $$h_t = \text{GRU}([\tilde{x}_t, c_t, m_t], h_{t-1})$$
   (where $c_t$ denotes temporal calendar features).

2. **Per-Entity Output Bias ($b_{e,h}$)**:
   To adjust for the extreme scale variations among entities, an entity embedding layer (`nn.Embedding` of size $N_{entities} \times H$, initialized to 0) maps each network entity ID to a step-specific output bias. This is directly added to the denormalized log-scale prediction:
   $$\hat{y}_{t+h}^{\log} = \hat{z}_{t+h}\sqrt{\sigma^2 + \epsilon} + \mu + b_{e,h}$$
   This mechanism maintains a compact backbone size of 88,009 parameters (for institutions) while enabling entity-specific adjustments.

3. **Masked Huber Loss for ISPs**:
   The Huber Loss is configured with $\delta = 1.0$ (corresponding to 1 IQR of the normalized traffic data), applying a quadratic penalty for small errors and a linear penalty for extreme spikes. The loss is computed only at timestamps with valid telemetry data:
   $$\mathcal{L} = \frac{\sum_{h=1}^H m_{t+h} H_\delta(r_{t+h})}{\sum_{h=1}^H m_{t+h} + \epsilon}$$

---

## 📊 Experimental Results

### 1. Mask-Aware SMAPE Metrics
*Comparison between the global **Mean** baseline, the best performing **Local DL Baseline**, and the proposed **ISPNet** under our observed-telemetry evaluation. SMAPE is computed on observed target points only; missing telemetry points are handled through `observed_mask` rather than scored as zero traffic.*

| Dataset | Horizon | Model | SMAPE (Test) | $R^2$ (Test) | SMAPE (Outage) | $R^2$ (Outage) | SMAPE Improvement |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **Institutions** | **24h (Daily)** | Mean baseline<br>Local Best (Transformer)<br>**ISPNet (Ours)** | 95.06%<br>62.23%<br>**60.84%** | -0.110<br>0.242<br>**0.269** | -<br>-<br>**68.09%** | -<br>-<br>**0.228** | **-36.00% vs Mean**<br>**-2.23% vs Local Best** |
| **Institutions** | **168h (Weekly)** | Mean baseline<br>Local Best (Transformer)<br>**ISPNet (Ours)** | 95.06%<br>71.36%<br>**67.75%** | -0.110<br>0.107<br>**0.134** | -<br>-<br>**82.97%** | -<br>-<br>**-0.010** | **-28.73% vs Mean**<br>**-5.06% vs Local Best** |
| **Subnets** | **24h (Daily)** | Mean baseline<br>Local Best (GRU)<br>**ISPNet (Ours)** | 92.44%<br>61.85%<br>**60.15%** | -0.143<br>0.162<br>**0.187** | -<br>-<br>**67.11%** | -<br>-<br>**0.138** | **-34.93% vs Mean**<br>**-2.74% vs Local Best** |
| **Subnets** | **168h (Weekly)** | Mean baseline<br>Local Best (Transformer)<br>**ISPNet (Ours)** | 92.44%<br>71.09%<br>**67.84%** | -0.143<br>0.004<br>**0.052** | -<br>-<br>**83.87%** | -<br>-<br>**-0.125** | **-26.61% vs Mean**<br>**-4.58% vs Local Best** |

### 2. Published Benchmark Paper Comparison
*Head-to-head comparison with the released results of the CESNET deep-learning benchmark paper (`references/isp-forecasting-benchmark/results`). RMSE values are paper-style normalized per-entity medians for the matching windows: 168->24 for 24h forecasting and 744->168 for 168h forecasting. Lower is better.*

| Dataset | Horizon | Published Paper Best | Paper RMSE | ISPNet RMSE | RMSE Reduction |
| :--- | :---: | :--- | :---: | :---: | :---: |
| **Institutions** | **24h (168->24)** | GRU-FCN | 0.0654 | **0.0561** | **14.18%** |
| **Institutions** | **168h (744->168)** | Mean | 0.0729 | **0.0608** | **16.64%** |
| **Subnets** | **24h (168->24)** | GRU-FCN | 0.0623 | **0.0528** | **15.24%** |
| **Subnets** | **168h (744->168)** | Mean | 0.0689 | **0.0580** | **15.83%** |

### 3. Paper-Style Normalized Metrics vs Local Baselines
*CESNET benchmark-style metrics are computed on normalized values and reported as RMSE, $R^2$, and Harmonic score. Lower RMSE/Harmonic and higher $R^2$ are better.*

| Dataset | Horizon | Best Local Baseline | Baseline RMSE | ISPNet RMSE | Baseline $R^2$ | ISPNet $R^2$ | Baseline Harmonic | ISPNet Harmonic |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Institutions** | **24h (Daily)** | Transformer | 0.0602 | **0.0591** | 0.2423 | **0.2685** | 0.1080 | **0.1058** |
| **Institutions** | **168h (Weekly)** | Transformer | 0.0693 | **0.0667** | 0.1074 | **0.1337** | 0.1257 | **0.1207** |
| **Subnets** | **24h (Daily)** | GRU | 0.0669 | **0.0653** | 0.1622 | **0.1873** | 0.1171 | **0.1140** |
| **Subnets** | **168h (Weekly)** | DLinear | 0.0763 | **0.0733** | -0.0005 | **0.0518** | 0.1351 | **0.1295** |

### 4. Ablation Study
*Investigating the contribution of ISPNet components on the Institutions dataset (Horizon: 24h).*

| Variant | Configuration | SMAPE median ↓ (±std) | $R^2$ median ↑ (±std) |
| :--- | :---: | :---: | :---: |
| **A - GRU** | Without RevIN Normalization | 63.13 ± 0.19 | 0.2416 ± 0.0119 |
| **B - GRU + Standard RevIN** | Standard RevIN (Non-mask-aware) | 62.84 ± 0.64 | **0.2704 ± 0.0102** |
| **C - ISPNet** | Proposed: Mask-Aware RevIN + GRU | **62.39 ± 0.33** | 0.2643 ± 0.0138 |

### 5. Telemetry Outage Analysis
*Performance comparisons during normal operations vs. periods of simulated telemetry outages.*

| Dataset | Horizon | SMAPE Normal | SMAPE Outage | Absolute Diff | Remarks |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Institutions** | **24h (Daily)** | 62.39% | 70.26% | +7.87% | Highly stable; error increases by less than 8 percentage points. |
| **Institutions** | **168h (Weekly)** | 70.35% | 85.43% | +15.08% | Expected error increase due to prolonged missing data (1 week). |
| **Subnets** | **24h (Daily)** | 58.81% | 66.29% | +7.49% | High robustness demonstrated at subnet levels. |
| **Subnets** | **168h (Weekly)** | 65.85% | 82.49% | +16.64% | Error increase remains well-bounded for complex scenarios. |

---

## 📂 Project Structure

```
ISPNet-CESNET24/
  ├── requirements.txt                  # Python dependencies
  ├── data/                             # Original raw and preprocessed datasets (ignored by git)
  ├── notebooks/                        # Jupyter notebooks (00 to 07) for full execution pipeline
  └── results/                          # Evaluation and visualization metrics (CSVs & figures)
```

---

## 🛠️ Installation & Usage

### 1. Environment Setup
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/nvtanphat/ISPNet-CESNET24.git
cd ISPNet-CESNET24
pip install -r requirements.txt
```

### 2. Dataset Setup
You can set up the dataset in one of two ways:

* **Option A: Preprocessed Data (Recommended & Faster)**:
  Download the preprocessed dataset from [Kaggle: CESNET-TimeSeries24 Preprocessed](https://www.kaggle.com/datasets/nguynvntnpht/cesnet-timeseries24-preprocessed) and extract its contents into the `data/preprocessed/` directory. You can then skip the preprocessing notebooks and start training immediately.

* **Option B: Original Raw Data**:
  1. Download the official **CESNET-TimeSeries24** raw dataset from the link provided in the [Nature Scientific Data Paper](https://www.nature.com/articles/s41597-025-04603-x).
  2. Extract the raw data files into the `data/raw/` directory.

### 3. Execution Pipeline
Open and run the notebooks in the `notebooks/` directory sequentially:
1. **Preprocessing**: (Skip if using Option A) Run [01_preprocessing_pipeline.ipynb](notebooks/01_preprocessing_pipeline.ipynb) to handle telemetry gaps and generate Parquet feature files.
2. **Classical Baselines**: Run [03_classical_baselines.ipynb](notebooks/03_classical_baselines.ipynb) and [04_sarima_baseline.ipynb](notebooks/04_sarima_baseline.ipynb).
3. **Deep Learning Baselines**: Run the notebooks prefixed with `05_` to train and evaluate the baseline models (LSTM, GRU, TCN, PatchTST, and iTransformer).
4. **ISPNet (Proposed)**: Run the notebooks prefixed with `06_` to train the proposed ISPNet models and generate the experimental comparisons.

---

## 📚 References

* **Official Dataset Paper**: ["CESNET-TimeSeries24: A dataset of network traffic time series at the institution and subnet level"](https://www.nature.com/articles/s41597-025-04603-x) (Scientific Data, 2025).
* **Deep Learning Benchmark Paper**: ["Comparative Analysis of Deep Learning Models for Real-World ISP Network Traffic Forecasting"](https://arxiv.org/abs/2503.17410) (arXiv:2503.17410, 2025).

---

## 📜 Citation

If you use this repository or our findings in your research, please cite:

```bibtex
@thesis{phat2026deep,
  author       = {Nguyễn Văn Tấn Phát},
  title        = {Xây dựng và đánh giá mô hình học sâu cho dự báo chuỗi thời gian lưu lượng mạng ISP},
  school       = {Phân hiệu Trường Đại học Thủy lợi},
  year         = {2026},
  type         = {Course Project},
  url          = {https://github.com/nvtanphat/ISPNet-CESNET24}
}


```

## ✉️ Contact
For questions or collaborations, please reach out to the project authors:
* **Nguyễn Văn Tấn Phát** - Gmail: [nvtanphat69@gmail.com](nvtanphat69@gmail.com)
