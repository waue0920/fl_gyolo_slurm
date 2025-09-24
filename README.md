
# Gyolo 聯邦式學習框架 (FL-GYOLO-SLURM)

一個基於 GeneralistYOLO 的聯邦式學習 (Federated Learning) 系統，專為 HPC/Slurm/Singularity 環境設計，支援多種聚合演算法與自動化分散式訓練。

---

## 目錄
- [系統概述](#系統概述)
- [環境需求](#環境需求)
- [安裝與設定](#安裝與設定)
- [目錄結構](#目錄結構)
- [快速開始](#快速開始)
- [模型驗證與監控](#模型驗證與監控)


---

## 系統概述
本專案實現 GeneralistYOLO 聯邦式學習全流程，支援多 client 分散式訓練、server 聚合、Slurm 任務排程、Singularity 容器化運行，並可彈性切換多種聚合演算法（FedAvg、FedProx、FedOpt、FedNova、FedAvgM、FedAWA）。

- 支援初始權重自訂、全零初始化
- 可自動化多輪訓練與聚合
- 適用於國網中心 TWCC/N5 等 HPC 叢集

---

## 環境需求
- **作業系統**：Linux (國網中心 TWCC/N5 )
- **作業調度器**：Slurm (國網中心 TWCC/N5 )
- **容器引擎**：Singularity
- **Python**：≥ 3.10
- **PyTorch**：≥ 2.1.0
- **GPU**：NVIDIA GPU (V100/H100)
- **wandb**：≥ 0.18.7

---

## 安裝與設定
1. 取得專案與子模組
```bash
git clone <repository-url>
cd fl_gyolo_slurm
```
2. 準備必要檔案
	- Singularity 映像檔：`gyolo_ngc2306.sif`
	- 初始權重（可選）：`gyolo.pt` 或空字串
3. 準備資料集與分割 yaml
	- 將原始資料集放於指定目錄
	- 用 `federated_data/` 存放分割後的 client yaml
4. 編輯 `src/env.sh` 設定專案路徑與超參數

---

## 目錄結構
```
.
├── README.md
├── gyolo/                  # YOLOv9 原始碼與模型
├── src/                    # 主要腳本 (client_train.sh, server_fedagg.py, ...)
├── experiments/            # 實驗結果與 Slurm 輸出
├── federated_data/         # 分割後的 client 資料集 yaml
├── slurm/                  # Slurm 腳本與環境設定
├── gyolo_ngc2306.sif       # Singularity 容器
├── gyolo.pt                # 初始權重 (可選)
└── ...
```

---

## 快速開始
### 全自動模式
1. 編輯 src/env.sh
2. 登入國網中心 HPC環境的 Login 節點
```bash
# 以 4 client、3 輪為例
sbatch src/run.sb
```
- 支援多輪自動化流程

---

## 聚合演算法支援
- FedAvg
- FedProx
- FedOpt
- FedNova
- FedAvgM
- FedAWA

---

## 使用說明
1. 編輯 `src/env.sh` 設定專案路徑、超參數、初始權重
2. 用 `src/fl_client_train.sh` 提交所有 client 訓練任務
3. 用 `src/server_fedagg.py` 執行 server 聚合
4. 可重複多輪訓練與聚合

--
## 模型驗證與監控
- 支援 wandb 實驗追蹤
- 可用 `readme_val.md` 進行模型驗證
- Slurm 輸出與 log 於 `experiments/` 目錄

---

## 注意事項
- `INITIAL_WEIGHTS` 可設為空字串
- 其餘演算法直接用 client 權重，不需額外指定初始值
- 請確保所有路徑、Slurm/Singularity 設定正確
- 若需自訂聚合邏輯，請修改 `aggregation/` 相關程式

---

**最後更新**：2025-09-24  
**維護者**：nchc/waue0920
