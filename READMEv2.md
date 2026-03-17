# IMPERANDI — CT Imaging Preprocessing Pipeline (DICOM → NIfTI → Radiomics)

**IMPERANDI** is a Python framework and CLI for building **analysis-ready CT imaging datasets** from raw hospital data.

It provides an **end-to-end pipeline** from messy DICOM archives to structured datasets ready for **machine learning, radiomics, and clinical research**.

---

## 🚀 Key Features

* 📥 **DICOM ingestion & cohort building**
  Parse heterogeneous datasets into clean, structured metadata tables

* 🔄 **DICOM → NIfTI conversion**
  Standardize imaging volumes for downstream processing

* 🧠 **Organ & tumor segmentation**
  Integration with TotalSegmentator for automated anatomical extraction

* 🌗 **Contrast phase detection**
  Handle multi-phase CT (arterial, portal, delayed)

* 📊 **Radiomics feature extraction**
  Built-in support for PyRadiomics pipelines

* ⚙️ **CLI-first design**
  Reproducible, scriptable workflows for large-scale datasets

* 🧩 **Modular & extensible**
  Designed to integrate with custom pipelines and ML workflows

---

## 🧠 Why IMPERANDI?

Medical imaging pipelines are often fragmented:

* DICOM parsing → custom scripts
* Conversion → separate tools
* Segmentation → standalone models
* Radiomics → another pipeline

**IMPERANDI unifies everything into a single, consistent framework.**

👉 Think of it as the **missing glue between raw DICOM data and machine learning models**.

---

## 🏥 Use Cases

* Liver cancer imaging pipelines
* Longitudinal CT cohort analysis
* Multi-phase CT studies
* Radiomics-based prognosis modeling
* Clinical dataset curation from PACS exports

---

## 🧱 Pipeline Overview

```
DICOM → Parse → Clean → Convert → Segment → Radiomics → CSV / ML-ready dataset
```

---

## ⚡ Quick Start

### Installation

```bash
git clone https://github.com/dmandache/IMPERANDI.git
cd IMPERANDI
pip install -e .
```

---

### 1. Ingest DICOM data

```bash
imperandi ingest \
  --root_path ./dicom_data \
  --output_dir ./output
```

👉 Generates:

* structured metadata CSV
* cleaned dataset index

---

### 2. Convert to NIfTI

```bash
imperandi convert \
  --csv_path ./output/dicom_index_clean.csv
```

---

### 3. Run segmentation

```bash
imperandi segment \
  --input_csv ./output/nifti_index.csv
```

---

### 4. Extract radiomics

```bash
imperandi radiomics \
  --input_csv ./output/nifti_index_segmented.csv
```

---

## ⚙️ Design Principles

* **Reproducibility first**
  All steps are traceable and configurable

* **Real-world robustness**
  Designed for messy hospital data (missing tags, inconsistencies)

* **Scalability**
  Supports multiprocessing and large datasets

* **Separation of concerns**
  Ingestion, processing, and modeling are clearly decoupled

---

## 🔬 Comparison

| Feature                | IMPERANDI | MONAI | nnU-Net |
| ---------------------- | --------- | ----- | ------- |
| DICOM ingestion        | ✅         | ❌     | ❌       |
| Cohort building        | ✅         | ❌     | ❌       |
| Radiomics pipeline     | ✅         | ❌     | ❌       |
| End-to-end CLI         | ✅         | ⚠️    | ❌       |
| Hospital data handling | ✅         | ⚠️    | ❌       |

---

## 🧩 Integrations

* PyRadiomics
* TotalSegmentator
* PyTorch / TorchIO pipelines
* Custom ML workflows

---

## 📊 Output

IMPERANDI produces:

* Clean cohort CSVs
* NIfTI volumes
* Segmentation masks
* Radiomics feature tables

👉 Ready for:

* Machine learning
* Statistical analysis
* Clinical research

---

## 🛠️ Roadmap

* [x] Visualization tools
* [ ] Full documentation (docs/)
* [ ] Configuration system improvements
* [ ] Dataset versioning
* [ ] Web dashboard (optional)

---

## 🤝 Contributing

Contributions are welcome.

If you are working on:

* medical imaging
* radiomics
* clinical ML pipelines

feel free to open issues or PRs.

---

## 📜 License

Apache 2.0 License

---

## ⭐ Support

If this project helps you, consider giving it a star ⭐
It helps visibility and future development.

---

## 🧭 Project Context & Funding

**IMPERANDI** is developed within the context of the **[RHU OPERANDI project](https://rhu-operandi.com)**, a large-scale French University Hospital Research (RHU) initiative focused on improving the management of digestive cancers using **medical imaging, AI, and imagomics**.

The OPERANDI project aims to better **select, treat, and monitor patients** by leveraging imaging-based biomarkers and machine learning approaches.

The project is coordinated by AP-HP / Université Paris Cité and supported by the **French National Research Agency (ANR)** under the France 2030 investment plan.

👉 IMPERANDI focuses on the **data engineering layer**, transforming raw clinical imaging data (DICOM) into standardized datasets usable for radiomics and machine learning.

---

## ⚠️ Disclaimer

This project is intended for **research purposes only** and is not a certified medical device.

---

