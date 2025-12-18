# Synthetic-EHR-Audit-Dataset
A synthetic dataset simulating Electronic Health Record (EHR) access audit logs for anomaly detection research in healthcare cybersecurity. This dataset enables the development and evaluation of machine learning models for detecting unauthorized or suspicious access patterns in healthcare systems.



# Synthetic EHR Audit Dataset

[![Dataset](https://img.shields.io/badge/Records-120%2C000-blue)](.)
[![Features](https://img.shields.io/badge/Features-19-green)](.)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

A synthetic dataset simulating Electronic Health Record (EHR) access audit logs for anomaly detection research in healthcare cybersecurity. This dataset enables the development and evaluation of machine learning models for detecting unauthorized or suspicious access patterns in healthcare systems.

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset Description](#dataset-description)
- [Feature Dictionary](#feature-dictionary)
- [Data Statistics](#data-statistics)
- [Use Cases](#use-cases)
- [Getting Started](#getting-started)
- [Citation](#citation)
- [License](#license)

## Overview

Healthcare organizations face increasing cybersecurity threats, making the detection of anomalous access patterns in EHR systems critical. This synthetic dataset provides realistic audit log data that can be used to train and evaluate anomaly detection models without compromising patient privacy.

**Key Characteristics:**
- **120,000** audit log records
- **19** features covering user behavior, access patterns, and system metrics
- **Balanced** anomaly labels (~50% normal, ~50% anomalous)
- **No missing values** - clean, ready-to-use dataset
- **FHIR-aligned** resource types (Patient, Observation, Encounter, etc.)

## Dataset Description

| Property | Value |
|----------|-------|
| Total Records | 120,000 |
| Total Features | 19 |
| File Format | CSV |
| File Size | ~15 MB |
| Missing Values | None |
| Target Variable | `is_anom` (binary) |

### Anomaly Distribution

| Class | Count | Percentage |
|-------|-------|------------|
| Normal (0) | 59,939 | 49.9% |
| Anomalous (1) | 60,061 | 50.1% |

## Feature Dictionary

### Temporal Features

| Feature | Type | Description |
|---------|------|-------------|
| `ts` | Integer | Unix timestamp of the access event |
| `hour` | Integer | Hour of day (0-23) when access occurred |

### User & Session Features

| Feature | Type | Description | Unique Values |
|---------|------|-------------|---------------|
| `user_id` | String | Anonymized user identifier | 5,000 |
| `session_id` | String | Unique session identifier | 119,982 |
| `role` | Categorical | User's role in the healthcare system | 6 |
| `ip_sig` | String | Hashed IP signature for privacy | 253 |

**Role Distribution:**
- Doctor (33,430)
- Nurse (31,230)
- LabTech (18,076)
- Radiology (14,176)
- Billing (12,166)
- Admin (10,922)

### Access Pattern Features

| Feature | Type | Description | Values |
|---------|------|-------------|--------|
| `rtype` | Categorical | FHIR resource type being accessed | 6 types |
| `method` | Categorical | HTTP method / operation type | READ, WRITE, SEARCH |
| `access_reason` | Categorical | Stated reason for access | 5 reasons |
| `is_emergency` | Binary | Emergency access flag | 0 or 1 |

**Resource Types (FHIR-aligned):**
- Observation (38,277)
- Encounter (26,589)
- MedicationRequest (16,852)
- Condition (14,393)
- Patient (12,058)
- ImagingStudy (11,831)

**Access Reasons:**
- Treatment (65,711)
- Operations (24,296)
- Research (12,047)
- Billing (11,957)
- Audit (5,989)

### Device & Location Features

| Feature | Type | Description | Values |
|---------|------|-------------|--------|
| `geo_region` | Categorical | Geographic region of access | 5 regions |
| `device_type` | Categorical | Type of device used | 5 types |
| `auth_strength` | Categorical | Authentication method strength | 3 levels |

**Geographic Regions:**
- IN-South (31,122)
- IN-North (26,451)
- IN-West (21,647)
- IN-East (21,560)
- IN-Central (19,220)

**Device Types:**
- Desktop (41,940)
- Laptop (36,041)
- Mobile (23,921)
- Tablet (12,152)
- Kiosk (5,946)

**Authentication Strength:**
- Password (65,925)
- 2FA (42,006)
- Biometric (12,069)

### Behavioral Metrics

| Feature | Type | Description | Range |
|---------|------|-------------|-------|
| `req_rate_10m` | Integer | Request rate in last 10 minutes | 0-42 |
| `failures_10m` | Integer | Failed requests in last 10 minutes | 0-9 |
| `response_time_ms` | Integer | Server response time (milliseconds) | 0-480 |
| `bytes_sent` | Integer | Bytes sent in response | 0-101,505 |
| `bytes_received` | Integer | Bytes received in request | 0-133,967 |

### Target Variable

| Feature | Type | Description |
|---------|------|-------------|
| `is_anom` | Binary | Anomaly label (0 = Normal, 1 = Anomalous) |

## Data Statistics

### Numerical Feature Summary

| Feature | Mean | Std | Min | 25% | 50% | 75% | Max |
|---------|------|-----|-----|-----|-----|-----|-----|
| hour | 11.50 | 6.92 | 0 | 6 | 12 | 17 | 23 |
| req_rate_10m | 12.51 | 8.29 | 0 | 5 | 11 | 20 | 42 |
| failures_10m | 1.50 | 1.58 | 0 | 0 | 1 | 3 | 9 |
| response_time_ms | 140.5 | 71.2 | 0 | 85 | 135 | 190 | 480 |
| bytes_sent | 33,148 | 17,048 | 0 | 19,468 | 32,487 | 45,542 | 101,505 |
| bytes_received | 9,799 | 17,807 | 0 | 2,704 | 4,632 | 6,572 | 133,967 |

## Use Cases

This dataset is suitable for:

1. **Anomaly Detection in Healthcare Systems**
   - Supervised classification models
   - Unsupervised anomaly detection
   - Semi-supervised learning approaches

2. **User Behavior Analytics (UBA)**
   - Access pattern analysis
   - Role-based access monitoring
   - Session behavior profiling

3. **Healthcare Cybersecurity Research**
   - Insider threat detection
   - Unauthorized access identification
   - Compliance monitoring (HIPAA)

4. **Machine Learning Benchmarking**
   - Binary classification algorithms
   - Feature engineering experiments
   - Model comparison studies

## Getting Started

### Loading the Dataset

```python
import pandas as pd

# Load the dataset
df = pd.read_csv('synthetic_ehr_audit_dataset.csv')

# Basic exploration
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(df.head())
```

### Quick Preprocessing

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Separate features and target
X = df.drop('is_anom', axis=1)
y = df['is_anom']

# Encode categorical variables
categorical_cols = ['role', 'rtype', 'method', 'geo_region', 
                    'device_type', 'auth_strength', 'access_reason']

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

### Example: Random Forest Classifier

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Select numeric features for quick demo
feature_cols = ['hour', 'req_rate_10m', 'failures_10m', 
                'response_time_ms', 'bytes_sent', 'bytes_received']

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train[feature_cols], y_train)

# Evaluate
y_pred = clf.predict(X_test[feature_cols])
print(classification_report(y_test, y_pred))
```

## File Structure

```
.
├── README.md
├── synthetic_ehr_audit_dataset.csv
├── LICENSE
└── examples/
    └── basic_analysis.ipynb (optional)
```

## Authors

**Nagaraj S<sup>1</sup> and Vijayarajan V<sup>1*</sup>**

<sup>1</sup>School of Computer Science and Engineering, Vellore Institute of Technology (VIT), Vellore, India.

*Corresponding author

## Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{synthetic_ehr_audit_2025,
  title        = {Synthetic EHR Audit Dataset for Anomaly Detection},
  author       = {Nagaraj, S. and Vijayarajan, V.},
  year         = {2025},
  institution  = {Vellore Institute of Technology (VIT)},
  address      = {Vellore, India},
  publisher    = {GitHub},
  url          = {https://github.com/[username]/synthetic-ehr-audit-dataset}
}
```

## License

This dataset is released under the [MIT License](LICENSE).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Dataset generated for healthcare cybersecurity research
- Inspired by real-world EHR audit log structures
- FHIR resource types aligned with HL7 FHIR R4 specification

---

**Disclaimer:** This is a synthetic dataset generated for research purposes. It does not contain any real patient data or actual healthcare system logs.
