# TimeTabDiff

TimeTabDiff는 Relational Time-Series Tabular Data 생성을 위한 diffusion 기반 모델 구현이다.  
기존 TabDiff 계열 구조를 확장하여, 시계열/시간 조건을 잘 다루는 것을 목적으로 한다.

---

## Project Structure

```
TimeTabDiff/
├── tabdiff.yaml
├── timetabdiff/
│   ├── main.ipynb                # 학습, 생성, 평가 노트북
│   ├── data/
│   │   └── rossmann_subsampled/  # Example dataset
│   │       ├── historical.csv    # Child Table
│   │       ├── store.csv         # Parent Table
│   │       └── metadata.json     # SDV 메타 데이터
│   ├── tmodels/
│   │   ├── diffusion.py          # TUnifiedCtimeDiffusion
│   │   ├── noise_schedule.py     # PowerMeanNoise, LogLinearNoise
│   │   ├── transformer.py        # Tokenizer, Transformer, Reconstructor
│   │   └── tmodules.py           # TUniModMLP, Precond
├── utils/
│   └── datasets.py               # TimeTabDiffDataset
└── .gitignore
```

---

### Model Components

```
TimeTabDiffDataset
├── Parent Table Processing
│   ├── Numerical features → MinMaxScaler
│   └── Categorical features → Label Encoding
│
└── Child Table Processing (Time-series)
    ├── Numerical features → MinMaxScaler
    ├── Categorical features → Label Encoding
    └── Sequence Grouping by Foreign Key

TUniModMLP (Backbone)
├── Tokenizer
├── RoPE
├── Transformer Encoder
├── MLP + Timestep Injection
├── Transformer Decoder
└── Reconstructor

TUnifiedCtimeDiffusion
├── Continuous-time Noise Schedules
│   ├── PowerMeanNoise (numerical)
│   └── LogLinearNoise (categorical)
├── EDM Preconditioning
├── Mixed Loss (continuous + discrete)
└── Sampling with Denoising
```


## Quick Start (main.ipynb)

### Training

```python
import torch
from torch.utils.data import DataLoader
from utils.datasets import TimeTabDiffDataset
from tmodels.diffusion import TUnifiedCtimeDiffusion
from tmodels.tmodules import TUniModMLP

# Load dataset
dataset = TimeTabDiffDataset(
    data_dir='./data/rossmann_subsampled',
    tablename='historical',
    is_train=True,
    seq_len=62
)

# Get dataset info
info = dataset.get_info()
num_numerical = info['num_numerical_features']
num_classes = info['num_classes']

# Create model
backbone = TUniModMLP(
    d_numerical=num_numerical,
    categories=[k + 1 for k in num_classes],
    num_layers=2,
    d_token=4,
    n_heads=1,
    factor=32,
    bias=True,
    dim_t=1024,
    use_mlp=True
).to('cuda')

diffusion = TUnifiedCtimeDiffusion(
    num_classes=num_classes,
    num_numerical_features=num_numerical,
    denoise_fn=backbone,
    y_only_model=None,
    num_timesteps=1000,
    scheduler='power_mean',
    cat_scheduler='log_linear',
    device='cuda'
).to('cuda')

optimizer = torch.optim.AdamW(diffusion.parameters(), lr=1e-3)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

for epoch in range(100):
    for batch in train_loader:
        x = batch['data'].to('cuda')
        optimizer.zero_grad()
        
        d_loss, c_loss = diffusion.mixed_loss(x)
        loss = d_loss + c_loss
        loss.backward()
        optimizer.step()
```

### Sampling

```python
diffusion.eval()
with torch.no_grad():
    samples = diffusion.sample(num_samples=100, seq_len=62)

# Inverse transform to original scale
samples_np = samples.cpu().numpy()
original_scale = dataset.inverse_transform_child(samples_np)

# Save as CSV
import pandas as pd
B, S, F = original_scale.shape
flat_samples = original_scale.reshape(B*S, F)
columns = dataset.dt_cols + dataset.num_cols + dataset.cat_cols
df = pd.DataFrame(flat_samples, columns=columns)
df.to_csv('generated_samples.csv', index=False)
```

### Evaluation

```python
import pandas as pd
import numpy as np
from sdmetrics.reports.single_table import QualityReport

with open(os.path.join(DATA_DIR, 'metadata.json'), 'r') as f:
    multi_metadata = json.load(f)
raw_cols = multi_metadata['tables']['historical']['columns']
sdmetrics_metadata = {'columns': {}}
for col, info in raw_cols.items():
    if col in ['Id', 'Date', 'Store']:
        continue
    sdtype = info['sdtype']
    sdmetrics_metadata['columns'][col] = {'sdtype': sdtype}
    
real = pd.read_csv(os.path.join(DATA_DIR, 'historical.csv'))
syn = pd.read_csv(os.path.join(CKPT_DIR, 'generated_samples.csv'))

real = real.drop(columns=['Id', 'Date', 'Store'], errors='ignore')
syn = syn.drop(columns=['Date'], errors='ignore')
syn['Customers'] = syn['Customers'].astype(int)
syn = syn[real.columns]

report = QualityReport()
report.generate(real, syn, sdmetrics_metadata)
print(f"Overall Quality Score: {report.get_score():.4f}")
print(report.get_properties())

# 컬럼별 점수
report.get_details(property_name='Column Shapes')
report.get_details(property_name='Column Pair Trends')
```

---

## References

* TabDiff (ICLR 2025) [Paper](https://openreview.net/forum?id=swvURjrt8z) [Code](https://github.com/MinkaiXu/TabDiff)
* TabDiT (ICLR 2025) [Paper](https://openreview.net/forum?id=bhOysNJvWm) [Code](https://github.com/fabriziogaruti/TabDiT)
* SDMetrics [Website](https://docs.sdv.dev/sdmetrics)
