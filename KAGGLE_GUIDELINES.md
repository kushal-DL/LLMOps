# Kaggle API Guidelines for LLMOps Workshop

##  Overview

This guide ensures efficient use of Kaggle resources within the workshop's constraints.

**Hard Limits:**
- 1000 Kaggle API requests per day
- 30 hours GPU usage total (5 days × ~6 hrs/day max)

---

##  Request Budget Strategy

### Daily Allocation (1000 requests/day)

```
Setup & Authentication:     ~50 requests
Notebook Creation:          ~200 requests
Data/Model Downloads:       ~300 requests (batched)
Training Runs (logged):     ~250 requests
Evaluation & Logging:       ~150 requests
Buffer:                     ~50 requests
━━━━━━━━━━━━━━━━━━━━━━━
TOTAL:                      ~1000 requests
```

### Batching Strategy

**Group operations to minimize per-request overhead:**
-  Batch multiple model downloads in single request
-  Queue training jobs and check status in batches
-  Aggregate logging/metrics before transmission
-  Don't make individual requests for each epoch
-  Don't poll status every minute

---

##  Authentication Setup

### 1. Configure Kaggle API

```bash
# In notebook, set token from environment
import os
os.environ['KAGGLE_USERNAME'] = 'your_username'
os.environ['KAGGLE_KEY'] = 'your_api_key'

# Verify authentication
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
print(" Authenticated")
```

### 2. Use Kaggle Datasets Efficiently

```python
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

#  DO: Batch download multiple datasets
datasets = ['tokenization_data', 'pretrained_models', 'eval_benchmarks']
for ds in datasets:
    api.dataset_download_files(ds, path='./data', unzip=True)

#  DON'T: Individual downloads in loops
```

### 3. Model Caching

```python
import os
from pathlib import Path

CACHE_DIR = Path('/kaggle/input/cache')  # Use persistent storage
MODEL_NAME = "meta-llama/Llama-2-7b-hf"

def get_model(name, cache_dir=CACHE_DIR):
    """Efficient model loading with caching"""
    cache_path = cache_dir / name.replace('/', '_')
    
    if cache_path.exists():
        print(f" Loading from cache: {cache_path}")
        return cache_path
    
    # Download once
    print(f"↓ Downloading: {name}")
    # Download logic here
    return cache_path
```

---

##  GPU Usage Optimization

### Notebook Settings

```python
# Cell 1: GPU Configuration
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# Use mixed precision to reduce memory
from torch.cuda.amp import autocast
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### Memory-Efficient Training

```python
# Use gradient accumulation
accumulation_steps = 4
for batch_idx, batch in enumerate(dataloader):
    with autocast():
        outputs = model(batch)
        loss = criterion(outputs, batch['labels']) / accumulation_steps
        loss.backward()
    
    if (batch_idx + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### GPU Time Budget Per Day

| Day | Task | Est. GPU Time | Priority |
|-----|------|---------------|----------|
| 1 | Setup + Forward pass tests | 30 min | Medium |
| 2 | Architecture visualization | 45 min | Medium |
| 3 | LoRA training (~3 epochs) | 2 hrs | High |
| 4 | vLLM inference profiling | 1.5 hrs | High |
| 5 | Full deployment test + monitoring | 1 hr | High |
| **Total** | | **~6 hrs** | - |

---

##  Monitoring & Logging

### Track API Usage

```python
import json
from datetime import datetime

API_LOG = 'api_usage.json'

def log_api_call(endpoint, request_count, timestamp=None):
    """Log API calls for tracking"""
    timestamp = timestamp or datetime.now().isoformat()
    
    log_entry = {
        'timestamp': timestamp,
        'endpoint': endpoint,
        'request_count': request_count
    }
    
    with open(API_LOG, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')

# Usage
log_api_call('dataset_download', 300)
```

### Daily Report Template

```python
def generate_daily_report():
    """Generate resource usage report"""
    report = {
        'date': datetime.now().isoformat(),
        'api_requests_used': 0,
        'gpu_hours_used': 0,
        'notebooks_completed': [],
        'warnings': []
    }
    
    # Parse logs
    total_requests = 0
    with open(API_LOG) as f:
        for line in f:
            entry = json.loads(line)
            total_requests += entry['request_count']
    
    report['api_requests_used'] = total_requests
    
    if total_requests > 900:
        report['warnings'].append(' Approaching daily limit (900/1000)')
    
    return report
```

---

##  Batching Examples

### Example 1: Batch Model Downloads

```python
#  Inefficient (multiple API calls)
for model in ['gpt2', 'distilbert-base', 'roberta-base']:
    api.dataset_download_files(f'model_{model}')

#  Efficient (one call)
models_file = 'models_metadata.json'
with open(models_file) as f:
    models = json.load(f)['models']
    for model in models:
        # process batch
```

### Example 2: Training Job Batching

```python
#  Single submission, batched monitoring
jobs = []
for config in training_configs:
    job = submit_training_job(config)
    jobs.append(job)

# Check all statuses in one call
results = api.check_job_status(job_ids=[j['id'] for j in jobs])
```

### Example 3: Metric Logging

```python
#  Batch metrics before logging
metrics_buffer = []

for step in range(1000):
    loss = train_step()
    metrics_buffer.append({'step': step, 'loss': loss})
    
    # Log every 50 steps (not every step)
    if step % 50 == 0:
        log_metrics(metrics_buffer)
        metrics_buffer = []
```

---

##  Common Pitfalls

|  Don't |  Do |
|---------|--------|
| Poll API every 10 seconds | Check status every 5-10 minutes |
| Download same data repeatedly | Cache and reuse |
| Individual model uploads | Batch upload |
| Uncompressed logging | Aggregate + compress |
| GPU running idle | Use mixed precision & gradient checkpointing |

---

##  Contingency Plans

### If approaching API limit:
1. Switch to batch operations only
2. Use Kaggle datasets instead of API calls
3. Pre-compute and cache results
4. Defer non-critical logging

### If GPU hours running low:
1. Use CPU for non-critical tasks
2. Reduce batch sizes
3. Decrease training epochs
4. Use inference-only for later days

---

##  Checklist for Each Notebook

Before publishing each notebook:

- [ ] API requests estimated and budgeted
- [ ] GPU time monitored and logged
- [ ] Caching implemented for models/data
- [ ] Batching used for API calls
- [ ] Error handling for rate limits
- [ ] Fallbacks for GPU unavailability
- [ ] Final API usage report generated

---

**Last Updated:** 2026-08-25  
**Version:** 1.0
