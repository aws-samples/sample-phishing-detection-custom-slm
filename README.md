# 🔒 Phishing Detection with Sequence Classification

Notebooks from the AWS workshop: **Customizing SLMs for email phishing with Amazon SageMaker AI**.

## Technical Architecture

### Model: Qwen2.5-1.5B
- **Architecture**: Transformer-based decoder
- **Task**: Binary sequence classification (Safe vs. Phishing)
- **Fine-tuning**: RSLoRA (rank-stabilized LoRA) on classification head
- **Precision**: bfloat16 mixed precision
- **Training**: ~60-75 minutes on ml.g5.xlarge

### Dataset: `drorrabin/phishing_emails-data`
- **Size**: ~27k training samples, ~3.7k test samples
- **Format**: Email content with binary labels
- **Balance**: 50/50 safe vs. phishing in training set
- **Source**: [HuggingFace](https://huggingface.co/datasets/drorrabin/phishing_emails-data)

### Deployment: SageMaker + vLLM
- **Container**: LMI v18 with vLLM 0.12.0
- **Inference**: Text classification mode (single token prediction)
- **Instance**: `ml.g5.xlarge` (1x NVIDIA A10G, 24GB VRAM)
- **Routing**: Least Outstanding Requests for load balancing



## Repository Structure

```
phishing-detection-notebooks/
├── 01_data_processing.ipynb      # Load, preprocess, upload to S3
├── 02_model_training.ipynb       # Fine-tune with SageMaker + MLflow
├── 03_model_deployment.ipynb     # Deploy endpoint with vLLM
├── 04_benchmarking.ipynb         # Latency/throughput testing
├── utils.py                      # Helper functions (S3, model extraction)
└── README.md                     # This file
```

### Notebook Workflow

The notebooks are designed to run sequentially, with state passed via IPython's `%store` magic:

1. **01_data_processing.ipynb** → Stores: `train_s3_uri`, `val_s3_uri`, `test_s3_uri`, `NUM_LABELS`
2. **02_model_training.ipynb** → Stores: `model_s3_uri`, `training_job_name`, `mlflow_experiment_name`
3. **03_model_deployment.ipynb** → Stores: `endpoint_name`, `model_name`
4. **04_benchmarking.ipynb** → Uses stored endpoint info for testing