# 🔒 Phishing Detection with Sequence Classification

Notebooks from the AWS workshop: **Customizing SLMs for email phishing with Amazon SageMaker AI**.

**Updated for SageMaker Python SDK v3** - Uses modular architecture with `sagemaker-train`, `sagemaker-serve`, and `sagemaker-core`.

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
   - Uses SageMaker SDK v3 `ModelTrainer` with structured config objects (`Compute`, `SourceCode`, `InputData`)
3. **03_model_deployment.ipynb** → Stores: `endpoint_name`, `model_name`
   - Uses SageMaker SDK v3 `Model.create()` and `Endpoint.create()` from `sagemaker-core`
4. **04_benchmarking.ipynb** → Uses stored endpoint info for testing

## SageMaker SDK v3 Migration

These notebooks have been updated to use **SageMaker Python SDK v3**, which provides:

- **Modular architecture**: Separate packages for training (`sagemaker-train`), serving (`sagemaker-serve`), and core resources (`sagemaker-core`)
- **Type safety**: Full type hints and IDE support
- **Structured configuration**: Config objects like `Compute`, `SourceCode`, `InputData` instead of scattered parameters
- **Better resource management**: Improved resource chaining and lifecycle management

### Key Changes from v2:
- `Estimator` → `ModelTrainer` with structured config objects
- `Model.deploy()` → `Model.create()` + `Endpoint.create()` from `sagemaker-core`
- Import paths: `sagemaker.train.*` instead of `sagemaker.estimator.*`
- Configuration objects: `Compute`, `SourceCode`, `InputData`, `OutputDataConfig`

For full migration details, see the [official migration guide](https://github.com/aws/sagemaker-python-sdk/blob/master/migration.md).