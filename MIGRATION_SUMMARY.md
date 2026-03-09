# SageMaker SDK v3 Migration Summary

This document summarizes the changes made to migrate the phishing detection notebooks from SageMaker Python SDK v2 to v3.

## Changes Applied

### 1. Package Version Updates

All notebooks now use `sagemaker>=3.0.0` instead of `sagemaker==2.253.1`:

- **01_data_processing.ipynb**: Updated to `sagemaker>=3.0.0`
- **02_model_training.ipynb**: Updated to `sagemaker>=3.0.0` added sagemaker-core
- **03_model_deployment.ipynb**: Updated to `sagemaker>=3.0.0` added sagemaker-core
- **04_benchmarking.ipynb**: Updated to `sagemaker>=3.0.0`

### 2. Import Path Updates

**02_model_training.ipynb** - Updated import paths to use v3 modular architecture:

```python
# V2 (Old)
from sagemaker.modules.configs import Compute, OutputDataConfig, SourceCode, StoppingCondition, InputData
from sagemaker.modules.train import ModelTrainer
from sagemaker.model import Model
from sagemaker.predictor import Predictor

# V3 (New)
from sagemaker.train.configs import Compute, OutputDataConfig, SourceCode, StoppingCondition, InputData
from sagemaker.train import ModelTrainer
from sagemaker.core.resources import Model, Endpoint, EndpointConfig
from sagemaker.core.shapes.shapes import ContainerDefinition, ProductionVariant, ProductionVariantRoutingConfig
```

#  v2 (old) "sagemaker.utils.name_from_base does not exist, so have to make use of time
this change model_name = f"qwen-phishing-{int(time.time())} can be observed in notebook 3 and 4

### 3. Code Patterns Already Using v3

The notebooks were already using several v3 patterns:

- **ModelTrainer** with structured config objects (`Compute`, `SourceCode`, `InputData`)
- **Model.create()** and **Endpoint.create()** from `sagemaker-core`
- Structured configuration objects instead of scattered parameters

## Key v3 Features Used

### Training (02_model_training.ipynb)
- `ModelTrainer` with structured configuration
- `Compute` config for instance settings
- `SourceCode` config for training code
- `InputData` config for data channels
- `OutputDataConfig` for model artifacts
- `StoppingCondition` for training limits

### Deployment (03_model_deployment.ipynb)
- `Model.create()` from `sagemaker-core`
- `Endpoint.create()` from `sagemaker-core`
- `ContainerDefinition` for container configuration
- `ProductionVariant` for endpoint configuration

## Benefits of v3

1. **Type Safety**: Full type hints and IDE support
2. **Modular Architecture**: Separate packages for training, serving, and core resources
3. **Structured Configuration**: Config objects provide better organization
4. **Better Resource Management**: Improved resource chaining and lifecycle management

## Testing Recommendations

After migration, test the following workflows:

1. **Data Processing** (01_data_processing.ipynb)
   - Load dataset from HuggingFace
   - Preprocess and upload to S3
   - Verify S3 URIs are stored correctly

2. **Model Training** (02_model_training.ipynb)
   - Create training job with ModelTrainer
   - Verify MLflow integration works
   - Check model artifacts are saved to S3

3. **Model Deployment** (03_model_deployment.ipynb)
   - Deploy model with Model.create() and Endpoint.create()
   - Test inference with sample emails
   - Verify batch inference works

4. **Benchmarking** (04_benchmarking.ipynb)
   - Run latency tests
   - Test concurrent load
   - Verify cleanup works

## Reference

For complete migration details, see the [official SageMaker SDK v3 migration guide](https://github.com/aws/sagemaker-python-sdk/blob/master/migration.md).
