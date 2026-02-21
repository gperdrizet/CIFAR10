# Uploading models to Hugging Face Hub

This project includes optional integration with Hugging Face Hub for sharing and loading trained models.

## Quick start

Each notebook has two control parameters in the hyperparameters cell:

```python
# Model control
retrain_model = True        # Set to False to load existing model
model_source = 'local'      # 'local' to load from disk, 'huggingface' to download from Hub
```

### Load a pre-trained model

To use an existing model instead of training:

1. Set `retrain_model = False`
2. Set `model_source = 'local'` (load from disk) or `'huggingface'` (download from Hub)
3. Run the notebook - it will skip training and use the loaded model

### Train a new model

Keep `retrain_model = True` to train from scratch (default behavior).

## Setup

### Default configuration (students)

By default, the project is configured to download pre-trained models from `gperdrizet/CIFAR10` on Hugging Face Hub. No setup needed - just set `retrain_model=False` and `model_source='huggingface'` in any notebook.

### Custom configuration (for uploading your own models)

If you want to train and upload models to your own repository:

### 1. Create environment configuration

Copy the template and edit it:
```bash
cp .env.template .env
```

Edit `.env` and update:
```bash
HF_REPO_ID=your-username/your-repo-name
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx  # Optional, for uploading
```

**Important**: Never commit the `.env` file (it's in `.gitignore`)

### 2. Install Hugging Face CLI

The `huggingface_hub` package is already included in the project dependencies.

### 3. Create a Hugging Face account

If you don't have one already, sign up at [huggingface.co](https://huggingface.co/join)

### 4. Create an access token

1. Go to [Settings → Access Tokens](https://huggingface.co/settings/tokens)
2. Click "New token"
3. Give it a name (e.g., "CIFAR10-models")
4. Select "Write" permission
5. Copy the token

### 5. Login from the terminal

```bash
huggingface-cli login
```

Paste your token when prompted.

### 6. Create a model repository

1. Go to [huggingface.co](https://huggingface.co)
2. Click your profile → "New Model"
3. Name it (e.g., "cifar10-models")
4. Make it public or private as desired

## Repository configuration

The default repository is `gperdrizet/CIFAR10`, which contains pre-trained models for all notebooks. Students can use these models without any configuration.

If you created your own repository, make sure your `.env` file has:
```bash
HF_REPO_ID=your-username/your-repo-name
```

The configuration is automatically loaded from `.env` when you import the `configuration` module.

## Usage

In each training notebook, after the model is saved, there's an optional upload cell:

```python
upload_to_hf = False  # Set to True to enable upload

if upload_to_hf:
    config.upload_to_huggingface(
        model_path=model_path,
        model_name='model_name.pth',
        commit_message=f'Model - Test accuracy: {test_accuracy:.2f}%'
    )
```

Simply change `upload_to_hf = True` to upload the model after training.

## Models

The following models can be uploaded:

| Notebook | Model File | Description |
|----------|-----------|-------------|
| 01-DNN.ipynb | dnn.pth | Baseline fully-connected network |
| 02-CNN.ipynb | cnn.pth | Convolutional neural network |
| 03-RGB-CNN.ipynb | rgb_cnn.pth | RGB CNN |
| 04-architecture_optimization.ipynb | optimized_cnn.pth | Architecture-optimized CNN |
| 05-training-optimization.ipynb | training_optimized_cnn.pth | Training-optimized CNN |
| 06-augmented-CNN.ipynb | augmented_cnn.pth | CNN with data augmentation |
| 07-resnet50.ipynb | resnet50.pth | Fine-tuned ResNet50 |

## Downloading models

### From within notebooks

Models are automatically downloaded when you set:
```python
retrain_model = False
model_source = 'huggingface'
```

The model will be downloaded from Hugging Face Hub and cached locally.

### Programmatically

To download models from Hugging Face Hub in your own code:

```python
from huggingface_hub import hf_hub_download
import torch

# Download a specific model
model_path = hf_hub_download(
    repo_id='your-username/your-repo-name',
    filename='dnn.pth'
)

# Load the model
model = torch.load(model_path, weights_only=False)
```

## Model behavior

The notebooks will:

1. **retrain_model=True**: Always train a new model (default)
2. **retrain_model=False, model_source='local'**: Load from local disk (fails if not found)
3. **retrain_model=False, model_source='huggingface'**: Download from Hub (falls back to local if download fails)

This allows you to:
- Skip expensive training when testing evaluation code
- Share models with collaborators via Hugging Face Hub
- Quickly reproduce results from pre-trained models

## Security note

- Never commit your Hugging Face token to git
- The token is stored securely by the Hugging Face CLI
- You can revoke tokens at any time from your Hugging Face settings
