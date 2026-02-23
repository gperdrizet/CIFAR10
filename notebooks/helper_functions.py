'''Helper functions for model management and Hugging Face Hub integration.'''

from pathlib import Path
import os
import json
import torch
from huggingface_hub import HfApi, hf_hub_download


def load_history_from_source(
    history_path: Path,
    model_source: str = 'local'
) -> dict:
    '''
    Load training history from local disk or Hugging Face Hub.

    Args:
        history_path (Path): Full path to the history file (e.g., './models/dnn_history.json')
        model_source (str): 'local' to load from disk, 'huggingface' to download from Hub.

    Returns:
        dict or None: Training history dictionary, or None if not found or error occurs.
    '''

    try:
        if model_source == 'local':

            # Load history from local JSON file
            with open(history_path, 'r') as f:
                history = json.load(f)

            print(f'Training history loaded from {history_path}')
            return history

        elif model_source == 'huggingface':

            # Download history JSON from Hugging Face Hub
            repo_id = os.getenv('HF_REPO_ID') or 'gperdrizet/CIFAR10'

            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=f'training_histories/{history_path.name}',
                repo_type='model'
            )

            with open(downloaded_path, 'r') as f:
                history = json.load(f)

            print(f'Training history downloaded from Hugging Face')
            return history

    except Exception as e:
        print(f'Failed to load training history: {e}')
        return None


def upload_results(
        results_path: Path,
        results_name: str,
        commit_message: str = None
) -> None:
    '''
    Upload a performance results JSON file to Hugging Face Hub in the performance_results directory if credentials are available.

    Args:
        results_path (Path): Path to the results JSON file.
        results_name (str): Name for the file in the repository.
        commit_message (str, optional): Commit message for upload.
    '''

    if should_upload_to_huggingface():

        commit_msg = commit_message or f'Updated {results_name}'

        upload_to_huggingface(
            model_path=results_path,
            model_name=results_name,
            commit_message=commit_msg,
            path_in_repo=f'performance_results/{results_name}'
        )

        print(f'Performance results uploaded: {results_name}')

    else:
        print('Skipping Hugging Face upload for performance results (no .env file with HF_TOKEN found)')


# Results saving utility
def save_results(results: dict, results_path: Path) -> None:
    '''
    Save results dictionary as a JSON file, converting numpy arrays to lists for serialization.

    Args:
        results (dict): Results dictionary to save.
        results_path (Path): Path to save the results JSON file.
    '''

    results_json = {}

    for key, value in results.items():
    
        # Convert numpy arrays to lists for JSON serialization
        if hasattr(value, 'tolist'):
            results_json[key] = value.tolist()
    
        elif isinstance(value, list):
            results_json[key] = value

        else:
            results_json[key] = value

    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2)

    print(f'Results saved to: {results_path}')


def load_model_from_source(
    model_path: Path,
    model_source: str = 'local',
    device: torch.device = None
) -> torch.nn.Module:
    '''
    Load a PyTorch model from local disk or Hugging Face Hub.

    Args:
        model_path (Path): Full path to the model file (e.g., './models/dnn.pth')
        model_source (str): 'local' to load from disk, 'huggingface' to download from Hub.
        device (torch.device, optional): Device to load model on (defaults to CPU if not specified).

    Returns:
        torch.nn.Module or None: Loaded PyTorch model, or None if loading failed.
    '''

    model_name = model_path.name
    device = device or torch.device('cpu')

    try:
        if model_source == 'local':

            # Load model from local .pth file
            if not model_path.exists():
                print(f'Model not found at {model_path}')
                return None

            model = torch.load(
                model_path,
                map_location=device,
                weights_only=False
            )

            print(f'Model loaded from {model_path}')
            return model

        elif model_source == 'huggingface':

            # Download model .pth from Hugging Face Hub
            repo_id = os.getenv('HF_REPO_ID') or 'gperdrizet/CIFAR10'
            print(f'Downloading model from Hugging Face: {repo_id}/{model_name}')

            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=f'models/{model_name}',
                repo_type='model'
            )

            model = torch.load(downloaded_path, map_location=device, weights_only=False)
            print(f'Model downloaded from Hugging Face')
            return model

        else:
            print(f'Invalid model_source: "{model_source}". Must be "local" or "huggingface"')
            return None

    except Exception as e:
        print(f'Failed to load model: {e}')
        return None


def upload_to_huggingface(
    model_path: Path,
    model_name: str,
    commit_message: str = None,
    path_in_repo: str = None
) -> str:
    '''
    Upload a file to Hugging Face Hub.

    Args:
        model_path (Path): Path to the file to upload.
        model_name (str): Name for the file in the repository (e.g., 'dnn.pth').
        commit_message (str, optional): Commit message (defaults to model name).
        path_in_repo (str, optional): Path within repo (e.g., 'models/dnn.pth'). If None, uses model_name.

    Returns:
        str or None: URL of the uploaded file, or None if upload failed.
    '''

    try:
        api = HfApi()
        repo_id = os.getenv('HF_REPO_ID') or 'gperdrizet/CIFAR10'
        commit_message = commit_message or f'Upload {model_name}'
        repo_path = path_in_repo or model_name

        url = api.upload_file(
            path_or_fileobj=str(model_path),
            path_in_repo=repo_path,
            repo_id=repo_id,
            repo_type='model',
            commit_message=commit_message
        )

        print('Model uploaded to Hugging Face')
        return url

    except Exception as e:
        print(f'Failed to upload to Hugging Face: {e}')
        return None


def should_upload_to_huggingface() -> bool:
    '''
    Check if models should be automatically uploaded to Hugging Face Hub.
    Checks for the existence of .env file and HF_TOKEN environment variable.

    Returns:
        bool: True if .env exists and HF_TOKEN is set, False otherwise.
    '''

    env_path = Path(__file__).parent.parent / '.env'
    hf_token = os.getenv('HF_TOKEN')
    
    if env_path.exists() and hf_token:
        return True
    
    return False


def save_model(model: torch.nn.Module, model_path: Path) -> None:

    '''
    Save a PyTorch model to disk as a .pth file.

    Args:
        model (torch.nn.Module): The PyTorch model to save.
        model_path (Path): Path to save the model file.
    '''
    torch.save(model, model_path)
    print(f'Model saved to: {model_path}')


def save_history(history: dict, history_path: Path) -> None:

    '''
    Save training history as a JSON file.

    Args:
        history (dict): Training history dictionary.
        history_path (Path): Path to save the history JSON file.
    '''
    history_json = {}
    for key, value in history.items():
        # Convert numpy arrays to lists for JSON serialization
        if hasattr(value, 'tolist'):
            history_json[key] = value.tolist()
        elif isinstance(value, list):
            history_json[key] = value
        else:
            history_json[key] = value
    with open(history_path, 'w') as f:
        json.dump(history_json, f, indent=2)
    print(f'Training history saved to: {history_path}')


def upload_model(
        model_path: Path,
        model_name: str,
        commit_message: str = None
) -> None:
    
    '''
    Upload a model file to Hugging Face Hub if credentials are available.

    Args:
        model_path (Path): Path to the model file.
        model_name (str): Name for the file in the repository.
        commit_message (str, optional): Commit message for upload.
    '''
    if should_upload_to_huggingface():
        commit_msg = commit_message or f'Updated {model_name.replace(".pth", "")}'
        upload_to_huggingface(
            model_path=model_path,
            model_name=model_name,
            commit_message=commit_msg,
            path_in_repo=f'models/{model_name}'
        )
    else:
        print('Skipping Hugging Face upload for model (no .env file with HF_TOKEN found)')


def upload_history(
        history_path: Path,
        history_name: str,
        commit_message: str = None
) -> None:
    
    '''
    Upload a training history JSON file to Hugging Face Hub if credentials are available.

    Args:
        history_path (Path): Path to the history JSON file.
        history_name (str): Name for the file in the repository.
        commit_message (str, optional): Commit message for upload.
    '''
    if should_upload_to_huggingface():
        commit_msg = commit_message or f'Updated {history_name.replace("_history.json", "")}'
        upload_to_huggingface(
            model_path=history_path,
            model_name=history_name,
            commit_message=commit_msg,
            path_in_repo=f'training_histories/{history_name}'
        )
    else:
        print('Skipping Hugging Face upload for history (no .env file with HF_TOKEN found)')


def upload_optuna_study(
        optuna_study_path: Path,
        optuna_study_name: str,
        commit_message: str = None
) -> None:
    
    '''
    Upload an Optuna study database file to Hugging Face Hub if credentials are available.

    Args:
        optuna_study_path (Path): Path to the Optuna study database file.
        optuna_study_name (str): Name for the file in the repository.
        commit_message (str, optional): Commit message for upload.
    '''
    if should_upload_to_huggingface():
        commit_msg = commit_message or f'Updated {optuna_study_name}'
        upload_to_huggingface(
            model_path=optuna_study_path,
            model_name=optuna_study_name,
            commit_message=commit_msg,
            path_in_repo=f'optuna_results/{optuna_study_name}'
        )
        print(f'Optuna study uploaded: {optuna_study_name}')
    else:
        print('Skipping Hugging Face upload for Optuna study (no .env file with HF_TOKEN found)')