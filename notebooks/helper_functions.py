'''Helper functions for model management and Hugging Face Hub integration.'''

from pathlib import Path
import os
import pickle
import json
import torch
from huggingface_hub import HfApi, hf_hub_download


def load_history_from_source(
    model_path: Path,
    model_name: str,
    model_source: str = 'local',
    repo_id: str = None,
    repo_type: str = 'model'
) -> dict:
    '''Load training history from local disk or Hugging Face Hub.
    
    Args:
        model_path: Path to the local model file (history will be in same dir)
        model_name: Name of the model file (e.g., 'dnn.pth')
        model_source: 'local' to load from disk, 'huggingface' to download from Hub
        repo_id: Optional repository ID (uses HF_REPO_ID env var if not provided)
        repo_type: Type of repository (default: 'model')
    
    Returns:
        Training history dictionary, or None if not found
    '''
    history_name = model_name.replace('.pth', '_history.json').replace('.safetensors', '_history.json')
    history_path = model_path.parent / history_name
    
    try:
        if model_source == 'local':
            if not history_path.exists():
                # Try old pickle format as fallback
                history_name_pkl = model_name.replace('.pth', '_history.pkl').replace('.safetensors', '_history.pkl')
                history_path_pkl = model_path.parent / history_name_pkl
                
                if history_path_pkl.exists():
                    with open(history_path_pkl, 'rb') as f:
                        history = pickle.load(f)
                    print(f'Training history loaded from {history_path_pkl} (pickle format)')
                    return history
                else:
                    return None
            
            with open(history_path, 'r') as f:
                history = json.load(f)
            
            print(f'Training history loaded from {history_path}')
            return history
        
        elif model_source == 'huggingface':
            repo_id = repo_id or os.getenv('HF_REPO_ID')
            
            if not repo_id:
                return None
            
            try:
                # Download from training_histories/ subdirectory
                downloaded_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=f'training_histories/{history_name}',
                    repo_type=repo_type
                )
                
                with open(downloaded_path, 'r') as f:
                    history = json.load(f)
                
                print(f'Training history downloaded from Hugging Face')
                return history
            
            except Exception:
                # Fallback to pickle format in training_histories/
                history_name_pkl = model_name.replace('.pth', '_history.pkl').replace('.safetensors', '_history.pkl')
                downloaded_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=f'training_histories/{history_name_pkl}',
                    repo_type=repo_type
                )
                
                with open(downloaded_path, 'rb') as f:
                    history = pickle.load(f)
                
                print(f'Training history downloaded from Hugging Face (pickle format)')
                return history
        
        else:
            return None
            
    except Exception as e:
        # History file doesn't exist or can't be loaded
        return None


def load_model_from_source(
    model_path: Path,
    model_name: str,
    model_source: str = 'local',
    device: torch.device = None,
    repo_id: str = None,
    repo_type: str = 'model'
) -> torch.nn.Module:
    '''Load a model from local disk or Hugging Face Hub.
    
    Handles all error cases internally and returns None if model cannot be loaded.
    Supports both safetensors format (preferred) and legacy .pth format.
    
    Args:
        model_path: Path to the local model file
        model_name: Name of the model file (e.g., 'dnn.pth' or 'dnn.safetensors')
        model_source: 'local' to load from disk, 'huggingface' to download from Hub
        device: Device to load model on (defaults to CPU if not specified)
        repo_id: Optional repository ID (uses HF_REPO_ID env var if not provided)
        repo_type: Type of repository (default: 'model')
    
    Returns:
        Loaded PyTorch model, or None if loading failed
    '''
    device = device or torch.device('cpu')
    
    try:
        if model_source == 'local':
            if not model_path.exists():
                print(f'Model not found at {model_path}')
                return None
            
            # Check if it's a safetensors file or legacy .pth
            if str(model_path).endswith('.safetensors'):
                # For safetensors, we need the model architecture
                # This will only work if the model was saved as state_dict
                print(f'Loading safetensors requires model architecture to be defined first')
                return None
            else:
                # Legacy .pth format with full model
                model = torch.load(model_path, map_location=device, weights_only=False)
                print(f'Model loaded from {model_path}')
                return model
        
        elif model_source == 'huggingface':
            repo_id = repo_id or os.getenv('HF_REPO_ID')

            if not repo_id:
                print('HF_REPO_ID not set in environment')
                return None
            
            print(f'Downloading model from Hugging Face: {repo_id}/{model_name}')
            
            # Download from models/ subdirectory
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=f'models/{model_name}',
                repo_type=repo_type
            )
            
            # Check format and load accordingly
            if model_name.endswith('.safetensors'):
                print(f'Loading safetensors requires model architecture to be defined first')
                return None
            else:
                # Legacy .pth format
                model = torch.load(downloaded_path, map_location=device, weights_only=False)
                print(f'Model downloaded from Hugging Face')
                return model
        
        else:
            print(f'Invalid model_source: "{model_source}". Must be "local" or "huggingface"')
            return None
            
    except Exception as e:
        error_msg = str(e)
        print(f'Failed to load model: {error_msg}')
        
        # Check if it's a "not found" error from HuggingFace
        if model_source == 'huggingface' and ('not found' in error_msg.lower() or '404' in error_msg):
            print(f'Model "{model_name}" does not exist in repository "{repo_id}"')
            print(f'This is expected if you haven\'t uploaded it yet.')
            
            # Try local fallback
            if model_path.exists():
                print(f'Trying local fallback...')

                try:
                    print(f'Loading model from local disk: {model_path}')
                    model = torch.load(model_path, map_location=device, weights_only=False)
                    print(f'Model loaded from {model_path}')
                    return model

                except Exception as fallback_error:
                    print(f'Local fallback also failed: {fallback_error}')
                    return None
            else:
                print(f'No local copy found at {model_path}')
                return None
        
        # For other errors, just return None
        return None


def upload_to_huggingface(
    model_path: Path,
    model_name: str,
    commit_message: str = None,
    repo_id: str = None,
    repo_type: str = 'model',
    path_in_repo: str = None
) -> str:
    '''Upload a file to Hugging Face Hub.
    
    Args:
        model_path: Path to the file to upload
        model_name: Name for the file in the repository (e.g., 'dnn.pth')
        commit_message: Optional commit message (defaults to model name)
        repo_id: Optional repository ID (uses HF_REPO_ID env var if not provided)
        repo_type: Type of repository (default: 'model')
        path_in_repo: Optional path within repo (e.g., 'models/dnn.pth'). If None, uses model_name
    
    Returns:
        URL of the uploaded file, or None if upload failed
    '''
    try:
        api = HfApi()
        repo_id = repo_id or os.getenv('HF_REPO_ID')

        if not repo_id:
            raise ValueError('HF_REPO_ID not set in environment')
            
        commit_message = commit_message or f'Upload {model_name}'
        
        # Use path_in_repo if provided, otherwise use model_name
        repo_path = path_in_repo or model_name
        
        # Upload file to repository
        url = api.upload_file(
            path_or_fileobj=str(model_path),
            path_in_repo=repo_path,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=commit_message
        )
        
        print('Model uploaded to Hugging Face')
        return url
        
    except Exception as e:
        print(f'Failed to upload to Hugging Face: {e}')
        return None


def should_upload_to_huggingface() -> bool:
    '''Check if models should be automatically uploaded to Hugging Face Hub.
    Checks for the existence of .env file and HF_TOKEN environment variable.
    
    Returns:
        True if .env exists and HF_TOKEN is set, False otherwise
    '''
    env_path = Path(__file__).parent.parent / '.env'
    hf_token = os.getenv('HF_TOKEN')
    
    if env_path.exists() and hf_token:
        return True
    
    return False


def save_and_upload_model(
    model: torch.nn.Module,
    model_path: Path,
    model_name: str,
    history: dict = None,
    optuna_study_path: Path = None,
    optuna_study_name: str = None
) -> None:
    '''Save a model locally and optionally upload to Hugging Face Hub.
    Automatically uploads if .env file exists with HF_TOKEN.
    
    Uses pickle format (torch.save(model)) which is the standard PyTorch approach.
    HuggingFace will flag it, but this is normal for PyTorch models.
    
    Args:
        model: The PyTorch model to save
        model_path: Path where to save the model locally
        model_name: Name of the model file (e.g., 'dnn.pth')
        history: Optional training history dictionary to save
        optuna_study_path: Optional path to Optuna study database file
        optuna_study_name: Optional name for Optuna study file in repo (e.g., 'cnn_optimization.db')
    '''
    # Save full model with pickle (standard PyTorch approach)
    torch.save(model, model_path)
    print(f'Model saved to: {model_path}')
    
    # Save training history if provided
    if history is not None:
        history_name = model_name.replace('.pth', '_history.json')
        history_path = model_path.parent / history_name
        
        # Convert numpy arrays to lists for JSON serialization
        history_json = {}
        for key, value in history.items():
            if hasattr(value, 'tolist'):  # numpy array
                history_json[key] = value.tolist()
            elif isinstance(value, list):
                history_json[key] = value
            else:
                history_json[key] = value
        
        with open(history_path, 'w') as f:
            json.dump(history_json, f, indent=2)
        
        print(f'Training history saved to: {history_path}')
    
    # Check if we should upload to HuggingFace
    if should_upload_to_huggingface():
        print('\nUploading to Hugging Face Hub...')

        # Simple commit message indicating which model was updated
        base_name = model_name.replace('.pth', '')
        commit_msg = f'Updated {base_name}'
        
        # Upload model to models/ subdirectory
        upload_to_huggingface(
            model_path=model_path,
            model_name=model_name,
            commit_message=commit_msg,
            path_in_repo=f'models/{model_name}'
        )
        
        # Upload history to training_histories/ subdirectory if it exists
        if history is not None:
            upload_to_huggingface(
                model_path=history_path,
                model_name=history_name,
                commit_message=commit_msg,
                path_in_repo=f'training_histories/{history_name}'
            )
        
        # Upload Optuna study database to optuna_results/ subdirectory if provided
        if optuna_study_path is not None and optuna_study_path.exists():
            if optuna_study_name is None:
                optuna_study_name = optuna_study_path.name
            
            upload_to_huggingface(
                model_path=optuna_study_path,
                model_name=optuna_study_name,
                commit_message=commit_msg,
                path_in_repo=f'optuna_results/{optuna_study_name}'
            )
            print(f'Optuna study uploaded: {optuna_study_name}')
    else:
        print('\nSkipping Hugging Face upload (no .env file with HF_TOKEN found)')
        print('To enable auto-upload:')
        print('  1. Copy .env.template to .env')
        print('  2. Add your HF_TOKEN to .env')
        print('  3. Set HF_REPO_ID in .env (e.g., username/repo_name)')