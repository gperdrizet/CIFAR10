'''Helper functions for model management and Hugging Face Hub integration.'''

from pathlib import Path
import os
import torch
from huggingface_hub import HfApi, hf_hub_download


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
    
    Args:
        model_path: Path to the local model file
        model_name: Name of the model file (e.g., 'dnn.pth')
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
            
            model = torch.load(model_path, map_location=device, weights_only=False)
            print(f'Model loaded from {model_path}')
            return model
        
        elif model_source == 'huggingface':

            repo_id = repo_id or os.getenv('HF_REPO_ID')

            if not repo_id:
                print('HF_REPO_ID not set in environment')
                return None
            
            print(f'Downloading model from Hugging Face: {repo_id}/{model_name}')
            
            # Download model to cache
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=model_name,
                repo_type=repo_type
            )
            
            # Load model
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
    repo_type: str = 'model'
) -> str:
    '''Upload a model to Hugging Face Hub.
    
    Args:
        model_path: Path to the model file to upload
        model_name: Name for the model in the repository (e.g., 'dnn.pth')
        commit_message: Optional commit message (defaults to model name)
        repo_id: Optional repository ID (uses HF_REPO_ID env var if not provided)
        repo_type: Type of repository (default: 'model')
    
    Returns:
        URL of the uploaded file, or None if upload failed
    '''
    try:
        api = HfApi()
        repo_id = repo_id or os.getenv('HF_REPO_ID')

        if not repo_id:
            raise ValueError('HF_REPO_ID not set in environment')
            
        commit_message = commit_message or f'Upload {model_name}'
        
        # Upload file to repository
        url = api.upload_file(
            path_or_fileobj=str(model_path),
            path_in_repo=model_name,
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
    model_name: str
) -> None:
    '''Save a model locally and optionally upload to Hugging Face Hub.
    Automatically uploads if .env file exists with HF_TOKEN.
    
    Args:
        model: The PyTorch model to save
        model_path: Path where to save the model locally
        model_name: Name of the model file (e.g., 'dnn.pth')
    '''
    # Save model locally
    torch.save(model, model_path)

    print(f'Model saved to: {model_path}')
    
    # Check if we should upload to HuggingFace
    if should_upload_to_huggingface():

        print('\nUploading to Hugging Face Hub...')

        # Simple commit message indicating which model was updated
        base_name = model_name.replace('.pth', '')
        commit_msg = f'Updated {base_name}'
        
        upload_to_huggingface(
            model_path=model_path,
            model_name=model_name,
            commit_message=commit_msg
        )

    else:
        print('\nSkipping Hugging Face upload (no .env file with HF_TOKEN found)')
        print('To enable auto-upload:')
        print('  1. Copy .env.template to .env')
        print('  2. Add your HF_TOKEN to .env')
        print('  3. Set HF_REPO_ID in .env (e.g., username/repo_name)')