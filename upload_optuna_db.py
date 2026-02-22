'''Throwaway script to upload Optuna database to HuggingFace.

This uploads the shared cnn_optimization.db database which contains both:
- architecture_optimization study (from notebook 04)
- cnn_training_optimization study (from notebook 05)
'''

from pathlib import Path
import sys
import os

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent / 'notebooks'))

import helper_functions as hf

# Define path to shared database
data_dir = Path(__file__).parent / 'data' / 'pytorch'
cnn_opt_db = data_dir / 'cnn_optimization.db'

# Upload the shared optimization database
if cnn_opt_db.exists():
    print('Uploading cnn_optimization.db (shared database for both studies)...')
    hf.upload_to_huggingface(
        model_path=cnn_opt_db,
        model_name='cnn_optimization.db',
        commit_message='Upload Optuna optimization studies database'
    )
    print('Upload complete!')
else:
    print(f'File not found: {cnn_opt_db}')

print('\nDone!')
