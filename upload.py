#!/usr/bin/env python3
"""
Script to upload models to Hugging Face Hub
Supports various model formats including PyTorch, TensorFlow, and ONNX
"""

import os
import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder, upload_file
from huggingface_hub.utils import RepositoryNotFoundError
import json

def setup_hf_token():
    """
    Setup Hugging Face token for authentication
    """
    token = os.getenv('HF_TOKEN')
    if not token:
        token = input("Enter your Hugging Face token: ")
        print("Consider setting HF_TOKEN environment variable for future use")
    return token

def create_model_card(model_name, description="", tags=None, license="apache-2.0"):
    """
    Create a basic model card (README.md) for the model
    """
    # Always use empty array format for tags to avoid YAML issues
    model_card = f"""---
license: {license}
tags: []
---

# {model_name}

{description}

## Model Description

This model was uploaded using an automated script.

## Usage

```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("{model_name}")
tokenizer = AutoTokenizer.from_pretrained("{model_name}")
```

## Training Details

Add training details here...

## Citation

Add citation information here...
"""
    return model_card

def upload_model_to_hf(
    model_path,
    repo_name,
    token,
    organization=None,
    private=False,
    description="",
    tags=None,
    license="apache-2.0",
    create_model_card_file=True
):
    """
    Upload a model to Hugging Face Hub
    
    Args:
        model_path (str): Path to the model directory or file
        repo_name (str): Name of the repository on Hugging Face
        token (str): Hugging Face authentication token
        organization (str, optional): Organization name (if uploading to org)
        private (bool): Whether to create a private repository
        description (str): Description of the model
        tags (list): List of tags for the model
        license (str): License for the model
        create_model_card_file (bool): Whether to create a model card
    """
    
    # Initialize HF API
    api = HfApi(token=token)
    
    # Get current user info to construct proper repo name
    try:
        user_info = api.whoami(token=token)
        username = user_info['name']
        print(f"Authenticated as: {username}")
    except Exception as e:
        print(f"Warning: Could not get user info: {e}")
        username = None
    
    # Create full repo name
    if organization:
        full_repo_name = f"{organization}/{repo_name}"
    elif username and '/' not in repo_name:
        full_repo_name = f"{username}/{repo_name}"
    else:
        full_repo_name = repo_name
    
    print(f"Uploading model to: {full_repo_name}")
    
    try:
        # Try to create the repository
        repo_url = create_repo(
            repo_id=full_repo_name,
            token=token,
            private=private,
            repo_type="model",
            exist_ok=True
        )
        print(f"Repository {full_repo_name} created/verified successfully")
        print(f"Repository URL: {repo_url}")
        
        # Wait a moment for repo to be fully created
        import time
        time.sleep(2)
        
    except Exception as e:
        print(f"Error creating repository: {e}")
        return False
    
    model_path = Path(model_path)
    
    try:
        if model_path.is_dir():
            # Upload entire directory
            print(f"Uploading directory: {model_path}")
            
            # Create model card if requested
            if create_model_card_file:
                model_card_path = model_path / "README.md"
                # Always overwrite existing README.md to ensure proper formatting
                model_card_content = create_model_card(
                    full_repo_name, description, tags, license
                )
                with open(model_card_path, 'w') as f:
                    f.write(model_card_content)
                print("Created/updated model card (README.md)")
            
            # Upload the folder
            upload_folder(
                folder_path=str(model_path),
                repo_id=full_repo_name,
                token=token,
                repo_type="model",
                commit_message=f"Upload {repo_name} model"
            )
            
        else:
            # Upload single file
            print(f"Uploading file: {model_path}")
            upload_file(
                path_or_fileobj=str(model_path),
                path_in_repo=model_path.name,
                repo_id=full_repo_name,
                token=token,
                repo_type="model",
                commit_message=f"Upload {model_path.name}"
            )
            
            # Create model card if requested
            if create_model_card_file:
                model_card_content = create_model_card(
                    full_repo_name, description, tags, license
                )
                # Upload model card as separate file
                upload_file(
                    path_or_fileobj=model_card_content.encode(),
                    path_in_repo="README.md",
                    repo_id=full_repo_name,
                    token=token,
                    repo_type="model",
                    commit_message="Add model card"
                )
        
        print(f"✅ Model uploaded successfully to: https://huggingface.co/{full_repo_name}")
        return True
        
    except Exception as e:
        print(f"❌ Error uploading model: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Upload models to Hugging Face Hub")
    parser.add_argument("model_path", help="Path to model directory or file")
    parser.add_argument("repo_name", help="Repository name on Hugging Face")
    parser.add_argument("--organization", "-o", help="Organization name (optional)")
    parser.add_argument("--private", action="store_true", help="Create private repository")
    parser.add_argument("--description", "-d", default="", help="Model description")
    parser.add_argument("--tags", "-t", nargs="*", default=[], help="Model tags")
    parser.add_argument("--license", "-l", default="apache-2.0", help="Model license")
    parser.add_argument("--no-model-card", action="store_true", help="Skip creating model card")
    parser.add_argument("--token", help="Hugging Face token (or set HF_TOKEN env var)")
    
    args = parser.parse_args()
    
    # Get token
    token = args.token or setup_hf_token()
    
    # Validate model path
    if not Path(args.model_path).exists():
        print(f"❌ Error: Model path '{args.model_path}' does not exist")
        return
    
    # Upload model
    success = upload_model_to_hf(
        model_path=args.model_path,
        repo_name=args.repo_name,
        token=token,
        organization=args.organization,
        private=args.private,
        description=args.description,
        tags=args.tags,
        license=args.license,
        create_model_card_file=not args.no_model_card
    )
    
    if success:
        print("🎉 Upload completed successfully!")
    else:
        print("💥 Upload failed!")

if __name__ == "__main__":
    main()