import json
import os

def load_labels(filepath='datasets/customLabels.json'):
    """Load labels from a JSON file."""
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return {}
    
    try:
        with open(filepath, 'r') as f:
            labels = json.load(f)
        return labels
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return {}
    except Exception as e:
        print(f"Unexpected error: {e}")
        return {}

def save_labels(labels, filepath='datasets/customLabels.json'):
    """Save labels to a JSON file."""
    try:
        with open(filepath, 'w') as f:
            json.dump(labels, f, indent=4)
    except Exception as e:
        print(f"Error saving labels: {e}")

def ensure_directory_exists(directory):
    """Ensure a directory exists, create if not."""
    if not os.path.exists(directory):
        os.makedirs(directory)
