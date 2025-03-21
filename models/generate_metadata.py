import os
import json
import argparse
from datetime import datetime

def create_model_metadata(model_name, accuracy, val_accuracy, description=None, training_params=None):
    """
    Create metadata JSON file for a model
    
    Args:
        model_name: Name of the model file (without path)
        accuracy: Training accuracy
        val_accuracy: Validation accuracy
        description: Optional description of the model
        training_params: Optional dictionary of training parameters
    """
    # Get base name without extension
    base_name = os.path.splitext(model_name)[0]
    
    # Create metadata dictionary
    metadata = {
        "model_name": model_name,
        "created_date": datetime.now().strftime("%Y-%m-%d"),
        "accuracy": accuracy,
        "val_accuracy": val_accuracy,
    }
    
    # Add optional fields if provided
    if description:
        metadata["description"] = description
        
    if training_params:
        metadata.update(training_params)
    
    # Create metadata file
    metadata_file = f"{base_name}_metadata.json"
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata file created: {metadata_file}")
    
    return metadata_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate metadata for model files")
    parser.add_argument("model_name", type=str, help="Name of the model file")
    parser.add_argument("--accuracy", type=float, required=True, help="Training accuracy")
    parser.add_argument("--val_accuracy", type=float, required=True, help="Validation accuracy")
    parser.add_argument("--description", type=str, help="Model description")
    parser.add_argument("--learning_rate", type=float, help="Learning rate used for training")
    parser.add_argument("--epochs", type=int, help="Number of epochs trained")
    parser.add_argument("--batch_size", type=int, help="Batch size used for training")
    parser.add_argument("--dropout", type=float, help="Dropout rate used")
    parser.add_argument("--base_model", type=str, help="Base model architecture")
    parser.add_argument("--test_accuracy", type=float, help="Test set accuracy if available")
    
    args = parser.parse_args()
    
    # Collect training parameters if provided
    training_params = {}
    if args.learning_rate:
        training_params["learning_rate"] = args.learning_rate
    if args.epochs:
        training_params["epochs"] = args.epochs
    if args.batch_size:
        training_params["batch_size"] = args.batch_size
    if args.dropout:
        training_params["dropout"] = args.dropout
    if args.base_model:
        training_params["base_model"] = args.base_model
    if args.test_accuracy:
        training_params["test_accuracy"] = args.test_accuracy
    
    # Create metadata file
    create_model_metadata(
        args.model_name,
        args.accuracy,
        args.val_accuracy,
        args.description,
        training_params
    )