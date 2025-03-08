import os
import random
from collections import defaultdict

def read_data_list(file_path):
    data_by_class = defaultdict(list)
    with open(file_path, 'r') as f:
        for line in f:
            path, label = line.strip().split()
            data_by_class[int(label)].append(path)
    return data_by_class

def create_imbalanced_lists(root_dir, train_base=600, val_base=100, ratio=8.0, seed=123):
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Read original lists
    train_file = os.path.join(root_dir, 'image_list/train_list.txt')
    val_file = os.path.join(root_dir, 'image_list/validation_list.txt')
    
    train_data = read_data_list(train_file)
    val_data = read_data_list(val_file)
    
    num_classes = 12
    new_train_data = []
    new_val_data = []
    
    # Print available samples
    print("\nAvailable samples for each class:")
    for label in range(num_classes):
        print(f"Class {label}:")
        print(f"  Train samples: {len(train_data[label])}")
        print(f"  Validation samples: {len(val_data[label])}")
    
    # Process each class for training set
    print("\nProcessing training set...")
    for label in range(num_classes):
        train_samples = train_data[label]
        
        # Odd-numbered classes get train_base * ratio, even-numbered get train_base
        target_samples = train_base * ratio if label % 2 == 1 else train_base
        
        if len(train_samples) >= target_samples:
            selected_samples = random.sample(train_samples, int(target_samples))
        else:
            print(f"Warning: Not enough training samples for class {label}. Using all {len(train_samples)} available samples.")
            selected_samples = train_samples
        
        new_train_data.extend((path, str(label)) for path in selected_samples)
    
    # Process each class for validation set
    print("\nProcessing validation set...")
    for label in range(num_classes):
        val_samples = val_data[label]
        
        # Even-numbered classes get val_base * ratio, odd-numbered get val_base
        # Note the opposite pattern from training set
        target_samples = val_base * ratio if label % 2 == 0 else val_base
        
        if len(val_samples) >= target_samples:
            selected_samples = random.sample(val_samples, int(target_samples))
        else:
            print(f"Warning: Not enough validation samples for class {label}. Using all {len(val_samples)} available samples.")
            selected_samples = val_samples
        
        new_val_data.extend((path, str(label)) for path in selected_samples)
    
    # Shuffle the data
    random.shuffle(new_train_data)
    random.shuffle(new_val_data)
    
    # Save new lists
    with open(os.path.join(root_dir, 'image_list/train_list_imbalanced.txt'), 'w') as f:
        for path, label in new_train_data:
            f.write(f"{path} {label}\n")
            
    with open(os.path.join(root_dir, 'image_list/validation_list_imbalanced.txt'), 'w') as f:
        for path, label in new_val_data:
            f.write(f"{path} {label}\n")
    
    # Print final distributions
    print("\nFinal training set distribution:")
    for label in range(num_classes):
        count = sum(1 for p, l in new_train_data if int(l) == label)
        expected = train_base * ratio if label % 2 == 1 else train_base
        print(f"Class {label}: {count} (Expected: {int(expected)})")
        
    print("\nFinal validation set distribution:")
    for label in range(num_classes):
        count = sum(1 for p, l in new_val_data if int(l) == label)
        expected = val_base * ratio if label % 2 == 0 else val_base
        print(f"Class {label}: {count} (Expected: {int(expected)})")

if __name__ == "__main__":
    root_dir = '/data/home/yilingliu/VisDA-2017'
    create_imbalanced_lists(
        root_dir,
        train_base=600,  # Base samples for training (600 vs 4800)
        val_base=100,    # Base samples for validation (100 vs 800)
        ratio=8.0,
        seed=123
    ) 