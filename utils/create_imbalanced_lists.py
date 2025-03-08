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

def create_imbalanced_lists(root_dir, ratio=8.0, base_samples_per_class=3000, seed=123):
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Read original lists
    train_file = os.path.join(root_dir, 'image_list/train_list.txt')
    val_file = os.path.join(root_dir, 'image_list/validation_list.txt')
    
    train_data = read_data_list(train_file)
    val_data = read_data_list(val_file)
    
    # Split classes into two groups
    num_classes = 12
    first_half = list(range(num_classes//2, num_classes))  # 6-11 
    second_half = list(range(0, num_classes//2))  # 0-5
    
    # Create new data lists
    new_train_data = []
    new_val_data = []
    
    # Print original counts
    print("\nOriginal counts before sampling:")
    for label in range(num_classes):
        print(f"Class {label}:")
        print(f"  Train samples available: {len(train_data[label])}")
        print(f"  Val samples available: {len(val_data[label])}")
        print(f"  Desired train samples for ratio {ratio}: {len(val_data[label]) * ratio if label < 6 else len(val_data[label]) / ratio}")
    
    # Process each class
    for label in range(num_classes):
        train_samples = train_data[label]
        val_samples = val_data[label]
        
        if label in first_half:
            # First half: source should have exactly 8x samples
            if len(train_samples) >= base_samples_per_class * ratio:
                # We have enough samples for the ideal case
                val_count = base_samples_per_class
                train_count = int(val_count * ratio)
            else:
                # We don't have enough samples, but maintain ratio
                train_count = len(train_samples)
                val_count = int(train_count / ratio)
        else:
            # Second half: source should have exactly 1/8x samples
            if len(train_samples) >= int(base_samples_per_class / ratio) and len(val_samples) >= base_samples_per_class:
                # We have enough samples for both train and val
                val_count = base_samples_per_class
                train_count = int(val_count / ratio)
            else:
                # Use minimum of what we can get from train or val while maintaining ratio
                train_count = min(len(train_samples), int(len(val_samples) / ratio))
                val_count = int(train_count * ratio)
        
        selected_val = random.sample(val_samples, min(val_count, len(val_samples)))
        selected_train = random.sample(train_samples, train_count)
        
        new_train_data.extend((path, str(label)) for path in selected_train)
        new_val_data.extend((path, str(label)) for path in selected_val)
    
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
    
    # Print statistics
    print("\nClass distribution in training set:")
    for label in range(num_classes):
        count = sum(1 for p, l in new_train_data if int(l) == label)
        print(f"Class {label}: {count} samples")
        
    print("\nClass distribution in validation set:")
    for label in range(num_classes):
        count = sum(1 for p, l in new_val_data if int(l) == label)
        print(f"Class {label}: {count} samples")

if __name__ == "__main__":
    root_dir = '/data/home/yilingliu/VisDA-2017'
    create_imbalanced_lists(root_dir, ratio=8.0, seed=123) 