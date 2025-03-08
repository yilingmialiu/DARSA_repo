import os

def trim_path(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    trimmed_lines = []
    for line in lines:
        # Split into path and label
        parts = line.strip().split()
        path = parts[0]
        label = parts[-1]
        
        # Remove 'VisDA-2017/' from the beginning if it exists
        if 'VisDA-2017/' in path:
            path = path.split('VisDA-2017/')[-1]
        
        # Create new line
        new_line = f"{path} {label}\n"
        trimmed_lines.append(new_line)
    
    # Write trimmed lines to new file
    with open(output_file, 'w') as f:
        f.writelines(trimmed_lines)

def main():
    root_dir = '/data/home/yilingliu/VisDA-2017'
    image_list_dir = os.path.join(root_dir, 'image_list')
    
    # Create image_list directory if it doesn't exist
    os.makedirs(image_list_dir, exist_ok=True)
    
    # Process train list
    train_input = os.path.join(root_dir, 'image_list/train_list_original.txt')
    train_output = os.path.join(root_dir, 'image_list/train_list.txt')
    if os.path.exists(train_input):
        trim_path(train_input, train_output)
        print(f"Processed train list: {train_output}")
    
    # Process validation list
    val_input = os.path.join(root_dir, 'image_list/validation_list_original.txt')
    val_output = os.path.join(root_dir, 'image_list/validation_list.txt')
    if os.path.exists(val_input):
        trim_path(val_input, val_output)
        print(f"Processed validation list: {val_output}")

if __name__ == '__main__':
    main() 