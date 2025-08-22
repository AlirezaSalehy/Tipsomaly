
import logging
import os
import datetime

def read_train_args(training_path):
    # Check if args.txt exists in the training_path
    args_file_path = os.path.join(training_path, 'args.txt')
    configurations_dict = {}  # Dictionary to store configurations
    last_config = {}
    if os.path.exists(args_file_path):
        with open(args_file_path, 'r') as f:
            # Read the entire content of the file
            file_content = f.read().strip()
        
        # Split the content into different configurations based on the 'Timestamp' keyword
        configurations = file_content.split('Timestamp:')
        
        # Iterate over each configuration to populate the dictionary
        for config in configurations:
            if config.strip():
                # Convert the configuration to a dictionary
                file_args_dict = {}
                for line in config.strip().split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        file_args_dict[key.strip()] = value.strip()
                
                # Store the configuration in the dictionary with a unique key
                timestamp = file_args_dict.get('Timestamp', 'Unknown')
                configurations_dict[timestamp] = file_args_dict
                
        if configurations_dict:
            last_timestamp = max(configurations_dict.keys())
            last_config = configurations_dict[last_timestamp]
    else:
        print('train args does not exists')
    return last_config

def save_args_to_file(args, command, log_dir=''):
    args_file_path = os.path.join(args.save_path, log_dir, 'args.txt')
    os.makedirs(os.path.dirname(args_file_path), exist_ok=True)
    if os.path.exists(args_file_path):
        print(f"Warning: The file {args_file_path} already exists and will be overwritten.")
    with open(args_file_path, 'a') as f:  # Change 'w' to 'a' to append to the file
        f.write("\n")  # Add new line before writing to the file
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")  # Add timestamp
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
        f.write(f"Command arguments: {' '.join(command)}\n")  # Add the command arguments to the file

def get_logger(save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    txt_path = os.path.join(save_path, 'log.txt')
    # logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.WARNING)
    logger = logging.getLogger('test')
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                    datefmt='%y-%m-%d %H:%M:%S')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(txt_path, mode='a')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger