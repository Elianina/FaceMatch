# ============================================================================
# Imports and Dependencies
# ============================================================================

import os
import sys


# ============================================================================
# Terminal Output Logging Functions
# ============================================================================

def start_logging(model_name='DenseNet-121', log_dir='../../Results/logs/'):

    os.makedirs(log_dir, exist_ok=True)

    log_file_path = os.path.join(log_dir, f"{model_name}_training_log.txt")

    original_stdout = sys.stdout

    log_file = open(log_file_path, 'w', encoding='utf-8')

    def write_to_both(text):
        original_stdout.write(text)
        log_file.write(text)
        original_stdout.flush()
        log_file.flush()

    class DualWriter:
        def write(self, text):
            write_to_both(text)

        def flush(self):
            original_stdout.flush()
            log_file.flush()

    sys.stdout = DualWriter()

    return original_stdout, log_file, log_file_path


def log_output(text, log_file, original_stdout):
    original_stdout.write(text)
    log_file.write(text)
    original_stdout.flush()
    log_file.flush()


def stop_logging(original_stdout, log_file, log_file_path):

    sys.stdout = original_stdout

    log_file.close()

    print(f"Training log saved to: {log_file_path}")

    return log_file_path


# ============================================================================
# Capture Output at End
# ============================================================================

def save_terminal_output_to_file(output_text, model_name='DenseNet-121', log_dir='../../Results/logs/'):

    os.makedirs(log_dir, exist_ok=True)

    log_file_path = os.path.join(log_dir, f"{model_name}_log.txt")

    with open(log_file_path, 'w', encoding='utf-8') as f:
        f.write(output_text)

    print(f"Training log saved to: {log_file_path}")
    return log_file_path