"""
Terminal Output Logger

This module captures the terminal output during the model's training sessions.
It enables the display of training progress in the console while saving its
output to a timestamped log file for the purpose of review, and documentation.

Authors: Carl Fokstuen, YuTing Lee, Mark Malady, Nayani Samaranayake, Vishal Cheroor Ravi
Course: COSC595 Information Technology Project - Implementation
Institution: The University of New England (UNE)
Date: September, 2025
"""


# ============================================================================
# Imports and Dependencies
# ============================================================================

import os
import sys


# ============================================================================
# Terminal Output Logging Functions
# ============================================================================

def start_logging(model_name='DenseNet-121', log_dir='../../Results/logs/'):
    """ Initialises logging to both the console and the file."""

    # Creates the log's directory structure if it does not exist prior
    os.makedirs(log_dir, exist_ok=True)

    # Constructs the full path for the log's file
    log_file_path = os.path.join(log_dir, f"{model_name}_training_log.txt")

    # Preserves the reference to the original `stdout`
    original_stdout = sys.stdout

    # Opens the log's file with UTF-8 encoding for writing
    log_file = open(log_file_path, 'w', encoding='utf-8')

    def write_to_both(text):
        # An internal helper function that writes text to the console and log's file
        original_stdout.write(text)
        log_file.write(text)
        original_stdout.flush()
        log_file.flush()

    class DualWriter:
        """A custom `stdout` wrapper that writes to the console and the log's file"""

        def write(self, text):
            """Writes the method called by `print()` and `sys.stdout.write()"""
            write_to_both(text)

        def flush(self):
            """Uses a flush method that ensures output on both streams"""
            original_stdout.flush()
            log_file.flush()

    # Redirects the `sys.stdout` to the dual writer
    sys.stdout = DualWriter()

    # Returns any references required for cleanup
    return original_stdout, log_file, log_file_path


def log_output(text, log_file, original_stdout):
    """Writes text to both console and log file."""

    # Writes to the console
    original_stdout.write(text)
    # Writes to the log file
    log_file.write(text)
    # Forces and output to both streams
    original_stdout.flush()
    log_file.flush()


def stop_logging(original_stdout, log_file, log_file_path):
    """Stops logging and restores the console's normal output."""

    # Restores the original `stdout`
    sys.stdout = original_stdout

    # Closes the log's file to flush any buffers/release file lock
    log_file.close()

    # Informs the user via the terminal where the log was saved
    print(f"Training log saved to: {log_file_path}")

    # The return path
    return log_file_path


# ============================================================================
# Capture Output at End
# ============================================================================

def save_terminal_output_to_file(output_text, model_name='DenseNet-121', log_dir='../../Results/logs/'):
    """Saves the pre-captured terminal output to the log file"""

    # Creates the log directory structure if it does not already exist
    os.makedirs(log_dir, exist_ok=True)

    # Constructs the full log file path
    log_file_path = os.path.join(log_dir, f"{model_name}_log.txt")

    # Writes the output text to the file
    with open(log_file_path, 'w', encoding='utf-8') as f:
        f.write(output_text)

    # Informs the user where the log was saved
    print(f"Training log saved to: {log_file_path}")

    # The return path
    return log_file_path