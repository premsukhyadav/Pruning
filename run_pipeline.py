import subprocess
import sys
import os

# List of scripts
scripts_to_run = [
    {"path": "00_save_pretrained_models.py", "enabled": False},
    {"path": "01_calculate_accuracy_base_model.py", "enabled": False},
    {"path": "02_analyze_and save_pruned_model.py", "enabled": True},
    {"path": "03_calculate_accuracy_masked_model.py", "enabled": True},
    {"path": "04_compare_results.py", "enabled": True},
]

python_executable = sys.executable  # ensures using .venv python

# Run each script
for script in scripts_to_run:
    if script["enabled"]:
        script_path = os.path.abspath(script["path"])
        print(f"\n=== Running {script_path} ===")
        try:
            # Run the script with the same Python executable as the current environment
            subprocess.run([python_executable, script_path], check=True, cwd=os.path.dirname(script_path))
            print(f"=== Finished {script_path} ===\n")
        except subprocess.CalledProcessError as e:
            print(f"Error running {script_path}: {e}")
    else:
        print(f"\n--- Skipping {script['path']} ---")
