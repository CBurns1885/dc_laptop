"""
Trace which Python files are actually imported/used in the main workflow
"""
import ast
import sys
from pathlib import Path
from collections import defaultdict

def extract_local_imports(file_path):
    """Extract local module imports from a Python file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=str(file_path))

        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module.split('.')[0])

        return imports
    except Exception as e:
        print(f"  Error parsing {file_path.name}: {e}")
        return []

def find_used_files(start_file, base_dir):
    """Recursively find all Python files imported from start_file"""
    base_dir = Path(base_dir)
    used_files = set()
    to_check = [start_file]
    checked = set()

    # Get all .py files in the directory
    all_py_files = {f.stem: f for f in base_dir.glob("*.py")}

    while to_check:
        current = to_check.pop(0)
        if current in checked:
            continue
        checked.add(current)

        current_path = base_dir / f"{current}.py"
        if not current_path.exists():
            continue

        used_files.add(current_path)

        # Extract imports from this file
        imports = extract_local_imports(current_path)

        # Check which imports are local Python files
        for imp in imports:
            if imp in all_py_files and imp not in checked:
                to_check.append(imp)

    return used_files

# Main analysis
print("="*70)
print("ANALYZING IMPORT DEPENDENCIES")
print("="*70)

base_dir = Path(__file__).parent

# Start from the main entry point
main_files = ['run_weekly', 'dc_predict', 'backtest', 'train_evaluate']

all_used = set()
for main_file in main_files:
    print(f"\n Analyzing from: {main_file}.py")
    used = find_used_files(main_file, base_dir)
    all_used.update(used)
    print(f"   Found {len(used)} files used from this entry point")

# Get all Python files
all_py_files = list(base_dir.glob("*.py"))
all_py_files = [f for f in all_py_files if f.name != 'trace_imports.py']

print(f"\n{'='*70}")
print(f"SUMMARY")
print(f"{'='*70}")
print(f"Total Python files in directory: {len(all_py_files)}")
print(f"Files used by main workflows: {len(all_used)}")
print(f"Unused files: {len(all_py_files) - len(all_used)}")

print(f"\n{'='*70}")
print(f"ESSENTIAL FILES (Used by main workflows):")
print(f"{'='*70}")
for f in sorted(all_used, key=lambda x: x.name):
    print(f"  {f.name}")

print(f"\n{'='*70}")
print(f"POTENTIALLY UNUSED FILES:")
print(f"{'='*70}")
unused = set(all_py_files) - all_used
for f in sorted(unused, key=lambda x: x.name):
    print(f"  {f.name}")

print(f"\n{'='*70}")
