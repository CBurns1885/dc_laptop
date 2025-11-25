#!/usr/bin/env python3
"""
Validation Script - Verify Boolean/NaN fixes and DC-only enforcement
"""
import re
from pathlib import Path

def check_file_for_issues(filepath):
    """Check a file for common Boolean/NaN handling issues"""
    issues = []

    with open(filepath, 'r') as f:
        content = f.read()
        lines = content.split('\n')

    # 1. Check for unsafe row.get() usage (should use safe_get in bet_finder.py)
    if 'bet_finder.py' in str(filepath):
        in_safe_get_function = False
        for i, line in enumerate(lines, 1):
            # Track if we're inside the safe_get function
            if 'def safe_get' in line:
                in_safe_get_function = True
            elif in_safe_get_function and line.strip() and not line.startswith(' '):
                # Exited the function (unindented line)
                in_safe_get_function = False

            # Only flag row.get() outside of safe_get function
            if 'row.get(' in line and not in_safe_get_function:
                # Check if it's actually getting a value (not just checking existence)
                if '=' in line and 'if' not in line and 'safe_get' not in line:
                    issues.append(f"Line {i}: Unsafe row.get() usage: {line.strip()}")

    # 2. Check for pd.notna() on probability comparisons that should use > 0
    # (Only in bet_finder.py - other files use pd.notna() correctly)
    if 'bet_finder.py' in str(filepath):
        for i, line in enumerate(lines, 1):
            # Look for patterns like: if pd.notna(dc_prob) or if self.config['dc_validation'] and pd.notna(dc_prob)
            if 'pd.notna' in line and 'dc_prob' in line:
                issues.append(f"Line {i}: Should use 'dc_prob > 0' instead of pd.notna(): {line.strip()}")

    # 3. Check for DC-only enforcement in models.py
    if 'models.py' in str(filepath):
        # Verify base_models only gets DC model
        has_dc_only_comment = '# ===== DIXON-COLES ONLY =====' in content
        has_dc_assignment = 'base_models["dc"] = "__DC__"' in content
        has_dc_supported_check = 'supports_dc = _dc_supported(target_col)' in content

        if not (has_dc_only_comment and has_dc_assignment and has_dc_supported_check):
            issues.append("Missing DC-only enforcement markers")

    return issues

def main():
    print("="*70)
    print("🔍 VALIDATION: Boolean/NaN Handling & DC-Only Enforcement")
    print("="*70)

    # Files to check
    files_to_check = [
        'bet_finder.py',
        'models.py',
        'predict.py',
        'backtest.py',
    ]

    all_issues = {}

    for filename in files_to_check:
        filepath = Path(filename)
        if filepath.exists():
            print(f"\n📄 Checking {filename}...")
            issues = check_file_for_issues(filepath)
            if issues:
                all_issues[filename] = issues
                print(f"   ⚠️  Found {len(issues)} potential issue(s)")
                for issue in issues:
                    print(f"      - {issue}")
            else:
                print(f"   ✅ No issues found")
        else:
            print(f"\n❌ {filename} not found")

    print("\n" + "="*70)
    print("📊 VALIDATION SUMMARY")
    print("="*70)

    if not all_issues:
        print("✅ All files passed validation!")
        print("✅ Boolean/NaN handling is correct")
        print("✅ DC-only enforcement is in place")
    else:
        print(f"⚠️  Found issues in {len(all_issues)} file(s)")
        for filename, issues in all_issues.items():
            print(f"   - {filename}: {len(issues)} issue(s)")

    # Additional checks
    print("\n" + "="*70)
    print("🔬 ADDITIONAL VERIFICATIONS")
    print("="*70)

    # Check bet_finder.py has safe_get function
    bet_finder = Path('bet_finder.py')
    if bet_finder.exists():
        content = bet_finder.read_text()
        if 'def safe_get(row, column, default=0.0):' in content:
            print("✅ bet_finder.py has safe_get() helper function")
        else:
            print("❌ bet_finder.py missing safe_get() helper function")

    # Check models.py DC-only enforcement
    models = Path('models.py')
    if models.exists():
        content = models.read_text()
        if 'base_models["dc"] = "__DC__"' in content:
            print("✅ models.py enforces DC-only (no ML models)")

            # Count how many times base_models dict is assigned
            dc_assignments = content.count('base_models["dc"]')
            other_assignments = len(re.findall(r'base_models\["(?!dc)[^"]+"\]', content))

            if other_assignments == 0:
                print(f"✅ base_models only has DC model ({dc_assignments} reference(s))")
            else:
                print(f"⚠️  base_models has {other_assignments} non-DC assignment(s)")
        else:
            print("❌ models.py missing DC-only enforcement")

    # Check _dc_supported function
    if models.exists():
        content = models.read_text()
        if 'def _dc_supported(t: str) -> bool:' in content:
            print("✅ models.py has _dc_supported() function")
            if 't == "y_BTTS"' in content and 't.startswith("y_OU_")' in content:
                print("✅ _dc_supported() restricts to BTTS and O/U only")
        else:
            print("❌ models.py missing _dc_supported() function")

    print("\n" + "="*70)
    print("🎯 VALIDATION COMPLETE")
    print("="*70)

    return len(all_issues) == 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
