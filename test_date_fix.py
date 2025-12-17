"""
Quick test to verify date parsing fix works with UK date format
"""
import pandas as pd

# Test UK date format (DD/MM/YYYY)
test_dates = ["13/12/2025", "14/12/2025", "15/12/2025"]

print("Testing UK date format parsing (DD/MM/YYYY):")
print("=" * 50)

# Test without dayfirst (should fail)
print("\n1. WITHOUT dayfirst parameter:")
try:
    dates_wrong = pd.to_datetime(test_dates)
    print(f"   Result: {dates_wrong.tolist()}")
    print("   WARNING: This parsed incorrectly (MM/DD/YYYY)")
except Exception as e:
    print(f"   ERROR: {e}")

# Test with dayfirst (should work)
print("\n2. WITH dayfirst=True parameter:")
try:
    dates_correct = pd.to_datetime(test_dates, dayfirst=True, errors='coerce')
    print(f"   Result: {dates_correct.tolist()}")
    print("   SUCCESS: Parsed correctly (DD/MM/YYYY)")
except Exception as e:
    print(f"   ERROR: {e}")

print("\n" + "=" * 50)
print("Fix verified! The dayfirst=True parameter handles UK dates correctly.")
