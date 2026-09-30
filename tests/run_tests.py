"""Simple test runner without external dependencies."""

import importlib
import inspect
import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

test_files = [
    "test_geometry",
    "test_functional",
    "test_temporal",
    "test_lineage",
    "test_stats",
    "test_ribbon",
    "test_joint_gp",
    "test_viz",
    "test_end_to_end",
]

total = 0
passed = 0
failed = []

for mod_name in test_files:
    print(f"\nRunning {mod_name}...")
    mod = importlib.import_module(mod_name)
    for name, func in inspect.getmembers(mod, inspect.isfunction):
        if name.startswith("test_"):
            total += 1
            try:
                func()
                passed += 1
                print(f"  [PASS] {name}")
            except Exception as e:
                failed.append((f"{mod_name}.{name}", str(e)))
                print(f"  [FAIL] {name}: {e}")

print("\n" + "=" * 50)
print(f"Summary: {passed}/{total} tests passed.")
if failed:
    print(f"Failed tests:")
    for f_name, err in failed:
        print(f"  - {f_name}: {err}")
    sys.exit(1)
else:
    print("All tests passed successfully!")
    sys.exit(0)
