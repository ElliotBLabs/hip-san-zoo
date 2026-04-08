import os
import shutil
import subprocess
import tempfile
from pathlib import Path

ZOO_ROOT = Path(".").absolute()
INTERESTING_DIR = ZOO_ROOT / "interestingness" 

def run_test():
    for san in ["asan", "ubsan", "msan"]:
        san_dir = ZOO_ROOT / "tests" / san

        print(f"\nRunning {san} Zoo Validation")

        for test_file in san_dir.iterdir():       
            with tempfile.TemporaryDirectory() as tmp:
                run_dir = Path(tmp) 
                target_name = test_file.name
                
                shutil.copy(test_file, run_dir / target_name)
                
                shutil.copytree(INTERESTING_DIR, run_dir, dirs_exist_ok=True)

                print(f"Verifying {target_name}...", end=" ", flush=True)
                
                res = subprocess.run(["python3", "interesting.py", target_name], 
                                    cwd=run_dir, capture_output=True, text=True)
                
                if res.returncode == 0:
                    combined = res.stdout + res.stderr
                    for line in combined.splitlines():
                        if "[PASS]" in line:
                            print(line)
                else:
                    print("FAIL")
                    output = res.stdout + res.stderr
                    if output:
                        print(f"--- FAILURE LOG ---\n{output.strip()}\n------------------")

if __name__ == "__main__":
    run_test()