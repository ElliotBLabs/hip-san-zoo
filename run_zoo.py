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

        print(f"\n=== Running {san.upper()} Zoo Validation ===")

        for contains_ub in ["has_ub", "no_ub"]:
            category_dir = san_dir / contains_ub
            
            if not category_dir.exists():
                continue

            print(f"\n  [{contains_ub.upper()}]")

            for test_file in category_dir.iterdir():
                       
                with tempfile.TemporaryDirectory() as tmp:
                    run_dir = Path(tmp) 
                    target_name = test_file.name
                    
                    shutil.copy(test_file, run_dir / target_name)
                    shutil.copytree(INTERESTING_DIR, run_dir, dirs_exist_ok=True)

                    # shorten name
                    # 'asan_heap_oob.hip' -> 'heap_oob'
                    display_name = target_name.replace(f"{san}_", "").replace(".hip", "")
                    
                    # padding for output alignment
                    print(f"    {display_name:<25} ", end="", flush=True)
                    
                    res = subprocess.run(["python3", "interesting.py", target_name], 
                                        cwd=run_dir, capture_output=True, text=True)
                    
                    if res.returncode == 0:
                        combined = res.stdout + res.stderr
                        print(combined)
                    else:
                        print("[FAIL]")
                        output = res.stdout + res.stderr
                        if output:
                            print(f"\n      --- FAILURE LOG ---\n      {output.strip().replace(chr(10), chr(10) + '      ')}\n      ------------------")

if __name__ == "__main__":
    run_test()