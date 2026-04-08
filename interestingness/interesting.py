#!/usr/bin/env python3
import subprocess
import sys
import os
from pathlib import Path

# ANSI color codes
GREEN = '\033[92m'
RED = '\033[91m'
RESET = '\033[0m'

# Get the current working directory
WORK_DIR = Path.cwd()

# input file as a command line arg
INPUT_FILE = Path(sys.argv[1]).resolve()
SCRIPT_DIR = Path(__file__).resolve().parent
HIP_CPU_DIR = SCRIPT_DIR / "hip-cpu"
HIP_CPU_INCLUDE = HIP_CPU_DIR / "include"
MSAN_IGNORE_LIST = SCRIPT_DIR / "msan_ignore.txt"

# expect san type name in the name of the test
SAN_TYPE = (
    "asan" if "asan" in INPUT_FILE.name.lower()
    else "ubsan" if "ubsan" in INPUT_FILE.name.lower()
    else "msan"
)

def main():
    # compile with hip-cpu
    CPU_BASE = ["clang++", "-x", "c++", INPUT_FILE, "-I", HIP_CPU_INCLUDE, "-pthread", "-ltbb", "-O0", "-g"]
    
    # SANITIZER flags
    # fno-sanitize-recover=all: crash as soon as a sanitizer 
    # halt_on_error/abort_on_error: stop and crash on sanitizer detection
    # 1. ASAN:
    #    - fsanitize=address: run with asan
    #    - detect_leaks=0: disables LeakSanitizer memory leaks are ok
    #
    # 2. UBSAN:
    #    - fsanitize=undefined: run with ubsan
    #
    # 3. MSAN:
    #    - fsanitize=memory: run with msan
    #    - fsanitize-ignorelist: if MSAN triggers here ignore it, for example an external library we do not care about testing
    configs = {
        "asan": {
            "cmd": CPU_BASE + ["-fsanitize=address", "-fno-sanitize-recover=all", "-o", "bin_asan"],
            "env": {"ASAN_OPTIONS": "halt_on_error=1:abort_on_error=1:detect_leaks=0"}
        },
        "ubsan": {
            "cmd": CPU_BASE + ["-fsanitize=undefined", "-fno-sanitize-recover=all", "-o", "bin_ubsan"],
            "env": {"UBSAN_OPTIONS": "halt_on_error=1:abort_on_error=1"}
        },
        "msan": {
            "cmd": CPU_BASE + ["-fsanitize=memory", f"-fsanitize-ignorelist={MSAN_IGNORE_LIST}", "-fno-sanitize-recover=all", "-o", "bin_msan"],
            "env": {"MSAN_OPTIONS": "halt_on_error=1:abort_on_error=1"}
        }
    }

    target = configs[SAN_TYPE]

    bin_path = WORK_DIR / f"bin_{SAN_TYPE}"

    comp = subprocess.run(target["cmd"], capture_output=True, text=True)
    if comp.returncode != 0:
        # compilation failed 
        print(f"{RED}[FAIL]{RESET} {SAN_TYPE}", file=sys.stderr)
        print(comp.stderr, file=sys.stderr)
        sys.exit(1)

    try:
        env = os.environ.copy()
        env.update(target["env"])

        res = subprocess.run([bin_path], capture_output=True, text=True, timeout=10, env=env)
        
        combined_out = res.stdout + res.stderr
        out_lower = combined_out.lower()
        
        if (res.returncode != 0):
            # non-zero exit code so sanitizer hopefully has worked
            for line in combined_out.splitlines():
                if "sanitizer" in line.lower():
                    relevant_line = line.strip()
                    break
            
            print(f"{GREEN}[PASS]{RESET} {relevant_line}")
            sys.exit(0) 
        else:
            sys.exit(1) 
            
    except subprocess.TimeoutExpired:
        print(f"{RED}[TIMEOUT]{RESET}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        sys.exit(1)

if __name__ == "__main__":
    main()