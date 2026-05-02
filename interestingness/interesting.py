#!/usr/bin/env python3
import subprocess
import sys
import os
import re
from pathlib import Path

# Fun colour codes for output
GREEN = '\033[92m'
RED = '\033[91m'
RESET = '\033[0m'
YELLOW = '\033[93m' 
CYAN = '\033[96m' 

WORK_DIR = Path.cwd()

# the test to conduct on
INPUT_FILE = Path(sys.argv[1]).resolve()

SCRIPT_DIR = Path(__file__).resolve().parent
HIP_CPU_DIR = SCRIPT_DIR / "hip-cpu"
HIP_CPU_INCLUDE = HIP_CPU_DIR / "include"
MSAN_IGNORE_LIST = SCRIPT_DIR / "msan_ignore.txt"

# PRE: sanitiser type name in the name of the test
SAN_TYPE = (
    "asan" if "asan" in INPUT_FILE.name.lower()
    else "ubsan" if "ubsan" in INPUT_FILE.name.lower()
    else "msan"
)

def parse_sanitiser_output(output):
    # regex looks for a filename ending in h, c, cc, cpp, or hip, followed by :line:col
    source_loc_pattern = r"([^/\s]+\.(?:h|c|cc|cpp|hip|hpp):\d+(?::\d+)?)"

    for line in output.splitlines():
        # UBSan useful reporting is of the form 'runtime error: XXXXX'
        if "runtime error:" in line:
            ub_desc = line.split("runtime error:")[1].strip()

            # line of interest on the same line
            loc_match = re.search(source_loc_pattern, line)
            loc_str = f" at {YELLOW}{loc_match.group(1)}{RESET}" if loc_match else ""
            return f"{CYAN}UBSan:{RESET} {ub_desc}{loc_str}"

        # MSan and ASan useful reporting is of the form '{Address|Memory}Sanitizer: XXXXX'
        san_match = re.search(r"(AddressSanitizer|MemorySanitizer):\s*([\w-]+)", line)
        if san_match:
            san_name = "ASan" if "Address" in san_match.group(1) else "MSan"
            
            # M/ASan usually puts the location on a different line so rescan whole thing
            loc_match = re.search(source_loc_pattern, output)
            loc_str = f" at {YELLOW}{loc_match.group(1)}{RESET}" if loc_match else ""
            
            return f"{CYAN}{san_name}:{RESET} {san_match.group(2)}{loc_str}"

    return "Sanitiser triggered (no useful summary found)"

def main():
    # compile with hip-cpu
    CPU_BASE = ["clang++", "-x", "c++", INPUT_FILE, "-I", HIP_CPU_INCLUDE, "-pthread", "-ltbb", "-O0", "-g"]
    
    # SANITIZER flags
    # fno-sanitize-recover=all: crash as soon as a sanitizer 
    # halt_on_error/abort_on_error: stop and crash on sanitizer detection
    # ASAN:
    #    - fsanitize=address: run with asan
    #    - detect_leaks=0: disables LeakSanitizer memory leaks are ok
    #
    # UBSAN:
    #    - fsanitize=undefined: run with ubsan
    #
    # MSAN:
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
        
        if (res.returncode != 0):
            # non-zero exit code so sanitiser hopefully has worked
            summary = parse_sanitiser_output(combined_out)
            print(f"{GREEN}[PASS]{RESET} {summary}")
            sys.exit(0) 
        else:
            print(f"{RED}[FAIL]{RESET} {INPUT_FILE.name} (No sanitizer error detected)")
            sys.exit(1) 
            
    except subprocess.TimeoutExpired:
        print(f"{RED}[TIMEOUT]{RESET}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"{RED}[SCRIPT ERROR]{RESET} {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()