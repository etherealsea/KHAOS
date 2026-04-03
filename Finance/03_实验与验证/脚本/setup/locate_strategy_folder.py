import os
import sys
import platform

def find_vnpy_dir():
    # 默认路径通常是 C:\Users\Username\vnpy_run
    home = os.path.expanduser("~")
    vnpy_run_dir = os.path.join(home, "vnpy_run")
    strategies_dir = os.path.join(vnpy_run_dir, "strategies")
    
    if not os.path.exists(vnpy_run_dir):
        print(f"Warning: vnpy_run directory not found at {vnpy_run_dir}")
        print("Have you launched VeighNa Station at least once?")
        # Try to create it if it doesn't exist? Better not, might mess up permissions.
        return None
        
    if not os.path.exists(strategies_dir):
        print(f"Creating strategies directory at {strategies_dir}")
        os.makedirs(strategies_dir)
        
    return strategies_dir

def open_folder(path):
    if platform.system() == "Windows":
        os.startfile(path)
    elif platform.system() == "Darwin":
        os.system(f"open {path}")
    else:
        os.system(f"xdg-open {path}")

if __name__ == "__main__":
    print("Locating VeighNa strategy folder...")
    path = find_vnpy_dir()
    
    if path:
        print(f"Found: {path}")
        print("Opening folder...")
        open_folder(path)
        print("\nPlease copy 'src/vnpy_integration/khaos_strategy_bundled.py' into this folder.")
    else:
        print("Could not locate VeighNa strategy folder.")
