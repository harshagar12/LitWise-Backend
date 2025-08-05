"""
Setup script to install required Python packages for the recommendation engine
"""

import subprocess
import sys

def install_packages():
    """Install required Python packages"""
    packages = [
        'pandas',
        'numpy', 
        'scikit-learn',
        'fastapi',
        'uvicorn'
    ]
    
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {package}: {e}")
            return False
    
    print("\n✓ All packages installed successfully!")
    return True

if __name__ == "__main__":
    install_packages()
