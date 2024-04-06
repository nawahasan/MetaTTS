import os
import subprocess

def setup_project():
    # Clone the repository
    os.system("git clone https://github.com/jaywalnut310/vits.git")
    os.chdir("vits")

    # Install dependencies
    os.system("pip install -r requirements.txt")

    # Navigate to monotonic_align directory
    os.chdir("monotonic_align")
    os.makedirs("monotonic_align", exist_ok=True)

    # Build the extension
    subprocess.check_output(["python", "setup.py", "build_ext", "--inplace"])

    # Return to the main project directory
    os.chdir("../")

if __name__ == "__main__":
    setup_project()
