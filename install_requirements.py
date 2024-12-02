import subprocess
import sys
import re
import os

def is_package_installed(package_name, version=None):
    """
    Check if a package is installed using pip.
    If version is specified, also checks the version.
    """
    try:
        # Using pip show to check installed packages and their versions
        result = subprocess.run([sys.executable, "-m", "pip", "show", package_name], capture_output=True, text=True)
        if result.returncode == 0:
            installed_version = None
            # If a version is specified, compare it with the installed version
            if version:
                # Extract the installed version from the pip show output
                for line in result.stdout.splitlines():
                    if line.startswith("Version:"):
                        installed_version = line.split(":", 1)[1].strip()
                        break
                if installed_version != version:
                    return False  # Version mismatch
            return True
        else:
            return False
    except Exception as e:
        print(f"Error checking if {package_name} is installed: {e}")
        return False

def install_package(package):
    """
    Install a package using pip with --user flag.
    """
    try:
        print(f"Installing {package} with --user flag...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", package])
    except subprocess.CalledProcessError:
        print(f"Error installing {package}.")
        sys.exit(1)

def install_requirements(requirements_file):
    """
    Install packages listed in the requirements.txt file only if not already installed.
    """
    with open(requirements_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue  # Skip empty lines and comments

            # Handle Git URLs separately
            if line.startswith('git+'):
                print(f"Skipping package from Git URL: {line}")
                install_package(line)  # Install directly from the URL
                continue  # Skip the normal check for Git URLs

            # Check if package has a version constraint (==)
            package_name, *version = line.split("==")
            version = version[0] if version else None  # Get version if specified

            # Check if the package is already installed
            if is_package_installed(package_name, version):
                print(f"{package_name} is already installed.")
            else:
                print(f"{package_name} is not installed or has a different version.")
                install_package(line)

if __name__ == "__main__":
    # The path to your requirements.txt file
    requirements_file = "requirements.txt"
    
    # Call the function to install the requirements
    install_requirements(requirements_file)
