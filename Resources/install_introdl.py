import os
import subprocess
import sys

def ensure_introdl_installed(force_update=False, local_path_pkg='../../resources/introdl'):
    """Ensure the `introdl` module is installed, optionally forcing an update."""
    def uninstall_introdl():
        """Uninstall the `introdl` module."""
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "introdl"])
            print("The `introdl` module was uninstalled successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to uninstall `introdl`: {e}")
            sys.exit(1)

    # Check if `introdl` is already installed
    try:
        import introdl
        if force_update:
            print("Force update requested. Uninstalling `introdl`...")
            uninstall_introdl()
        else:
            print("The `introdl` module is already installed.")
            return
    except ImportError:
        if not force_update:
            print("The `introdl` module is not installed. Proceeding with installation...")

    # Determine if running in CoCalc
    is_cocalc = "COCALC" in os.environ or "COCALC_PROJECT_ID" in os.environ
    user_flag = "--user" if is_cocalc else ""

    # Check for the local package path
    if os.path.isdir(local_path_pkg):
        # Install from the specified local directory
        print(f"Installing `introdl` from local directory: {local_path_pkg}")
        command = [sys.executable, "-m", "pip", "install", user_flag, "-q", local_path_pkg]
    else:
        # Install from GitHub
        print(f"Local directory not found at {local_path_pkg}. Installing `introdl` from GitHub...")
        github_url = "git+https://github.com/DataScienceUWL/DS776.git#subdirectory=resources/introdl"
        command = [sys.executable, "-m", "pip", "install", user_flag, "-q", github_url]

    # Execute the installation command
    try:
        subprocess.check_call([arg for arg in command if arg])  # Remove empty flags
        print("Installation completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install `introdl`: {e}")
        sys.exit(1)

    # Verify the installation
    try:
        import introdl
        print("The `introdl` module is now installed.")
    except ImportError:
        print("Installation failed. `introdl` is still not available.")
        sys.exit(1)


