import os

def ensure_introdl_installed(force_update=False, local_path_pkg='~/Lessons/Course_Tools/introdl'):
    """Ensure the `introdl` module is installed, using the same environment as Jupyter."""
    local_path_pkg = os.path.expanduser(local_path_pkg)

    # Check if `introdl` is already installed
    try:
        import introdl
        if not force_update:
            print("The `introdl` module is already installed.")
            return
        print("Force update requested. Uninstalling `introdl`...")
        os.system("pip uninstall -y introdl")
    except ImportError:
        print("The `introdl` module is not installed. Proceeding with installation...")

    # Install the package
    if os.path.isdir(local_path_pkg):
        print(f"Installing `introdl` from local directory: {local_path_pkg}")
        os.system(f"pip install {local_path_pkg}")
    else:
        print(f"Local directory not found at {local_path_pkg}. Installing from GitHub...")
        github_url = "git+https://github.com/DataScienceUWL/DS776.git#subdirectory=resources/introdl"
        os.system(f"pip install {github_url}")

    # Verify installation
    try:
        import introdl
        print("The `introdl` module is now installed.")
    except ImportError:
        print("Try restarting the kernel and running this cell again to see if installation was successful.")
        raise
