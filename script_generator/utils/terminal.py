import subprocess
import platform
import os


def open_new_terminal(command, conda_env="VRFunAIGen", relative_path_up=2):
    """
    Opens a new terminal window, navigates up a specified number of directory levels,
    activates the given Conda environment, runs the command, and then (on Windows)
    automatically closes when finished.

    :param command: The command to run (string).
    :param conda_env: Name of the Conda environment to activate.
    :param relative_path_up: Number of levels to move up from this script's directory.
    :return: Popen handle (or None on failure)
    """
    system = platform.system()
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * relative_path_up))

    if isinstance(command, list):
        command = " ".join(command)

    if system == "Windows":
        conda_activate = f'C:\\Users\\{os.getenv("USERNAME")}\\miniconda3\\Scripts\\activate.bat'
        if not os.path.exists(conda_activate):
            print(f"Conda not found at {conda_activate}. Please check your installation.")
            return None
        # Use /WAIT so that the call blocks until the spawned terminal (and its command) completes.
        cmd = f'start /WAIT cmd /C "cd /d {base_path} && call \"{conda_activate}\" {conda_env} && {command}"'
        return subprocess.Popen(cmd, shell=True)

    elif system == "Darwin":  # macOS
        # On macOS, using osascript; 'exit' makes the terminal close when done.
        cmd = (f'osascript -e \'tell application "Terminal" to do script '
               f"\"cd {base_path}; source ~/.bash_profile; conda activate {conda_env}; {command}; exit\"\'")
        return subprocess.Popen(cmd, shell=True)

    elif system == "Linux":
        # Try common terminal emulators; note: behavior may vary.
        term_cmds = [
            f'gnome-terminal -- bash -c "cd {base_path}; source ~/.bashrc; conda activate {conda_env}; {command}; exit"',
            f'konsole -e bash -c "cd {base_path}; source ~/.bashrc; conda activate {conda_env}; {command}; exit"',
            f'x-terminal-emulator -e bash -c "cd {base_path}; source ~/.bashrc; conda activate {conda_env}; {command}; exit"'
        ]
        for term_cmd in term_cmds:
            try:
                return subprocess.Popen(term_cmd, shell=True)
            except FileNotFoundError:
                continue
        return None
    else:
        print(f"Unsupported OS: {system}")
        return None