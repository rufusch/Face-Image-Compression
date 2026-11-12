"""Run python main.py, or use Run Python File in VS Code.

Missing dependencies are installed automatically (internet required).
The SVD comparison window opens. Close the window to stop.
"""
from pathlib import Path
import importlib.util
import subprocess
import sys
import socket
from launch import start_application


def loopback_available():
    """Verify an actual local TCP connection, not just a successful bind."""
    try:
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            listener.listen(1)
            with socket.create_connection(listener.getsockname(), timeout=2):
                return True
    except OSError as error:
        print(f'Local browser connection unavailable: {error}', flush=True)
        return False


def main():
    if sys.version_info < (3, 10):
        raise SystemExit('Python 3.10 or newer is required.')
    if any(importlib.util.find_spec(name) is None for name in ('flask', 'numpy', 'scipy', 'PIL', 'sklearn')):
        print('Installing required packages...', flush=True)
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r',
                                   str(Path(__file__).resolve().with_name('requirements.txt'))])
        except subprocess.CalledProcessError:
            raise SystemExit('Installation failed. Check the error above.')
    if '--web' not in sys.argv or not loopback_available():
        print('Opening FaceLab desktop interface.', flush=True)
        from desktop import start_desktop
        start_desktop()
    else:
        start_application()


if __name__ == '__main__':
    main()
