"""Start the local app and open a browser after binding successfully."""
import webbrowser
from threading import Timer

def start_application():
    from werkzeug.serving import make_server
    from app import app
    server = None
    for port in (5000, 8765, 8766):
        try:
            server = make_server('127.0.0.1', port, app, threaded=True)
            break
        except (OSError, SystemExit):
            continue
    if server is None:
        raise SystemExit('Could not start the local server. Please check the error messages above.')
    url = f'http://127.0.0.1:{server.server_port}/'
    print(f'FaceLab is available at {url}', flush=True)
    print('Keep this window open while using the app. Press Ctrl+C to stop.', flush=True)
    # Serve immediately; a slow browser launch must not prevent HTTP responses.
    browser = Timer(0.5, webbrowser.open, args=(url,))
    browser.daemon = True
    browser.start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        browser.cancel()
        server.server_close()


if __name__ == '__main__':
    from main import main
    main()
