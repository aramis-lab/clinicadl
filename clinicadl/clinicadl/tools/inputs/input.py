# coding: utf8

def fetch_file(url, filename):
    """Function to download a specific file and save it into the ressources
    folder of the package.
    Args:
        url: url where to request is done
        filename: absolute path to the filename where the file is downloaded
    Returns:

    Raises:
    """
    from clinica.utils.exceptions import ClinicaException
    from urllib.request import Request, urlopen
    from urllib.error import URLError
    import shutil
    import ssl
    import os.path
    from clinica.utils.stream import cprint

    head_tail = os.path.split(filename)
    if not os.path.exists(head_tail[0]):
        cprint('Path to the file does not exist')
        cprint('Stop Clinica and handle this error')

    # Download the file from `url` and save it locally under `file_name`:
    cert = ssl.get_server_certificate(("aramislab.paris.inria.fr", 443))
    gcontext = ssl.SSLContext()
    req = Request(url)
    try:
        response = urlopen(req, context=gcontext)
    except URLError as e:
        if hasattr(e, 'reason'):
            cprint('We failed to reach a server.')
            cprint(['Reason: ' + e.reason])
        elif hasattr(e, 'code'):
            cprint('The server could not fulfill the request.')
            cprint(['Error code: ' + e.code])
    else:
        try:
            with open(filename, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
        except OSError as err:
            cprint("OS error: {0}".format(err))
