# coding: utf8
import hashlib
from collections import namedtuple

RemoteFileStructure = namedtuple(
        'RemoteFileStructure',
        ['filename', 'url', 'checksum']
        )


def _sha256(path):
    """Calculate the sha256 hash of the file at path."""
    sha256hash = hashlib.sha256()
    chunk_size = 8192
    with open(path, "rb") as f:
        while True:
            buffer = f.read(chunk_size)
            if not buffer:
                break
            sha256hash.update(buffer)
    return sha256hash.hexdigest()


def fetch_file(remote, dirname=None):
    """Function to download a specific file and save it into the ressources
    folder of the package.
    Args:
        remote: satructure containing url, filename and checksum
        dirname: absolute path where the file will be downloaded
    Returns:
        file_path: absolute file path
    Raises:
    """
    from clinica.utils.exceptions import ClinicaException
    from urllib.request import Request, urlopen
    from urllib.error import URLError
    import shutil
    import ssl
    import os.path
    from clinica.utils.stream import cprint

    if not os.path.exists(dirname):
        cprint('Path to the file does not exist')
        cprint('Stop Clinica and handle this error')

    file_path = os.path.join(dirname, remote.filename)
    # Download the file from `url` and save it locally under `file_name`:
    gcontext = ssl.SSLContext()
    req = Request(remote.url + remote.filename)
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
            with open(file_path, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
        except OSError as err:
            cprint("OS error: {0}".format(err))

    checksum = _sha256(file_path)
    if remote.checksum != checksum:
        raise IOError("{} has an SHA256 checksum ({}) "
                      "differing from expected ({}), "
                      "file may be corrupted.".format(file_path, checksum,
                                                      remote.checksum)
                      )
    return file_path
