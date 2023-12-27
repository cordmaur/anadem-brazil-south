""" Docstring"""
import sys
import shutil
from typing import Union
from pathlib import Path
from pyicloud import PyiCloudService
from pyicloud.services.drive import DriveNode
import click
import unidecode

import geopandas as gpd
import xarray as xr


def refresh_folder(folder_node):
    """ Docstring"""
    folder_node._children = None  # pylint: disable=protected-access

    if 'items' in folder_node.data:
        del folder_node.data['items']


def get_folder_node(api, remote_folder: Union[str, DriveNode]):
    """ Docstring"""
    if isinstance(remote_folder, DriveNode):
        return remote_folder
    else:
        return api.drive[remote_folder]


def upload_file(api, local_file, remote_folder):
    """
    Upload a file to the iCloud drive
    Remote folder can be given as `str` or DriveNode
    """
    folder = get_folder_node(api, remote_folder)

    with open(local_file, 'rb') as file_in:
        folder.upload(file_in)

    refresh_folder(folder)


def download_file(api, remote_folder, file, local_folder):
    """
    Download a file from the iCloud
    """

    folder_node = get_folder_node(api, remote_folder)

    file_node = folder_node[str(file)]
    local_file = Path(local_folder)/file

    with file_node.open(stream=True) as response:
        with open(local_file, "wb") as file_out:
            shutil.copyfileobj(response.raw, file_out)    

    refresh_folder(folder_node)

    return local_file


def delete_file(api, remote_folder, file):
    """Delete a file from the iCloud drive"""
    folder = get_folder_node(api, remote_folder)
    if file in folder.dir():
        file_img = folder[file]
        file_img.delete()

        refresh_folder(api.drive[folder])


def icloud_login():
    """Login to icloud and return a session"""
    api = PyiCloudService('cordmaur@gmail.com', 'Eethath8@')

    if api.requires_2fa:
        print("Two-factor authentication required.")
        code = input("Enter the code you received of one of your approved devices: ")
        result = api.validate_2fa_code(code)
        print("Code validation result: %s" % result)

        if not result:
            print("Failed to verify security code")
            sys.exit(1)

        if not api.is_trusted_session:
            print("Session is not trusted. Requesting trust...")
            result = api.trust_session()
            print("Session trust result %s" % result)

            if not result:
                print("""Failed to request trust. You will likely be prompted 
                      for the code again in the coming weeks""")
    elif api.requires_2sa:
        print("Two-step authentication required. Your trusted devices are:")

        devices = api.trusted_devices
        for i, device in enumerate(devices):
            print(
                "  %s: %s" % (i, device.get('deviceName',
                "SMS to %s" % device.get('phoneNumber')))
            )

        device = click.prompt('Which device would you like to use?', default=0)
        device = devices[device]
        if not api.send_verification_code(device):
            print("Failed to send verification code")
            sys.exit(1)

        code = click.prompt('Please enter validation code')
        if not api.validate_verification_code(device, code):
            print("Failed to verify verification code")
            sys.exit(1)    

    api.drive.dir()
    api._drive.params['clientId'] = api.client_id  # pylint: disable=protected-access

    return api


def create_fname(place):
    """Create the filename for a place. A place is Series with municipio, uf and name"""
    return Path(f"Flood_Report_{place.name}_{unidecode.unidecode(place['municipio'])}_{place['uf']}.pdf")


def get_file(remote_folder, fname, local_folder): 
    """Get the file from the iCloud"""

    local_folder = Path(local_folder)

    if (local_folder/fname).exists():
        local_path = local_folder/fname

    else:
        local_path = download_file(None, remote_folder, fname, local_folder)

    return local_path


def get_aoi(remote_folder, place, local_folder):
    """Get the geojson AOI from the iCloud"""

    fname = create_fname(place).with_suffix('.geojson')
    local_path = get_file(remote_folder, fname, local_folder)
    return gpd.read_file(local_path)


def get_flood(remote_folder, place, local_folder):
    """Get the flood raster from the iCloud"""

    fname = create_fname(place).with_suffix('.netcdf')

    local_path = get_file(remote_folder, fname, local_folder)

    return xr.load_dataset(local_path)

def get_report(remote_folder, place, local_folder):
    """Get the PDF report for the specific location"""

    fname = create_fname(place).with_suffix('.pdf')
    local_path = get_file(remote_folder, fname, local_folder)
    return local_path
