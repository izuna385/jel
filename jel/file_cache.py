"""
Utilities for working with the local dataset cache.
"""
import os
from pathlib import Path
from google_drive_downloader import GoogleDriveDownloader
from jel.common_config import CACHE_ROOT, RESOURCES_GOOGLE_DRIVE_ID
import logging
logger = logging.getLogger(__name__)

def resource_downloader():
    try:
        logger.info('Downloading 4GB model resources...')
        GoogleDriveDownloader.download_file_from_google_drive(file_id=RESOURCES_GOOGLE_DRIVE_ID,
                                            dest_path=str(CACHE_ROOT)+'/resources.zip',
                                            unzip=True)
    except:
        logger.info('shutil download cache because downloading is stopped.')
        os.remove(str(CACHE_ROOT)+'/resources.zip')