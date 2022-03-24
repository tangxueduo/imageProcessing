import pydicom
import nibabel as nib
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d  %H:%M:%S %a",
)

file = "/media/tx-deepocean/Data/DICOMS/ct_cerebral/CN010002-13696724/1.2.840.113619.2.416.10634142502611409964348085056782520111/1.2.840.113619.2.416.77348009424380358976506205963520437809/1.2.840.113619.2.416.2106685279137426449336212617073446717.21"
nii_file = "/media/tx-deepocean/Data/DICOMS/RESULT/volume/1.2.840.113619.2.416.77348009424380358976506205963520437809.nii.gz"
ds = pydicom.read_file(file)
WindowWidth = ds.get("WindowWidth", 800)
logging.info(f'WindowWidth: {WindowWidth}')

img = nib.load(nii_file)
