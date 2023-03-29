import pydicom
import SimpleITK as sitk
from pydicom.dataset import FileDataset


def modify_dcm_tags(dcm_path: str, original_ds: FileDataset, options):
    print(options["origin"])
    ds = pydicom.read_file(dcm_path, force=True)
    if "origin" in options:
        ds.ImagePositionPatient = list(options["origin"])
    set_common_dcm_tag(ds, original_ds)
    ds.ImageOrientationPatient = original_ds.get("ImageOrientationPatient", "")
    ds.save_as(dcm_path)

def set_common_dcm_tag(ds: pydicom.FileDataset, original_ds: pydicom.FileDataset):
    """
    更新通用 tag
    """
    ds.SliceThickness = original_ds.get("SliceThickness", "")
    ds.WindowWidth = original_ds.get("WindowWidth", 800)
    ds.WindowCenter = original_ds.get("WindowCenter", 300)
    ds.Modality = original_ds.get("Modality", "")
    ds.SeriesNumber = original_ds.get("SeriesNumber", "")
    ds.StudyDate = original_ds.get("StudyDate", "")
    ds.SeriesDate = original_ds.get("SeriesDate", "")
    ds.StudyTime = original_ds.get("StudyTime", "")
    ds.SeriesTime = original_ds.get("SeriesTime", "")
    ds.PatientID = original_ds.get("PatientID", "")
    ds.PatientName = original_ds.get("PatientName", "")
    ds.PatientAge = original_ds.get("PatientAge", "")
    ds.PatientBirthDate = original_ds.get("PatientBirthDate", "")
    ds.PatientSex = original_ds.get("PatientSex", "")
    ds.AccessionNumber = original_ds.get("AccessionNumber", "")
    ds.StudyInstanceUID = original_ds.get("StudyInstanceUID", "")
    ds.SeriesInstanceUID = original_ds.get("SeriesInstanceUID", "")
    ds.InstitutionName = str(original_ds.get("InstitutionName", ""))
    ds.SOPInstanceUID = "2342355267"