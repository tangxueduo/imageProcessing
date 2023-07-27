import math
import time
from copy import deepcopy

import numpy as np
import pydicom
import SimpleITK as sitk
from PIL import Image, ImageDraw, ImageFont
from pydicom.dataset import FileDataset
from pydicom.uid import ExplicitVRLittleEndian, ImplicitVRLittleEndian

FONT = ImageFont.truetype("statics/SourceHanSansCN-Normal.ttf", 18)


def modify_dcm_tags(dcm_path: str, original_ds: FileDataset, options):
    print(options["origin"])
    ds = pydicom.read_file(dcm_path, force=True)  # type:ignore
    if "origin" in options:
        ds.ImagePositionPatient = list(options["origin"])
    set_common_dcm_tag(ds, original_ds)
    ds.ImageOrientationPatient = original_ds.get("ImageOrientationPatient", "")
    ds.save_as(dcm_path)


def remove_extra_tags(ds: FileDataset):
    """remove unused tags"""
    ds[0x7FE0, 0x10].VR = "OB"
    for tag in (
        (0x0040, 0x0275),  # Request Attributes Sequence
        (0x0008, 0x1120),  # Referenced Patient Sequence
        (0x0008, 0x1110),  # Referenced Study Sequence
        (0x0008, 0x1115),  # Referenced Series Sequence
        (0x0008, 0x1140),  # Referenced Image Sequence
        (0x0088, 0x2112),  # Source Image Sequence
        (0x0088, 0x0200),  # Icon Image Sequence
        (0x0008, 0x0012),  # [DA] InstanceCreationDate:
        (0x0008, 0x0013),  # [TM] InstanceCreationTime:
        (0x0008, 0x0022),  # [DA] AcquisitionDate:
        (0x0008, 0x0032),  # [TM] AcquisitionTime:
        (0x0010, 0x4000),  # [LT] PatientComments:
        (0x0018, 0x0022),  # [CS] ScanOptions:
        (0x0018, 0x0050),  # [DS] SliceThickness:
        (0x0018, 0x0060),  # [DS] KVP:
        (0x0018, 0x0090),  # [DS] DataCollectionDiameter:
        (0x0018, 0x1000),  # [LO] DeviceSerialNumber:
        (0x0018, 0x1020),  # [LO] SoftwareVersions:
        (0x0018, 0x1030),  # [LO] ProtocolName:
        (0x0018, 0x1100),  # [DS] ReconstructionDiameter:
        (0x0018, 0x1120),  # [DS] GantryDetectorTilt:
        (0x0018, 0x1130),  # [DS] TableHeight:
        (0x0018, 0x1140),  # [CS] RotationDirection:
        (0x0018, 0x1150),  # [IS] ExposureTime:
        (0x0018, 0x1151),  # [IS] XRayTubeCurrent:
        (0x0018, 0x1152),  # [IS] Exposure:
        (0x0018, 0x1160),  # [SH] FilterType:
        (0x0018, 0x1170),  # [IS] GeneratorPower:
        (0x0018, 0x1190),  # [DS] FocalSpots:
        (0x0018, 0x1210),  # [SH] ConvolutionKernel:
        (0x0018, 0x5100),  # [CS] PatientPosition:
        (0x0018, 0x9302),  # [CS] AcquisitionType:
        (0x0018, 0x9305),  # [FD] RevolutionTime:
        (0x0018, 0x9306),  # [FD] SingleCollimationWidth:
        (0x0018, 0x9307),  # [FD] TotalCollimationWidth:
        (0x0018, 0x9318),  # [FD] ReconstructionTargetCenterPatient:
        (0x0018, 0x9327),  # [FD] TablePosition:
        (0x0018, 0x9334),  # [CS] FluoroscopyFlag:
        (0x0018, 0x9345),  # [FD] CTDIvol:
        (0x0020, 0x0012),  # [IS] AcquisitionNumber:
        (0x0020, 0x0032),  # [DS] ImagePositionPatient:
        (0x0020, 0x0052),  # [UI] FrameOfReferenceUID:
        (0x0020, 0x1040),  # [LO] PositionReferenceIndicator:
        (0x0020, 0x1041),  # [DS] SliceLocation:
        (0x0040, 0x0002),  # [DA] ScheduledProcedureStepStartDate:
        (0x0040, 0x0003),  # [TM] ScheduledProcedureStepStartTime:
        (0x0040, 0x0004),  # [DA] ScheduledProcedureStepEndDate:
        (0x0040, 0x0005),  # [TM] ScheduledProcedureStepEndTime:
        (0x0040, 0x0244),  # [DA] PerformedProcedureStepStartDate:
        (0x0040, 0x0245),  # [TM] PerformedProcedureStepStartTime:
        (0x0040, 0x0253),  # [SH] PerformedProcedureStepID:
        (0x0070, 0x0080),  # [CS] ContentLabel:
        (0x0070, 0x0081),  # [LO] ContentDescription:
        (0x0070, 0x0082),  # [DA] PresentationCreationDate:
        (0x0070, 0x0083),  # [TM] PresentationCreationTime:
        (0x0070, 0x0084),  # [PN] ContentCreatorName:
        (0x7005, 0x0010),  # [LO] PrivateCreatorID:
        (0x7005, 0x1007),  # [UN] PrivateTag: binary data
        (0x7005, 0x1008),  # [UN] PrivateTag: binary data
        (0x7005, 0x100B),  # [UN] PrivateTag: binary data
        (0x7005, 0x100D),  # [UN] PrivateTag: binary data
        (0x7005, 0x100E),  # [UN] PrivateTag: binary data
        (0x7005, 0x100F),  # [UN] PrivateTag: binary data
        (0x7005, 0x1012),  # [UN] PrivateTag: binary data
        (0x7005, 0x1013),  # [UN] PrivateTag: binary data
        (0x7005, 0x1016),  # [UN] PrivateTag: binary data
        (0x7005, 0x1017),  # [UN] PrivateTag: binary data
        (0x7005, 0x1018),  # [UN] PrivateTag: binary data
        (0x7005, 0x1019),  # [UN] PrivateTag: binary data
        (0x7005, 0x101A),  # [UN] PrivateTag: binary data
        (0x7005, 0x101B),  # [UN] PrivateTag: binary data
        (0x7005, 0x101D),  # [UN] PrivateTag: binary data
        (0x7005, 0x101E),  # [UN] PrivateTag: binary data
        (0x7005, 0x101F),  # [UN] PrivateTag: binary data
        (0x7005, 0x1020),  # [UN] PrivateTag: binary data
        (0x7005, 0x1024),  # [UN] PrivateTag: binary data
        (0x7005, 0x1030),  # [UN] PrivateTag: binary data
        (0x7005, 0x1040),  # [UN] PrivateTag: binary data
        (0x7005, 0x1041),  # [UN] PrivateTag: binary data
        (0x7005, 0x1043),  # [UN] PrivateTag: binary data
        (0x7005, 0x1063),  # [UN] PrivateTag: binary data
    ):
        try:
            ds.pop(tag)
        except KeyError:
            continue


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


def array_to_dicom(
    origin_ds: pydicom.dataset.FileDataset,
    np_array: np.ndarray,
    filename: str,
    is_vr: bool,
    instance_number: int,
    full_tag: bool = False,
):
    """将一个 numpy array 转换为一张 dicom 并存储

    Args:
        origin_ds: 原始 dicom 摸一张图的 dataset
        np_array: 待转换的 numpy array
        filename: 存储路径
        is_vr: TODO update desc
        instance_number: new dicom instance number
        full_tag: 是否保留所有原始图像的 tag
    """
    creation_date = time.strftime("%Y%m%d", time.localtime(time.time()))
    creation_time = time.strftime("%H%M%S", time.localtime(time.time()))
    ds = origin_ds.copy()
    # edit tags
    ds.file_meta.TransferSyntaxUID = (
        ImplicitVRLittleEndian if origin_ds.is_implicit_VR else ExplicitVRLittleEndian
    )
    ds.Rows, ds.Columns = np_array.shape[:2]
    ds.InstanceNumber = instance_number
    ds.ImageComments = origin_ds.SeriesDescription
    ds.SeriesInstanceUID = gen_suid(origin_ds.SeriesInstanceUID)
    ds.InstanceCreationDate = creation_date
    ds.InstanceCreationTime = creation_time
    ds.ContentDate = creation_date
    ds.ContentTime = creation_time
    ds.PresentationCreationDate = creation_date
    ds.PresentationCreationTime = creation_time
    ds.RescaleIntercept = 0
    ds.RescaleSlope = 1
    ds["PixelData"].is_undefined_length = False
    ds.SOPInstanceUID = gen_uuid()
    ds.file_meta.MediaStorageSOPInstanceUID = origin_ds.SOPInstanceUID
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    ds.ContentLabel = "UNNAMED"
    ds.ContentDescription = "UNNAMED"
    ds.ContentCreatorName = "Infervision"
    if not full_tag:
        remove_extra_tags(ds)  # type:ignore
    if is_vr:
        ds.WindowWidth = 255
        ds.WindowCenter = 127
        ds.BitsStored = 8
        ds.BitsAllocated = 8
        ds.HighBit = 7
        ds.PixelRepresentation = 0
        ds.PhotometricInterpretation = "RGB"
        ds.SamplesPerPixel = 3
        ds.PlanarConfiguration = 0
    else:
        ds.WindowWidth = 800
        ds.WindowCenter = 300
        ds.BitsStored = 16
        ds.BitsAllocated = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
    ds.PixelData = np_array.tobytes()
    ds.save_as(filename)


def gen_suid(uuid: str) -> str:
    """generate SeriesInstanceUid with Infervision Dicom Prefix and origin SeriesInstanceUid"""
    # seg:0~3 Organization Root seg:4 Application ID seg:5 Application Version
    IDP = "1.2.826.0.1.3680043.10"  # Infervision Dicom Prefix
    IDPS = IDP.split(".")
    index = len(IDPS)
    if 1 < len(uuid.split(".")) <= len(IDPS) + 1:
        index = -2
    uid = ".".join(IDPS + uuid.split(".")[index:])
    return uid[:64]


def gen_uuid(uuid: str = "") -> str:
    """generate random StudyInstanceUid with Infervision Dicom Prefix"""
    # seg:0~3 Organization Root seg:4 Application ID seg:5 Application Version
    IDP = "1.2.826.0.1.3680043.10"  # Infervision Dicom Prefix
    if not uuid.startswith(IDP):
        uuid = pydicom.uid.generate_uid()
    IDPS = IDP.split(".")
    uid = ".".join(IDPS + uuid.split(".")[len(IDPS) :])
    return uid[:64]


def draw_text(np_array: np.ndarray, text: str, ds: FileDataset) -> np.ndarray:
    final_height, final_width = np_array.shape
    im_frame = Image.new("L", (final_width, final_height))
    draw = ImageDraw.Draw(im_frame)
    draw.text(
        ((final_width - _get_width(text)) // 2, final_height - 18),
        text,
        fill="#ffffff",
        font=FONT,
    )
    im_array = np.array(im_frame)
    ww, wc = 800, 300
    slope, inter = 1, 0
    plus_array = deepcopy(im_array)
    plus_array[plus_array > 0] = 2
    plus_array[plus_array == 0] = 1
    plus_array[plus_array > 1] = 0
    new_array = np_array * plus_array
    temp = wc - ww / 2
    im_array = im_array / 255.0 * ww + temp
    im_array[im_array == temp] = inter
    im_array = (im_array - inter) / slope
    add_array = np.array(im_array, np.int16)
    new_array += add_array
    return new_array


def _get_width(text: str) -> int:
    draw = ImageDraw.Draw(Image.new(mode="L", size=(10, 10)))
    text_width, _ = draw.textsize(text, font=FONT)
    return text_width


def resize_hu(np_array: np.ndarray, final_width=512, final_height=512):
    """Resize np array to (final_width, final_height)"""
    lh, lw = np_array.shape
    if lw != final_width or lh != final_height:
        new_width = _calculate_width(lw, lh)
        image = sitk.GetImageFromArray(np_array)
        original_spacing = image.GetSpacing()
        new_spacing = [(lw - 1) * original_spacing[0] / (new_width - 1)] * 2
        new_height = int((lh - 1) * original_spacing[1] / new_spacing[1])
        new_size = [new_width, new_height]
        image = sitk.Resample(
            image,
            new_size,
            sitk.Transform(),
            sitk.sitkLinear,
            image.GetOrigin(),
            new_spacing,
            image.GetDirection(),
            0,
            sitk.sitkInt16,
        )
        new_image = sitk.Image([final_width, final_height], sitk.sitkInt16)
        new_image = sitk.RescaleIntensity(new_image, -2000, -2000)
        x, y = (final_width - new_width) // 2, (final_height - new_height) // 2
        new_array = sitk.GetArrayFromImage(new_image)
        image_array = sitk.GetArrayFromImage(image)
        new_array[y : y + new_height, x : x + new_width] = image_array
        return new_array
    print(f"np_array : {np_array.dtype}")
    return np_array


def _calculate_width(width: int, height: int):
    aspect = width / height
    x, y = 512, 477

    def round_aspect(number, key):
        return max(min(math.floor(number), math.ceil(number), key=key), 1)

    if x / y >= aspect:
        x = round_aspect(y * aspect, key=lambda n: abs(aspect - n / y))
    return x


from typing import Dict, Sequence, Union

import numpy as np
import SimpleITK as sitk

ENUM_VOLUME_TYPE: Sequence[str] = ["image", "mask", "other"]
Numeric = Union[int, float]
NumericArrayType = Union[np.ndarray, Sequence[Numeric]]
CenterlineDictType = Dict[str, Union[float, NumericArrayType]]
CenterlineType = Union["Centerline", CenterlineDictType]  # noqa
SitkTransformType = Union[
    sitk.Transform,
    sitk.AffineTransform,
    sitk.Euler3DTransform,
    sitk.DisplacementFieldTransform,
]
