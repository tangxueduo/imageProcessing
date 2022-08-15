import logging
import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pydicom
from PIL import Image, ImageDraw, ImageFont
from pydicom.dataset import FileDataset

t0 = time.time()


def main(sex, age, score):
    # marker_point = (age, score)
    ds = pydicom.read_file(
        "/media/tx-deepocean/Data/DICOMS/ct_heart/1.3.12.2.1107.5.1.4.74241.30000021112900504205800133754/1.3.12.2.1107.5.1.4.74241.30000021112900504205800133757"
    )
    vr_width = 512
    sex_data_dict = {
        "F": {
            "25%": [0, 0, 0, 0, 0, 0, 0],
            "50%": [0, 0, 0, 0, 0, 4, 24],
            "75%": [0, 0, 0, 10, 33, 87, 123],
            "90%": [4, 9, 23, 66, 140, 310, 362],
        },
        "M": {
            "25%": [0, 0, 0, 0, 3, 14, 28],
            "50%": [0, 0, 3, 16, 41, 118, 151],
            "75%": [2, 11, 44, 101, 187, 434, 569],
            "90%": [21, 64, 176, 320, 502, 804, 1178],
        },
    }
    fig = plt.figure(dpi=300, facecolor="#171D2E")
    fig.set_figwidth(1.71)
    fig.set_figheight(1.71)

    rect1 = [0, 0.83, 1, 0.08]  # [左, 下, 宽, 高] 规定的矩形区域 （全部是0~1之间的数，表示比例）
    rect2 = [0.15, 0.2, 0.7, 0.6]

    # 在fig中添加子图ax，并赋值位置rect
    ax1 = plt.axes(rect1)
    ax1.patch.set_facecolor("#262E48")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["bottom"].set_visible(False)
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax1.text(0.02, 0.5, "冠脉风险评估", fontsize=5, va="center", ha="left", color="#FFFFFF")
    ax1.tick_params(
        axis="both",
        pad=0,
        which="major",
        top=False,
        right=False,
        bottom=False,
        left=False,
        width=ax1.get_window_extent().width,
    )
    ax1.spines["left"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2 = plt.axes(rect2)
    ax2.patch.set_facecolor("#171D2E")
    x_range = [
        math.floor((35 + 39) / 2),
        math.floor((40 + 44) / 2),
        math.floor((45 + 49) / 2),
        math.floor((50 + 54) / 2),
        math.floor((55 + 59) / 2),
        math.floor((60 + 64) / 2),
        math.floor((65 + 70) / 2),
    ]
    ax2.xaxis.set_ticks(np.arange(37, 70, 10))
    # ax2.set_ylim(bottom=0, top=1800, auto=True)
    ax2.set_xlim(37, 70)
    ax2.set_ylim(0, (1500 if score <= 1500 else score) + 80)
    # 将左边框和下边框的颜色设为黑色
    ax2.spines["left"].set_color("#000000")
    ax2.spines["bottom"].set_color("#000000")
    # 去掉边框
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.vlines([47, 57, 67], 0, score, colors="#000000", alpha=0.05, linewidth=0.5)
    ax2.spines["bottom"].set_linewidth(0.5)
    ax2.spines["left"].set_linewidth(0.5)

    # 修改轴线刻度值的颜色和大小
    plt.tick_params(
        axis="both",
        pad=-2,
        which="major",
        labelsize=5,
        colors="#8A9EC3",
        bottom=False,
        left=False,
        width=ax2.get_window_extent().width,
    )

    # 修改轴线刻度值的颜色和大小
    plt.tick_params(
        axis="both",
        which="major",
        labelsize=5,
        colors="#8A9EC3",
        bottom=False,
        left=False,
        pad=-2,
        width=ax2.get_window_extent().width,
    )

    for k, v in sex_data_dict.items():
        if k == sex:
            for percent, yaxis in v.items():
                if percent == "25%":
                    line_color = "#0164FE"
                    y1_data = yaxis
                    ax2.plot(x_range, y1_data, line_color, linewidth=1, label="25%")
                elif percent == "50%":
                    line_color = "#2AC7F6"
                    y2_data = yaxis
                    ax2.plot(x_range, y2_data, line_color, linewidth=1, label="50%")
                elif percent == "75%":
                    line_color = "#F9A727"
                    y3_data = yaxis
                    ax2.plot(x_range, y3_data, line_color, linewidth=1, label="75%")
                else:
                    y4_data = yaxis
                    line_color = "#FF5959"
                    ax2.plot(x_range, y4_data, line_color, linewidth=1, label="90%")

    plt.legend(
        bbox_to_anchor=(0.05, 0.9),
        loc="upper left",
        borderaxespad=0,
        prop={"size": 4},
        labelcolor="#FFFFFF",
        facecolor="#171D2E",
        edgecolor="#171D2E",
    )

    ax2.plot(age, score, color="#FFFFFF", marker="+", label="该患者所在位置", markersize=3)
    ax2.set_xlabel(
        "年龄",
        labelpad=1,
        loc="right",
        fontdict=dict(fontsize=5, color="#8A9EC3", weight="bold"),
    )
    ax2.set_ylabel(
        "分数",
        labelpad=1,
        loc="top",
        fontdict=dict(fontsize=5, color="#8A9EC3", weight="bold"),
    )

    plt.figtext(
        0.22,
        0.76,
        "+",
        horizontalalignment="center",
        va="center",
        size=4,
        weight="light",
        color="#FFFFFF",
    )
    plt.figtext(
        0.27,
        0.76,
        "该患者所在位置",
        horizontalalignment="left",
        va="center",
        size=4,
        weight="light",
        color="#FFFFFF",
    )
    # TODO: Extract the plot as an array
    plt_array = canvas2rgb_array(fig.canvas)
    plt_array = plt_array.astype(dtype="uint8")
    # plt_shape = plt_array.shape

    im_frame = Image.fromarray(plt_array)
    save_path = "/media/tx-deepocean/Data/DICOMS/demos/000CACS1.dcm"

    file_name = os.path.basename(save_path).replace(".dcm", "")
    im_frame = draw_tag(im_frame, file_name[3:], vr_width)
    if im_frame.width != vr_width:
        im_frame = im_frame.resize((vr_width, vr_width), Image.Resampling.LANCZOS)
    np_frame = np.array(im_frame.getdata(), dtype=np.uint8)[:, :3]
    ds.Rows = im_frame.height
    ds.Columns = im_frame.width
    ds.PixelData = np_frame.tobytes()
    ds.InstanceNumber = instance_number(file_name, 2)
    full_tag = False
    edit_tags(ds, "seriesImages" in save_path, full_tag=full_tag, is_vr=True)
    ds.save_as(save_path)
    print((time.time() - t0) * 1000)


def draw_tag(im_frame: Image, tag: str, vr_width: int) -> Image:
    if im_frame.width != 1024 or im_frame.height != 1024:
        aspect = im_frame.width / im_frame.height
        x, y = get_new_size(512, 482, aspect)
        im_frame = im_frame.resize((x, y), Image.Resampling.LANCZOS)
        image1 = Image.new("RGB", (len(tag) * 15 if x < len(tag) * 15 else x, y + 30))
        bw, bh = image1.size
        lw, lh = im_frame.size
        image1.paste(im_frame, ((bw - lw) // 2, 0))
        im_frame = image1
    else:
        im_frame = im_frame.resize((vr_width, vr_width), Image.Resampling.LANCZOS)
    font = ImageFont.truetype("SourceHanSansCN-Normal.ttf", 18)
    draw = ImageDraw.Draw(im_frame)
    text_width, _ = draw.textsize(tag, font=font)
    draw.text(
        ((im_frame.width - text_width) // 2, im_frame.height - 28),
        tag,
        (255, 255, 255),
        font=font,
    )
    return im_frame


def instance_number(file_name: str, index: int) -> int:
    try:
        return int(file_name[:3])
    except Exception:
        logging.debug(f"invalid file_name for {file_name}")
    return index


def round_aspect(number, key):
    return max(min(math.floor(number), math.ceil(number), key=key), 1)


def get_new_size(x, y, aspect):
    if x / y >= aspect:
        x = round_aspect(y * aspect, key=lambda n: abs(aspect - n / y))
    else:
        y = round_aspect(x / aspect, key=lambda n: 0 if n == 0 else abs(aspect - x / n))
    return x, y


def canvas2rgb_array(canvas):
    """Adapted from: https://stackoverflow.com/a/21940031/959926"""
    canvas.draw()
    buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    ncols, nrows = canvas.get_width_height()
    scale = round(math.sqrt(buf.size / 3 / nrows / ncols))
    # return buf.reshape(scale * nrows, scale * ncols, 3)
    return buf.reshape(scale * nrows, scale * ncols, 3)


def edit_tags(
    ds: FileDataset, series: bool = True, full_tag: bool = False, is_vr: bool = False
):
    """edit tags according to our demand"""
    creation_date = time.strftime("%Y%m%d", time.localtime(time.time()))
    creation_time = time.strftime("%H%M%S", time.localtime(time.time()))
    ds.ImageComments = ds.SeriesDescription
    ds.SeriesInstanceUID = gen_suid(ds.SeriesInstanceUID)
    ds.InstanceCreationDate = creation_date
    ds.InstanceCreationTime = creation_time
    ds.ContentDate = creation_date
    ds.ContentTime = creation_time
    ds.PresentationCreationDate = creation_date
    ds.PresentationCreationTime = creation_time
    ds.RescaleIntercept = 0
    ds.RescaleSlope = 1
    ds.SOPInstanceUID = gen_uuid()
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    ds.SeriesDescription = "TX Push Series" if series else "TX Print Series"
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    if not ds.SeriesNumber:
        ds.SeriesNumber = "991"
    elif not str(ds.SeriesNumber).startswith("99"):
        ds.SeriesNumber = "99" + str(ds.SeriesNumber)
    ds.ContentLabel = "UNNAMED"
    ds.ContentDescription = "UNNAMED"
    ds.ContentCreatorName = "Infervision"
    ds["PixelData"].is_undefined_length = False
    # ds.SpecificCharacterSet = "ISO_IR 192"
    if not full_tag:
        remove_extra_tags(ds)
    if is_vr:
        ds.BitsStored = 8
        ds.BitsAllocated = 8
        ds.HighBit = 7
        ds.PixelRepresentation = 0
        ds.WindowWidth = 255
        ds.WindowCenter = 127
        ds.PhotometricInterpretation = "RGB"
        ds.SamplesPerPixel = 3
        ds.PlanarConfiguration = 0
    else:
        ds.BitsStored = 16
        ds.BitsAllocated = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        ds.WindowWidth = 800
        ds.WindowCenter = 300


def gen_suid(uuid: str) -> str:
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


if __name__ == "__main__":
    main("M", 38, 50)
