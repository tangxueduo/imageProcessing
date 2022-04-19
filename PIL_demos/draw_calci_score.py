import json
import logging
import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import requests
import SimpleITK as sitk
from PIL import Image
from pydicom.dataset import FileDataset


def main():
    """
    param: ip:port/series/series_iuid/predict/ct_heart_cac_score
    param: origin_dicom_file_path
    param: save_png_path
    param: save_dicom_path
    param: 
    """
    url = "http://172.16.6.6:3333/series/1.2.392.200036.9116.2.1796265542.1617755474.4.1362300001.1/predict/ct_heart_cac_score"
    res = requests.get(url).json()
    # TODO: dicom取中间那张
    dcm_file = "/media/tx-deepocean/Data/DICOMS/ct_heart/CN010002-03168797/1.2.392.200036.9116.2.6.1.44063.1796265542.1618201762.186046/1.2.392.200036.9116.2.1796265542.1618202018.4.1000300001.1/1.2.392.200036.9116.2.1796265542.1618202018.770035.1.398"
    save_path = "/media/tx-deepocean/Data/DICOMS/demos/Calcification_Score.dcm"
    ds = pydicom.read_file(dcm_file, force=True)
    # 画折线图 保存为png
    vessel_map = {"vessel1": "LM", "vessel2": "LAD", "vessel5": "LCX", "vessel9": "RCA"}
    cparameter = 0.777
    sliceThickness = 1.0
    vr_width = 512
    score = res["summary"]["score"]
    # 画钙化积分表
    if (sliceThickness < 1.5):
        alpha = sliceThickness / 1.5
    elif (sliceThickness >= 3.5):
        alpha = sliceThickness / 3.5
    else:
        alpha = 1
    
    #列名
    col=["冠状动脉", "容积\n(mm3)", "等效质量\n(mg CaHA)", "钙化积分"]
    #行名
    row=["LM", "LAD", "LCX", "RCA", "总计"]
    x, y, z = res["summary"]["spacing"]
    tabel_list = []
    ctaVolumePixel, equivalentMassHU, calcificationScore = 0, 0, 0
    for k, v in res["lesions"].items():
        for index, val in enumerate(v["contour"]["data"]):
            ctaVolumePixel += val["pixelArea"]
            if (val["avgHU"] != -9999):
                equivalentMassHU += val["pixelArea"] * val["avgHU"]
            for i in range(4):
                calcificationScore += val["agatstonPixelArea"][i] * score[i]
        # 计算容积
        ctaVolume = round(ctaVolumePixel * x * y * z, 2)
        # 计算等效质量
        equivalentMass = round((equivalentMassHU * cparameter * x * y * z) / 1000, 2)
        # 计算钙化积分
        calcificationScore = round(calcificationScore * alpha, 2)
        tabel_list.append([k, ctaVolume, equivalentMass, calcificationScore])  
    
    total = [0, 0, 0, 0]
    for i in range(len(tabel_list)):
        total[1] += tabel_list[i][1]
        total[2] += tabel_list[i][2]
        total[3] += tabel_list[i][3]
    total[0] = "总计"
    total[1] = round(total[1], 2)
    total[2] = round(total[2], 2)
    total[3] = round(total[3], 2)
    tabel_list.append(total)

    plt.rcParams["font.sans-serif"]=["SimHei"] #用来正常显示中文标签
    plt.rcParams["axes.unicode_minus"]=False #用来正常显示负号
    # px， py = width * dpi, height  * dpi
    fig = plt.figure(dpi=300, facecolor="#000000")
    fig.set_figwidth(2)
    fig.set_figheight(2)

    ax= fig.add_axes([0.1,0.1,0.8,0.8])
    ax.patch.set_facecolor("#171D2E")
    
    # 单元格背景置为黑色
    colors = [["#171D2E"] * 4 for _ in range(5)]
    
    tab = ax.table(cellText=tabel_list, 
                colLabels=col, 
                loc='center', 
                cellLoc='right',
                colWidths=[0.25] * 5,
                rowLoc='right',
                edges='closed',
                colColours=["#171D2E"] * 4,
                cellColours=colors
                )
    tab.auto_set_font_size(False)
    for row in range(len(tabel_list)):
        for col in range(len(tabel_list[row])):
            tab[row + 1, col].get_text().set_color("#FFFFFF")
            tab[row + 1, col].get_text().set_fontsize(5)
            tab[0, col].get_text().set_color('#8A9EC3')
            tab[0, col].get_text().set_fontsize(4)
        tab[row, 0].get_text().set_color('#8A9EC3')
    tab[5, 0].get_text().set_color('#8A9EC3')
    tab.scale(1, 1)
    
    annotation_text = "注释: 钙化积分基于Agatston分数计算\n    默认钙化的阈值: 130HU\n    等质量矫正因子: {cparameter}".format(cparameter=cparameter)
    # Add annotation
    plt.figtext(0.15, 0.15,
                annotation_text,
                horizontalalignment='left',
                size=4, weight='light',
                color="#8A9EC3",
                multialignment="left",
            )
    ax.set_title("钙化评分表", x=0.1, y=0.8, fontsize=5, color="#FFFFFF", loc="left",  bbox ={'facecolor':'#8A9EC3', 
                   'alpha':0.3, 'pad':5})
    # ax.axis('off')
    
    # get max score in table_list
    # TODO: Extract the plot as an array
    # plt_array = canvas2rgb_array(fig.canvas)
    # plt_array_shape = plt_array.shape
    # if plt_array.shape[1] != vr_width:
    #     plt_array = cv2.resize()
    plt.savefig('/media/tx-deepocean/Data/DICOMS/demos/Calcification_Score.png', bbox_inches='tight', pad_inches=0)
    # png_to_dcm
    im_frame = Image.open("/media/tx-deepocean/Data/DICOMS/demos/Calcification_Score.png")
    if im_frame.width != vr_width:
        im_frame = im_frame.resize((vr_width, vr_width), Image.ANTIALIAS)
        # RGBA (4x8-bit pixels, true colour with transparency mask)
    file_name = os.path.basename(save_path).replace(".dcm", "")
    np_frame = np.array(im_frame.getdata(), dtype=np.uint8)[:, :3]
    ds.Rows = im_frame.height
    ds.Columns = im_frame.width
    ds.PixelData = np_frame.tobytes()
    ds.InstanceNumber = 398
    edit_tags(ds, "seriesImages" in save_path, full_tag=False, is_vr=True)
    ds.save_as(save_path)

   
def canvas2rgb_array(canvas):
    """Adapted from: https://stackoverflow.com/a/21940031/959926"""
    canvas.draw()
    buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    ncols, nrows = canvas.get_width_height()
    print(f'****ncols: {ncols}, nrows: {nrows}')
    scale = round(math.sqrt(buf.size / 3 / nrows / ncols))
    # return buf.reshape(scale * nrows, scale * ncols, 3)
    return buf.reshape(3, scale * nrows, scale * ncols)

def edit_tags(ds: FileDataset, series: bool = True, full_tag: bool = False, is_vr: bool = False):
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


if __name__ == '__main__':
    main()
    