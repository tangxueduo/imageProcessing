import pydicom
import nibabel as nib
import logging
import SimpleITK as sitk

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d  %H:%M:%S %a",
)

file = "/media/tx-deepocean/Data/DICOMS/ct_cerebral/CN010002-13696724/1.2.840.113619.2.416.10634142502611409964348085056782520111/1.2.840.113619.2.416.77348009424380358976506205963520437809/1.2.840.113619.2.416.2106685279137426449336212617073446717.21"
nii_file = "/media/tx-deepocean/Data/DICOMS/RESULT/volume/1.2.840.113619.2.416.77348009424380358976506205963520437809.nii.gz"
series_dir = "/media/tx-deepocean/Data/DICOMS/ct_heart/CE027003-P0215270/1.2.840.113564.345052760333.90096.637571082457277115.1493/1.2.840.113704.7.32.07.5.1.4.76346.30000021052006532895300034627"

patient_age = series_reader.GetMetaData(0, "0010|1010")
# patient_sex = image3D.GetMetaData("0010|1010")

# 获取该序列对应的3D图像
image3D = series_reader.Execute()

# 查看该3D图像的尺寸
print(image3D.GetSize())

# 将序列保存为单个的NRRD文件
# sitk.WriteImage(image3D, 'img3D.nrrd')

# get metadata

ds = pydicom.read_file(file)
WindowWidth = ds.get("WindowWidth", 800)
logging.info(f'WindowWidth: {WindowWidth}')

img = nib.load(nii_file)