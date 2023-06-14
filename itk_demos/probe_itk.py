import time
from typing import Dict, Optional, Sequence, Union

import numpy as np
import process
import requests
import SimpleITK as sitk
import sitktools as st
from loguru import logger

# TODO: 以下放在常亮
Numeric = Union[int, float]
NumericArrayType = Union[np.ndarray, Sequence[Numeric]]
CenterlineDictType = Dict[str, Union[float, NumericArrayType]]
SitkTransformType = Union[
    sitk.Transform,
    sitk.AffineTransform,
    sitk.Euler3DTransform,
    sitk.DisplacementFieldTransform,
]
import math

import pydicom


class ProbeGenerater(object):
    def __init__(self, im, seg) -> None:
        self.sitk_interpolator = sitk.sitkLinear
        self.im = im
        self.seg = seg

    def resample(
        self,
        transform: SitkTransformType,
        origin: NumericArrayType,
        nominal_spacing: NumericArrayType,
        result_spacing: NumericArrayType,
        direction: NumericArrayType,
        size: NumericArrayType,
        default_value: Optional[Numeric] = None,
        return_numpy_array: bool = False,
        squeeze_d_axis: bool = True,
    ) -> Union[sitk.Image, np.ndarray]:
        if default_value is None:
            default_value = -1024
        resIm = sitk.Resample(
            self.im,
            np.array(size).tolist(),
            transform,
            self.sitk_interpolator,
            np.array(origin).tolist(),
            np.array(nominal_spacing).tolist(),
            np.array(direction).ravel().tolist(),
            default_value,
            self.im.GetPixelID(),
        )
        resIm.SetSpacing(result_spacing)
        if size[-1] == 1 and squeeze_d_axis:
            resIm = resIm[:, :, 0]
        if return_numpy_array:
            return sitk.GetArrayFromImage(resIm)
        else:
            return resIm

    def probe(
        self,
        centerline,
        output_shape: Sequence[int],
        output_spacing: Sequence[float],
        slice_index: int = 0,
        default_value: Optional[Numeric] = None,
        return_numpy_array: bool = False,
        squeeze_d_axis: bool = True,
    ):
        """im, plaque, plaque_mask 的类型"""
        orgs, dirs, spc = st.getStraightenSliceProps(
            centerline, output_shape, output_spacing
        )
        if len(output_shape) == 2:
            output_shape = tuple(output_shape) + (1,)
        else:
            output_shape = (output_shape[0], output_shape[1], 1)
        spc = np.diag(spc)
        return self.resample(
            transform=sitk.AffineTransform(3),
            origin=orgs[int(slice_index)],
            direction=dirs[int(slice_index)],
            nominal_spacing=spc,
            result_spacing=spc,
            size=output_shape,
            default_value=default_value,
            return_numpy_array=return_numpy_array,
            squeeze_d_axis=squeeze_d_axis,
        )

    @staticmethod
    def get_complete_lines(center_lines, type_show=None):
        """获取所有的中线
        Return belike : {
            "lines": {"vessel1": [[x,y,z], [x,y,z]...]...},
            "types": {"vessel1": [1,2,3,4,5]...}
        }
        """
        out = {"lines": {}, "types": {}}
        # type_show = [""]
        for vessel, line_idxs in center_lines["typeShow"].items():
            # if line_idxs["name"] not in type_show:
            #     continue
            line = []
            types = []
            for idx in line_idxs["data"]:
                idx = str(idx)
                line.extend(center_lines["lines"][idx])
                types.append(
                    {center_lines["types"][idx]: len(center_lines["lines"][idx])}
                )
            out["lines"][vessel] = line
            out["types"][vessel] = types
        logger.info(out["lines"].keys())
        return out

    @staticmethod
    def calculate_frenet(curve_points):
        length = len(curve_points)
        new_points = curve_points[1:].copy()
        new_points.append(curve_points[-1])

        tangents = []
        for i in range(length):
            sub_p = np.subtract(new_points[i], curve_points[i])
            normalize_p = vec3_normalize(sub_p)
            tangents.append(normalize_p)

        normals: list = list()
        bi_normals: list = list()
        n = vec3_normalize(np.subtract(tangents[1], tangents[0]))
        b = vec3_normalize(np.cross(tangents[0], n))
        for i in range(length):
            t = tangents[i]
            n = vec3_normalize(np.cross(b, t))
            b = vec3_normalize(np.cross(t, n))
            normals.append(n)
            bi_normals.append(b)
        tangents[-1] = tangents[-2]
        normals[-1] = normals[-2]
        bi_normals[-1] = bi_normals[-2]
        return tangents, normals, bi_normals


def vec3_normalize(p):
    x, y, z = p
    len_p = x * x + y * y + z * z
    if len_p > 0:
        len_p = 1 / math.sqrt(len_p)
    out = [x * len_p, y * len_p, z * len_p]
    return out


# TODO: 删除main
def main():
    t1 = time.time()
    nii_path = "/media/tx-deepocean/Data/DICOMS/demos/aorta/1.2.156.112605.189250946070725.20181212133247.3.5220.10.nii.gz"
    seg_path = "/media/tx-deepocean/Data/DICOMS/demos/aorta/output_lps-seg.nii.gz"
    im = sitk.ReadImage(nii_path)
    seg = sitk.ReadImage(seg_path)
    ds = pydicom.read_file(
        "/media/tx-deepocean/Data/DICOMS/demos/aorta/1.2.156.112605.189250946070725.20181202033847.3.5248.10/1.2.156.112605.189250946070725.20181202034504.4.3588.23.dcm",
        force=True,
    )
    logger.info(f"***Read data cost: {time.time() - t1}")
    probe_gen = ProbeGenerater(im, seg)

    heart_url = f"http://172.16.3.35:3333/series/1.2.156.112605.189250946070725.20181212133247.3.5220.10/predict/ct_aorta_treeline"
    heart_res = requests.get(heart_url).json()
    treeline = heart_res["treeLines"]["data"]
    # 拼接中线
    complete_lines_map = probe_gen.get_complete_lines(treeline)
    centerlines = complete_lines_map["lines"].get("vessel3")

    tangents, normals, bi_normals = probe_gen.calculate_frenet(centerlines)
    # 根据中线计算切线\法线\副法线
    centerline = {
        "y": np.array(centerlines),
        "T": np.array(tangents),
        "N": np.array(normals),
        "B": np.array(bi_normals),
    }
    t0 = time.time()

    probeItems = [30]
    for i in probeItems:
        probe_im = process.Process.probe(
            centerline=centerline,
            slice_index=i,
            output_shape=[64, 64],
            output_spacing=[0.7, 0.7],
            return_numpy_array=False,
        )
    np_array = sitk.GetArrayFromImage(probe_im)

    probe_im.SetMetaData("0028|1051", "800")
    probe_im.SetMetaData("0028|1050", "300")
    sitk.WriteImage(probe_im, "./demo.dcm")

    ds = pydicom.read_file("demo.dcm", force=True)
    ds.RescaleIntercept = 0
    ds.RescaleSlope = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.SamplesPerPixel = 1
    ds.BitsStored = 16
    ds.BitsAllocated = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1

    ds.PixelData = np_array.astype("int16").tobytes()
    ds.save_as("demo1.dcm")
    logger.info(f"Cost time is : {time.time() - t0}")


if __name__ == "__main__":
    main()
