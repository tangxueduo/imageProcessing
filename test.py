import math

import SimpleITK as sitk
import vtk
from vtk.util.numpy_support import numpy_to_vtk
from vtk.vtkInteractionWidgets import vtkOrientationMarkerWidget
from vtk.vtkRenderingAnnotation import vtkAnnotatedCubeActor


def build_property_vr(hus):
    """
    volume property for ct heart
    """
    opacity_transfer = vtk.vtkPiecewiseFunction()  # 设置不透明度传递函数，此为一维分段传输函数
    color_transfer = vtk.vtkColorTransferFunction()  # 设置颜色传递函数

    trans = [
        0.9,
        0.897,
        0.893,
        0.376,
        0,
        0,
        0,
        0.550,
        0.759,
        1.0,
        1.0,
        1.0,
    ]
    # 避免调整窗宽窗位时显示立方体
    for i in range(len(hus)):
        if -140 <= hus[i] <= -100:
            trans[i] = 0

    for i, opacity in enumerate(trans):
        opacity_transfer.AddPoint(hus[i], opacity)

    cors = [
        [1, 1, 1],
        [0.878431, 1, 0.976470],
        [0.980392, 1, 0.741176],
        [0.862745, 0.082352, 0.011764],
        [0, 0, 0],
        [0, 0, 0],
        [1, 0.509803, 0.313725],
        [1, 0.509803, 0.313725],
        [1, 0.878431, 0.439215],
        [1, 0.976470, 0.850980],
        [1, 1, 1],
        [1, 1, 1],
    ]
    for i, cor in enumerate(cors):
        color_transfer.AddRGBPoint(hus[i], cor[0], cor[1], cor[2])

    volume_property = vtk.vtkVolumeProperty()
    volume_property.SetColor(color_transfer)  # 传输函数颜色
    volume_property.SetScalarOpacity(opacity_transfer)  # 传输函数不透明度
    volume_property.SetInterpolationTypeToLinear()  # 函数插值方法
    volume_property.ShadeOn()  # 阴影
    volume_property.SetAmbient(0.25)  # 环境光
    volume_property.SetDiffuse(0.9)  # 漫反射
    volume_property.SetSpecular(0.1)  # 高光系数
    volume_property.SetSpecularPower(128)  # 高光强度
    volume_property.SetScalarOpacityUnitDistance(1.0)
    return volume_property


def get_vtk_img_data(volume_path):
    """
    :param volum_path: 体数据文件，mhd或者nii.gz
    读取niigz或者mhd文件，返回vtkImageData对象
    """
    sitk_image = sitk.ReadImage(volume_path)
    img_arr = sitk.GetArrayFromImage(sitk_image)
    spacing = sitk_image.GetSpacing()
    origin = sitk_image.GetOrigin()
    direction = sitk_image.GetDirection()

    vtk_matrix = vtk.vtkMatrix3x3()
    for i in range(3):
        for j in range(3):
            vtk_matrix.SetElement(i, j, direction[i * 3 + j])

    vtk_data = numpy_to_vtk(
        num_array=img_arr.ravel(), deep=False, array_type=vtk.VTK_SHORT
    )
    imdata = vtk.vtkImageData()
    imdata.SetDimensions(img_arr.shape[::-1])
    imdata.SetSpacing(spacing)
    imdata.SetOrigin(origin)
    imdata.SetDirectionMatrix(vtk_matrix)
    imdata.GetPointData().SetScalars(vtk_data)
    return imdata


def _calc_angle(camera):
    # 获取相机方向 (roll, pitch, yaw)
    angle = list(map(int, camera.GetOrientation()))
    # 为了页面上 roll 显示正数
    if angle[2] < 0:
        angle[2] += 360

    # 获取相机的焦点、 位置
    fpoint = list(camera.GetFocalPoint())
    cpos = list(camera.GetPosition())
    crelativepos = [cpos[i] - fpoint[i] for i in range(len(cpos))]

    r = math.sqrt(crelativepos[0] ** 2 + crelativepos[1] ** 2 + crelativepos[2] ** 2)

    theta = math.acos(crelativepos[2] / r)
    phi = math.atan2(crelativepos[0], crelativepos[1])
    print(f"origin theta is: {theta}")
    print(f"origin phi is: {phi}")

    # 球坐标转换为地理坐标
    longitude = int(phi * 180 / math.pi)
    latitude = int((math.pi / 2 - theta) * 180 / math.pi)

    print(f"origin longitude is: {longitude}")
    print(f"origin latitude is: {latitude}")

    param_1 = "RAO" if longitude <= 0 else "LAO"
    param_2 = int(180 - longitude) if param_1 == "LAO" else int(180 + longitude)

    param_3 = "Cranial" if 0 <= latitude else "Caudal"
    param_4 = abs(latitude)

    # 投射角度 Roll角度
    projection = [param_1, param_2, param_3, param_4]
    roll = angle[2]
    print(f"****angle: {angle}, projection: {projection}, roll: {roll}")
    return angle, projection, roll


def get_volume_property(render_type, new_ww, new_wl):
    """
    render volume property
    """
    hus = [-1500, -718, -443, -275, -164, -120, -80, -20, 260, 308, 824, 1500]
    if new_ww is not None and new_wl is not None:
        major_hus, minor_hus = [], []
        for hu in hus:
            if hu >= -120:
                major_hus.append(hu)
            else:
                minor_hus.append(-240 - hu)
        ct_max, ct_min = max(major_hus), min(major_hus)
        old_ww = int(ct_max - ct_min)  # 窗宽
        old_wl = int(ct_min + old_ww / 2)  # 窗位
        scale = new_ww / old_ww

        # 计算新的透明度传递函数的 hus
        for i, opacity_hu in enumerate(major_hus):
            tmp_hu = new_wl + (opacity_hu - old_wl) * scale
            major_hus[i] = max(-120, tmp_hu)

        for i, opacity_hu in enumerate(minor_hus):
            tmp_hu = new_wl + (opacity_hu - old_wl) * scale
            minor_hus[i] = max(-120, tmp_hu)

        new_hus = [-240 - hu for hu in minor_hus] + major_hus
        hus = new_hus

    volume_property = build_property_vr(hus)
    return volume_property


def get_render_volume(imdata, vmip, render_type, new_ww, new_wl):
    volume_property = get_volume_property(render_type, new_ww, new_wl)
    if not vmip:
        volume_mapper = vtk.vtkGPUVolumeRayCastMapper()
        volume_mapper.SetInputData(imdata)
        volume_mapper.SetSampleDistance(0.5)
        volume_mapper.SetAutoAdjustSampleDistances(0)
    else:
        volume_mapper = vtk.vtkSmartVolumeMapper()
        volume_mapper.SetInputData(imdata)
        volume_mapper.SetBlendModeToMaximumIntensity()

    volume = vtk.vtkVolume()
    volume.SetMapper(volume_mapper)
    volume.SetProperty(volume_property)
    return volume


def main():
    # colors = vtkNamedColors()
    mhd_file = "/home/turbo/Downloads/vr.mhd"
    imdata = get_vtk_img_data(mhd_file)
    volume = get_render_volume(imdata, False, "vr", 1620, 690)

    renderer = vtk.vtkRenderer()
    renderer.AddVolume(volume)
    # 相机A 方位
    camera = renderer.GetActiveCamera()
    camera.ComputeViewPlaneNormal()
    c = [0.0, 0, 0]
    camera.SetViewUp(0, 0, 1)
    camera.SetPosition(c[0], c[1] - 0.5, c[2])
    camera.SetFocalPoint(c[0], c[1], c[2])
    camera.Azimuth(0)  # 经度
    camera.Elevation(0)
    camera.SetRoll(0)

    print(f"****camera orientation: {camera.GetOrientation()}")
    _calc_angle(camera)

    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindow.SetWindowName("Camera")

    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)

    # Add the actor to the scene
    # renderer.AddActor(actor)
    renderer.SetBackground((0, 0, 0))
    renderer.ResetCamera()

    cube = vtkAnnotatedCubeActor()
    cube.SetXPlusFaceText("L")  # Left
    cube.SetXMinusFaceText("R")  # Right
    cube.SetYPlusFaceText("P")  # Posterior
    cube.SetYMinusFaceText("A")  # Anterior
    cube.SetZPlusFaceText("S")  # Superior/Cranial
    cube.SetZMinusFaceText("I")  # Inferior/Caudal
    cube.SetZFaceTextRotation(-90)
    cube.SetFaceTextScale(0.5)
    cube.GetCubeProperty().SetColor((0, 0, 0))
    cube.GetCubeProperty().SetEdgeColor((0.004, 0.392, 0.996))
    cube.GetCubeProperty().SetLineWidth(2)
    cube.GetCubeProperty().SetEdgeVisibility(1)
    # 文字边颜色
    cube.GetTextEdgesProperty().SetColor((1, 1, 1))
    axes = vtkOrientationMarkerWidget()
    axes.SetOrientationMarker(cube)
    axes.SetInteractor(renderer.GetRenderWindow().GetInteractor())
    # Position bottom right in the viewport.
    axes.SetViewport(0.82, 0.00, 0.94, 0.12)
    axes.SetEnabled(1)
    axes.EnabledOn()
    axes.InteractiveOff()

    # Render and interact
    # renderWindowInteractor.Initialize()
    renderWindow.Render()
    renderWindowInteractor.Start()


if __name__ == "__main__":
    main()
