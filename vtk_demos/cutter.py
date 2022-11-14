#!/usr/bin/env python
# -*- coding: utf-8 -*-

# noinspection PyUnresolvedReferences
import vtk
import vtkmodules.vtkInteractionStyle
# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonCore import vtkIdList
from vtkmodules.vtkCommonDataModel import vtkPlane
from vtkmodules.vtkFiltersCore import (
    vtkCutter,
    vtkStripper
)
from vtkmodules.vtkFiltersSources import vtkSphereSource
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer
)
import SimpleITK as sitk
from vtkmodules.util.vtkImageImportFromArray import vtkImageImportFromArray
from vtkmodules.vtkFiltersCore import vtkContourFilter
from vtkmodules.vtkFiltersModeling import vtkOutlineFilter


def vtk_image_from_array(slice_data, spacing, origin):
    vtk_image = vtkImageImportFromArray()
    vtk_image.SetArray(slice_data)
    vtk_image.SetDataSpacing(spacing)
    vtk_image.SetDataOrigin(origin)
    # vtk_image.SetDataExtent(extent)
    vtk_image.Update()
    return vtk_image

nii_path = "/media/tx-deepocean/Data/DICOMS/demos/cerebral_1/cerebral-seg.nii.gz"
# img = sitk.ReadImage(nii_path)
# spacing = img.GetSpacing()
# origin = img.GetOrigin()
# arr = sitk.GetArrayFromImage(img)
# vtk_img = vtk_image_from_array(arr, spacing, origin)
reader = vtk.vtkNIFTIImageReader()
reader.SetFileName(nii_path)
reader.Update()

def main():
    colors = vtkNamedColors()
    lineColor = colors.GetColor3d('yellow')
    modelColor = colors.GetColor3d('silver')
    backgroundColor = colors.GetColor3d('black')

    modelSource = vtkSphereSource()

    plane = vtkPlane()

    cutter = vtkCutter()
    cutter.SetInputConnection(reader.GetOutputPort())
    cutter.SetCutFunction(plane)
    #  切割10个， 从-0.5位置开始，
    cutter.GenerateValues(10, -0.5, 0.5)

    # stripper 获取三角片
    stripper = vtkStripper()
    stripper.SetInputConnection(cutter.GetOutputPort())
    stripper.JoinContiguousSegmentsOn()

    # 将三角片映射为几何数据
    linesMapper = vtkPolyDataMapper()
    linesMapper.SetInputConnection(stripper.GetOutputPort())

    lines = vtkActor()
    lines.SetMapper(linesMapper)
    lines.GetProperty().SetDiffuseColor(lineColor)
    lines.GetProperty().SetLineWidth(3.)

    # 提取contour， 渲染管线filter -> mapper -> actor 
    iso = vtkContourFilter()
    iso.SetInputConnection(cutter.GetOutputPort())
    iso.GenerateValues(12, -100, 1150)

    isoMapper = vtkPolyDataMapper()
    isoMapper.SetInputConnection(iso.GetOutputPort())
    isoMapper.ScalarVisibilityOff()

    isoActor = vtkActor()
    isoActor.SetMapper(isoMapper)
    isoActor.GetProperty().SetColor(colors.GetColor3d('yellow'))
    

    outline = vtkOutlineFilter()
    outline.SetInputConnection(cutter.GetOutputPort())

    outlineMapper = vtkPolyDataMapper()
    outlineMapper.SetInputConnection(outline.GetOutputPort())

    outlineActor = vtkActor()
    outlineActor.SetMapper(outlineMapper)
    outlineActor.GetProperty().SetColor(colors.GetColor3d("blue"))

    renderer = vtkRenderer()
    renderWindow = vtkRenderWindow()

    renderWindow.AddRenderer(renderer)
    renderWindow.SetSize(640, 480)
    renderWindow.SetWindowName('ExtractPolyLinesFromPolyData')

    interactor = vtkRenderWindowInteractor()
    interactor.SetRenderWindow(renderWindow)

    # Add the actors to the renderer.
    renderer.AddActor(isoActor)
    renderer.AddActor(outlineActor)
    renderer.AddActor(lines)
    renderer.SetBackground(backgroundColor)
    # 经度纬度
    renderer.GetActiveCamera().Azimuth(-45)
    renderer.GetActiveCamera().Elevation(-22.5)
    renderer.ResetCamera()

    # This starts the event loop and as a side effect causes an
    # initial render.
    renderWindow.Render()
    interactor.Initialize()
    interactor.Start()

    # Extract the lines from the polydata.
    numberOfLines = cutter.GetOutput().GetNumberOfLines()

    print('-----------Lines without using vtkStripper')
    print('There are {0} lines in the polydata'.format(numberOfLines))

    numberOfLines = iso.GetOutput().GetNumberOfLines()
    points = iso.GetOutput().GetPoints()
    cells = iso.GetOutput().GetLines()
    cells.InitTraversal()

    print('-----------Lines using vtkStripper')
    print('There are {0} lines in the polydata'.format(numberOfLines))

    indices = vtkIdList()
    lineCount = 0

    while cells.GetNextCell(indices):
        print('Line {0}:'.format(lineCount))
        for i in range(indices.GetNumberOfIds()):
            point = points.GetPoint(indices.GetId(i))
            print('\t({0:0.6f} ,{1:0.6f}, {2:0.6f})'.format(point[0], point[1], point[2]))
        lineCount += 1


if __name__ == '__main__':
    main()
