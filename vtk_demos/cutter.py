import vtk
import numpy as np
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonDataModel import vtkPlane
from vtkmodules.vtkFiltersCore import vtkCutter
from vtkmodules.vtkFiltersSources import vtkCubeSource
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer
)
colors = vtkNamedColors()

# 导入data
reader = vtk.vtkNIFTIImageReader()
reader.SetFileName('/media/tx-deepocean/Data/DICOMS/demos/cerebral_1/aneurysm_mask.nii.gz')
reader.Update()



# bounds
bounds = [0,0,0,0,0,0]
bounds = reader.GetOutput().GetBounds(bounds)

# create a plane to cut,here it cuts in the XZ direction (xz normal=(1,0,0);XY =(0,0,1),YZ =(0,1,0)
plane = vtkPlane()
plane.SetOrigin(10, 0, 100)
# 根据matrix 计算法向量
M = np.array([ [-0.2080068212629158,	 0.5134529000437192,	 0.8325258444906033],
      [-0.35252130345789634,	 0.7545900135604813,	 -0.55346421929799],
      [-0.9123934967526036,	 -0.40860742880115053,	 0.024043215510192506]])
M.T
# plane.SetNormal(0.8325258444906033, -0.24993655260345934, -0.49440109013624367)
plane.SetNormal(0, 0, 1)

# 三维空间中渲染对象最常用的 vtkProp 子类是 vtkActor(表达场景中的几何数据)和 vtkVolume(表达场景中的体数据)
"""vtkProp子类负责确定渲染场景中对象的位置、大小和方向信息。
Prop依赖于两个对象(Prop一词来源于戏剧里的“道具”，在VTK里表示的是渲染场景中可以看得到的对象。)
一个是Mapper(vtkMapper)对象，负责存放数据和渲染信息，另一个是属性(vtkProperty)对象，负责控制颜色、不透明度等参数。
"""
# create cutter
cutter = vtkCutter()
cutter.SetCutFunction(plane)
# cutter->GenerateValues(numberOfCuts, .99, .99 * high);
cutter.SetInputConnection(reader.GetOutputPort())
cutter.Update()
cutterMapper = vtkPolyDataMapper()
cutterMapper.SetInputConnection(cutter.GetOutputPort())

# create plane actor
planeActor = vtk.vtkActor()
planeActor.GetProperty().SetColor(0,1,1)
planeActor.GetProperty().SetLineWidth(2)
planeActor.GetProperty().SetAmbient(1.0)
planeActor.GetProperty().SetDiffuse(0.0)
planeActor.SetMapper(cutterMapper)

colorTable = vtk.vtkLookupTable()

# create modelActor
modelMapper = vtk.vtkPolyDataMapper()
modelMapper.SetInputConnection(reader.GetOutputPort())

modelActor = vtkActor()
modelActor.GetProperty().SetColor(0,1,0 )
# modelActor.GetProperty().SetOpacity(0.5)
modelActor.SetMapper(modelMapper)

# contourFilter = vtk.vtkContourFilter()
# contourFilter.SetValue(0, 100)
# contourFilter.GenerateValues(1, 0, 500)
# contourFilter.Update()

# contourMapper = vtk.vtkPolyDataMapper()
# contourMapper.SetInputConnection(contourFilter.GetOutputPort())

# contourActor = vtkActor()
# contourActor.GetProperty().SetColor(1.0,1.0,0.0)
# contourActor.GetProperty().SetOpacity(0.5)
# contourActor.SetMapper(contourMapper)

# create renderers and add actors of plane and cube
ren = vtk.vtkRenderer()
ren.AddActor(planeActor)
ren.AddActor(modelActor)
# ren.AddActor(contourActor)
ren.SetBackground(255,255,255)

# Add renderer to renderwindow and render
renWin = vtkRenderWindow()
renWin.AddRenderer(ren)
renWin.SetSize(600, 600)
renWin.SetWindowName('Cutter')
renWin.Render()

iren = vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)

camera = ren.GetActiveCamera()
# camera.SetPosition(-37.2611, -86.2155, 44.841)
# camera.SetFocalPoint(0.569422, -1.65124, -2.49482)
# camera.SetViewUp(0.160129, 0.42663, 0.890138)
# camera.SetDistance(10)
# camera.SetClippingRange(55.2019, 165.753)
camera.SetPosition(0, -1, 0)
camera.SetFocalPoint(0, 0, 0)
camera.SetViewUp(0, 0, 1)
# camera.SetDistance(10)
# camera.SetClippingRange(55.2019, 165.753)
ren.ResetCamera()
renWin.Render()

iren.Start()
renWin.Finalize()
iren.TerminateApp()