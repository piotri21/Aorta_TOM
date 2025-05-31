import numpy as np
import matplotlib.pyplot as plt
import nrrd
import vtk
from scipy.ndimage import binary_erosion, distance_transform_edt
import skimage
import scipy.ndimage
from vtkmodules.util import numpy_support

data, header = nrrd.read('../DATA/Dongyang/D1/D1.seg.nrrd')
print(data.shape)

def process(data):
    skeleton = skimage.morphology.skeletonize(data)
    return skeleton

processed_data = process(data)
    

colors = vtk.vtkNamedColors()
colors.SetColor('aorta_red', [255, 30, 30, 255])

a_renderer = vtk.vtkRenderer()
ren_win = vtk.vtkRenderWindow()
ren_win.AddRenderer(a_renderer)
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(ren_win)
ren_win.SetSize(640, 480)

def create_actor(data, color_name, opacity=1.0):
    vtk_data = vtk.vtkImageData()
    vtk_data.SetDimensions(data.shape)
    vtk_data.SetSpacing(header['space directions'][0][0], header['space directions'][1][1], header['space directions'][2][2])
    vtk_data.SetOrigin(header['space origin'][0], header['space origin'][1], header['space origin'][2])

    flat = data.ravel(order='F')
    vtk_array = numpy_support.numpy_to_vtk(num_array=flat, deep=True, array_type=vtk.VTK_FLOAT)
    vtk_data.GetPointData().SetScalars(vtk_array)

    extractor = vtk.vtkMarchingCubes()
    extractor.SetInputData(vtk_data)
    extractor.SetValue(0, 1.0)
    extractor.Update()

    stripper = vtk.vtkStripper()
    stripper.SetInputConnection(extractor.GetOutputPort())
    stripper.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(stripper.GetOutputPort())
    mapper.ScalarVisibilityOff()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetDiffuseColor(colors.GetColor3d(color_name))
    actor.GetProperty().SetSpecular(0.3)
    actor.GetProperty().SetSpecularPower(20.0)
    actor.GetProperty().SetOpacity(opacity)

    return actor

a_camera = vtk.vtkCamera()
a_camera.SetViewUp(0, 0, -1)
a_camera.SetPosition(0, -1, 0)
a_camera.SetFocalPoint(0, 0, 0)
a_camera.ComputeViewPlaneNormal()
a_camera.Azimuth(30.0)
a_camera.Elevation(30.0)

a_renderer.AddActor(create_actor(data, 'aorta_red', opacity=0.5))

#dilate the processed data to enhance visibility
a_renderer.AddActor(create_actor(scipy.ndimage.binary_dilation(processed_data, iterations=1), 'white', opacity=1.0))

a_renderer.SetActiveCamera(a_camera)
a_renderer.SetBackground(colors.GetColor3d('black'))
a_renderer.ResetCamera()
ren_win.Render()
iren.Initialize()
iren.Start()