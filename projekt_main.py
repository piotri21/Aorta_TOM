import numpy as np
import matplotlib.pyplot as plt
import nrrd
import vtk
from vtk.util import numpy_support

data, header = nrrd.read('../DATA/Dongyang/D1/D1.seg.nrrd')
print(data.shape)

# Tutaj wstawić przetwarzanie danych, nazwać zmienną "processed_data"
processed_data = data

vtk_data = vtk.vtkImageData()
vtk_data.SetDimensions(processed_data.shape)
vtk_data.SetSpacing(header['space directions'][0][0], header['space directions'][1][1], header['space directions'][2][2])
vtk_data.SetOrigin(header['space origin'][0], header['space origin'][1], header['space origin'][2])

flat = data.ravel(order='F')
vtk_array = numpy_support.numpy_to_vtk(num_array=flat, deep=True, array_type=vtk.VTK_FLOAT)
vtk_data.GetPointData().SetScalars(vtk_array)

colors = vtk.vtkNamedColors()
colors.SetColor('aorta_red', [255, 30, 30, 255])

a_renderer = vtk.vtkRenderer()
ren_win = vtk.vtkRenderWindow()
ren_win.AddRenderer(a_renderer)
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(ren_win)
ren_win.SetSize(640, 480)

aorta_extractor = vtk.vtkMarchingCubes()
aorta_extractor.SetInputData(vtk_data)
aorta_extractor.SetValue(0, 1.0)
aorta_extractor.Update()

aorta_stripper = vtk.vtkStripper()
aorta_stripper.SetInputConnection(aorta_extractor.GetOutputPort())
aorta_stripper.Update()

aorta_mapper = vtk.vtkPolyDataMapper()
aorta_mapper.SetInputConnection(aorta_stripper.GetOutputPort())
aorta_mapper.ScalarVisibilityOff()

aorta = vtk.vtkActor()
aorta.SetMapper(aorta_mapper)
aorta.GetProperty().SetDiffuseColor(colors.GetColor3d('aorta_red'))
aorta.GetProperty().SetSpecular(0.3)
aorta.GetProperty().SetSpecularPower(20.0)

a_camera = vtk.vtkCamera()
a_camera.SetViewUp(0, 0, -1)
a_camera.SetPosition(0, -1, 0)
a_camera.SetFocalPoint(0, 0, 0)
a_camera.ComputeViewPlaneNormal()
a_camera.Azimuth(30.0)
a_camera.Elevation(30.0)

a_renderer.AddActor(aorta)
a_renderer.SetActiveCamera(a_camera)
a_renderer.SetBackground(colors.GetColor3d('black'))
a_renderer.ResetCamera()
ren_win.Render()
iren.Initialize()
iren.Start()