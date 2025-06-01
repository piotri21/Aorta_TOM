import numpy as np
import matplotlib.pyplot as plt
import nrrd
import vtk
from scipy.ndimage import binary_erosion, distance_transform_edt
import skimage
import scipy.ndimage
from vtkmodules.util import numpy_support
import pandas as pd
from pathlib import Path
import networkx as nx



# Tutaj potrzebujemy tylko posegmentowanych danych do walidacji modelu

data_dir = Path('../DATA')
seg_files = sorted([f for f in data_dir.glob('**/*seg*.nrrd')])

def get_base_name(path):
    return path.stem.replace('.seg', '').replace('_seg', '')

seg_dict = {get_base_name(f): f for f in seg_files}

common_keys = set(seg_dict.keys())
df = pd.DataFrame({
    'seg_path': [str(seg_dict[k]) for k in common_keys]
})

print(df.head())

#data, header = nrrd.read(df.seg_path[1])
data, header = nrrd.read("../DATA/Dongyang/D1/D1.seg.nrrd")
print(data.shape)

def process(data):
    skeleton = skimage.morphology.skeletonize(data)
    distance = scipy.ndimage.distance_transform_edt(data)
    struct = scipy.ndimage.generate_binary_structure(3, 3)
    points = np.argwhere(skeleton)
    G = nx.Graph()
    for pt in points:
        pt = tuple(pt)
        G.add_node(pt, thickness=distance[pt])
        for offset in np.argwhere(struct) - 1:
            neighbor = tuple(pt + offset)
            if (0 <= neighbor[0] < skeleton.shape[0] and
                0 <= neighbor[1] < skeleton.shape[1] and
                0 <= neighbor[2] < skeleton.shape[2]):
                if skeleton[neighbor]:
                    #print(f"Adding edge from {pt} to {neighbor}")
                    G.add_edge(pt, neighbor)

    # Najgrubszy punkt
    max_thick_idx = np.argmax(distance * skeleton)
    thickest_point = np.unravel_index(max_thick_idx, data.shape)
    
    # Upewnij się, że najgrubszy punkt jest w grafie
    if thickest_point not in G.nodes:
        skeleton_points = np.argwhere(skeleton)
        distances_to_thick = np.linalg.norm(skeleton_points - np.array(thickest_point), axis=1)
        closest_idx = np.argmin(distances_to_thick)
        thickest_point = tuple(skeleton_points[closest_idx])

    # Szukamy dwóch najdalszych punktów na szkielecie (diameter)
    #lengths = dict(nx.all_pairs_path_length(G))
    ends = [pt for pt in G.nodes if len(list(G.neighbors(pt))) <= 2]
    print(f"Znaleziono {len(ends)} końców szkieletu")
    paths = []
    for end1 in ends:
        for end2 in ends:
            if end1 != end2:
                try:
                    path = nx.shortest_path(G, source=end1, target=end2)
                    paths.append(path)
                except nx.NetworkXNoPath:
                    continue
    lognest_path_length = 0
    print(f"Znaleziono {len(paths)} ścieżek pomiędzy końcami")
    longest_ends = (None, None)
    for path in paths:
        if thickest_point in path:
            if len(path) > lognest_path_length:
                lognest_path_length = len(path)
                longest_ends = (path[0], path[-1])

    # Najdłuższa ścieżka przechodząca przez najgrubszy punkt
    path = nx.shortest_path(G, source=longest_ends[0], target=longest_ends[1])

    # Zamiana ścieżki na maskę 3D
    mask = np.zeros(data.shape, dtype=np.uint8)
    for pt in path:
        mask[pt] = 1
    return mask


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
a_renderer.AddActor(create_actor(scipy.ndimage.binary_dilation(processed_data, iterations=3), 'white', opacity=1.0))
a_renderer.AddActor(create_actor(scipy.ndimage.binary_dilation(skimage.morphology.skeletonize(data)), 'blue', opacity=1.0))

a_renderer.SetActiveCamera(a_camera)
a_renderer.SetBackground(colors.GetColor3d('black'))
a_renderer.ResetCamera()
ren_win.Render()
iren.Initialize()
iren.Start()