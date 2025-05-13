import open3d as o3d
import numpy as np
import os
import vtk
import pandas as pd
import matplotlib.pyplot as plt
from scipy import spatial

# Calculate height
def calculate_height(pcd):
    points = np.asarray(pcd.points)
    z_points = sorted(points, key=(lambda z: z[2]))
    min_pts, max_pts = z_points[0][2], z_points[-1][2]
    height = max_pts - min_pts
    return height

# Calculate surface area and volume
def calculate_surfacearea_and_volumn(pcd,mesh_file,i):
    mesh, idx = pcd.compute_convex_hull()
    outfile = os.path.join(mesh_file,str(i)+".ply")
    o3d.io.write_triangle_mesh(outfile,mesh)
    vtkReader = vtk.vtkPLYReader()
    vtkReader.SetFileName(outfile)
    vtkReader.Update()
    polydata = vtkReader.GetOutput()
    mass = vtk.vtkMassProperties()
    mass.SetInputData(polydata)
    mesh_area = mass.GetSurfaceArea()
    mesh_volume = mass.GetVolume()
    return(mesh_area,mesh_volume)

# Calculate projected area
def out_2d_area(pcd):
    # Get 3D coordinates of point cloud
    points = np.asarray(pcd.points)
    # Get XY coordinates of point cloud
    point2d = np.c_[points[:, 0], points[:, 1]]
    # Get convex polygon boundary of planar point cloud
    ch2d = spatial.ConvexHull(point2d)
    area = ch2d.volume
    # Visualize convex polygon boundary results
    plt.figure()
    # Visualize
    ax = plt.subplot(aspect="equal")
    spatial.convex_hull_plot_2d(ch2d, ax=ax)
    plt.title("Point cloud convex hull")
    plt.show()
    return area


# Calculate flower cluster volume
def calculate_voxel_grid_volume(pcd_file_path, step=0.1):
    cloud = o3d.io.read_point_cloud(pcd_file_path)

    # Get point cloud coordinates and range
    point_cloud = np.asarray(cloud.points)
    x_min, y_min, z_min = np.amin(point_cloud, axis=0)
    x_max, y_max, z_max = np.amax(point_cloud, axis=0)

    # Calculate total row, column, and layer numbers
    row = int(np.ceil((x_max - x_min) / step))
    col = int(np.ceil((y_max - y_min) / step))
    cel = int(np.ceil((z_max - z_min) / step))
    grid_size = (row, col, cel)

    # Count points
    grid_indices = np.floor((point_cloud - [x_min, y_min, z_min]) / step).astype(int)

    # Ensure indices are within valid range
    grid_indices = np.clip(grid_indices, 0, [row - 1, col - 1, cel - 1])
    M = np.zeros((row, col, cel), dtype=int)
    for rID, cID, eID in grid_indices:
        M[rID, cID, eID] += 1

    # Calculate the number of non-empty voxels
    num = np.count_nonzero(M)
    grid_volume = num * step ** 3

    return grid_volume


