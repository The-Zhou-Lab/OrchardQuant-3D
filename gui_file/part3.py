# Import Necessary Libraries
import open3d as o3d
import numpy as np
from pc_skeletor.laplacian import SLBC
import laspy
import os


def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


# Convert LAS Format File to PCD Format File
def las_to_pcd(las_folder, pcd_folder, is_color=True):
    create_folder_if_not_exists(pcd_folder)
    pcd_file_paths = []
    offset_list = []

    for las_file in os.listdir(las_folder):
        las_path = os.path.join(las_folder, las_file)
        las = laspy.read(las_path)

        x, y, z = las.x, las.y, las.z
        x_offset = x - np.min(x)
        y_offset = y - np.min(y)
        z_offset = z - np.min(z)
        offset_list.append([np.min(x), np.min(y), np.min(z)])

        points = np.vstack((x_offset, y_offset, z_offset)).transpose()
        # Determine whether there is color information
        if is_color and hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
            r, g, b = las.red, las.green, las.blue
            colors = np.vstack((r / 65536, g / 65536, b / 65536)).transpose()
        else:
            colors = None

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)

        pcd_file_path = os.path.join(pcd_folder, las_file[:-4] + ".pcd")
        pcd_file_paths.append(pcd_file_path)
        o3d.io.write_point_cloud(pcd_file_path, pcd, write_ascii=True)

    return pcd_file_paths, offset_list

# Obtain the Main Trunk and Branches of Pear Trees
def extract_trunk_points(pcd, dist, radius):
    list_array = []
    points = np.asarray(pcd.points)
    max_pts = np.amax(points, axis=0)
    min_pts = np.amin(points, axis=0)
    # Calculate the number of vertical segments
    heis = np.ceil((max_pts[2] - min_pts[2]) / dist)
    # Vertical slicing of the point cloud
    for i in range(int(heis)):
        segment_points = points[
            np.where((points[:, 2] >= i * dist + min_pts[2]) & (points[:, 2] < (i + 1) * dist + min_pts[2]))]
        segment_pcd = o3d.geometry.PointCloud()
        segment_pcd.points = o3d.utility.Vector3dVector(segment_points)
        # If there is only one cluster, it is considered the trunk part
        if segment_points.shape[0] != 0:
            with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
                labels = np.array(segment_pcd.cluster_dbscan(eps=radius, min_points=1, print_progress=False))
            max_label = labels.max()
            clusters = [np.where(labels == j)[0] for j in range(max_label + 1)]
            if len(clusters) == 1:
                for j in range(len(clusters)):
                    clusters_cloud = segment_pcd.select_by_index(clusters[j])
                    points_cloud = np.asarray(clusters_cloud.points)
                    list_array.append(points_cloud)
            # If there is more than one cluster, branches are starting to appear
            elif len(clusters) != 1:
                number = i
                break
    # Concatenate points of the trunk part
    for k in range(len(list_array)):
        if k == 0:
            list_point_array_path = list_array[0]
        else:
            list_point_array_path = np.concatenate((list_array[k], list_point_array_path), axis=0)
    # Convert array to point cloud
    pcd_trunk = o3d.geometry.PointCloud()
    pcd_trunk.points = o3d.utility.Vector3dVector(np.asarray(list_point_array_path))
    other_points = points[np.where((points[:, 2] >= number * dist + min_pts[2]))]
    pcd_branch = o3d.geometry.PointCloud()
    pcd_branch.points = o3d.utility.Vector3dVector(np.asarray(other_points))
    print("pcd_trunk",pcd_trunk)
    print("pcd_branch",pcd_branch)
    return pcd_trunk, pcd_branch

# Obtain the pear tree skeleton point cloud using the Semantic Laplacian Beltrami Component (SLBC) algorithm
def extract_skeleton(trunk_pcd, branch_pcd, semantic_weighting, down_sample):
    slbc = SLBC(point_cloud={'trunk': trunk_pcd, 'branches': branch_pcd},
                semantic_weighting=semantic_weighting,
                down_sample=down_sample,
                debug=False)
    slbc.extract_skeleton()
    slbc.extract_topology()
    return slbc


# Invoking function
def process_point_clouds(pcd_folder, result_folder, radius):
    skeleton_folder = os.path.join(result_folder, "skeleton")
    create_folder_if_not_exists(skeleton_folder)
    for pcd_file in os.listdir(pcd_folder):
        pcd_path = os.path.join(pcd_folder, pcd_file)
        print(pcd_path)
        pcd = o3d.io.read_point_cloud(pcd_path)
        # extract trunk and branch point cloud
        try:
            trunk_pcd, branch_pcd = extract_trunk_points(pcd, 0.001, radius)
        except UnboundLocalError:
            trunk_pcd, branch_pcd = extract_trunk_points(pcd, 0.001, 0.5)
        trunk_folder = os.path.join(skeleton_folder, "trunk", pcd_file[:-4])
        branch_folder = os.path.join(skeleton_folder, "branch", pcd_file[:-4])
        create_folder_if_not_exists(trunk_folder)
        create_folder_if_not_exists(branch_folder)

        trunk_path = os.path.join(trunk_folder, "trunk_" + pcd_file)
        branch_path = os.path.join(branch_folder, "branch_" + pcd_file)
        o3d.io.write_point_cloud(trunk_path, trunk_pcd)
        o3d.io.write_point_cloud(branch_path, branch_pcd)
        # extract skeleton
        slbc = extract_skeleton(trunk_pcd, branch_pcd, semantic_weighting=5, down_sample=0.01)
        mesh_folder = os.path.join(skeleton_folder, "mesh", pcd_file[:-4])
        create_folder_if_not_exists(mesh_folder)
        slbc.save(mesh_folder)