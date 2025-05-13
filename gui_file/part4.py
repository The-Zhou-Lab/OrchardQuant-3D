# Import libraries
import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree
import dijkstra
import laspy
import os
import matplotlib.pyplot as plt
import scipy

def write_to_pcd(path):
    # Read the LAS file
    inFile = laspy.read(path)
    x,y,z = inFile.x, inFile.y, inFile.z
    # Remove offset.
    x_offset = x - np.min(x)
    y_offset = y - np.min(y)
    z_offset = z - np.min(z)
    points = zip(x_offset,y_offset,z_offset)
    # Write to pcd.
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    offset = [[np.min(x),np.min(y),np.min(z)]]
    return pcd,offset

def analyze_point_distances(pcd):
    points = np.asarray(pcd.points)
    unique_points = np.unique(points, axis=0)
    pcd_unique = o3d.geometry.PointCloud()
    pcd_unique.points = o3d.utility.Vector3dVector(unique_points)
     # Calculate nearest neighbor distances after deduplication
    nndist = np.array(pcd_unique.compute_nearest_neighbor_distance())
    # Get distance statistics
    min_distance = np.min(nndist)
    max_distance = np.max(nndist)
    mean_distance = np.mean(nndist)
    return min_distance, max_distance, mean_distance

def get_diameter_points(segment_pcd):
    points = np.asarray(segment_pcd.points)
    # Get coordinates in XY plane
    points_2d = points[:, :2]
    # Calculate distance matrix between point pairs
    distances = scipy.spatial.distance.cdist(points_2d, points_2d)
    # Get diameter
    diameter = np.max(distances)
    return diameter

def get_diameter(pcd):
    min_distance, max_distance, mean_distance = analyze_point_distances(pcd)
    points = np.asarray(pcd.points)
    min_pts = np.amin(points, axis=0)
    # Calculate the number of vertical segments
    # Vertical slicing of the point cloud
    segment_points = points[
            np.where((points[:, 2] >= min_pts[2]) & (points[:, 2] <  max_distance + min_pts[2]))]
    segment_pcd = o3d.geometry.PointCloud()
    segment_pcd.points = o3d.utility.Vector3dVector(segment_points)
        # If there is only one cluster, it is considered the trunk part
    diameter = get_diameter_points(segment_pcd)
    return diameter


# Euclidean Distance Clustering Algorithm
def cluster_points(pcd, radius, min_cluster_size=1, max_cluster_size=10000):
    # kd tree
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    points_length = len(pcd.points)
    # Initializes a list of the same size as the point cloud, recording whether each point has been visited
    visited = [-1] * points_length
    clusters = []
    for index in range(points_length):
        # If the point has already been visited, skip it
        if visited[index] == 1:
            continue
        active_points = []
        active_index = 0
        # Mark the current point as visited and add it to the active point list
        active_points.append(index)
        visited[index] = 1
        while active_index < len(active_points):
            [k, indices, _] = kdtree.search_radius_vector_3d(pcd.points[active_points[active_index]],radius)
            if k == 1:
                active_index += 1
                continue
            for i in range(k):
                if indices[i] == points_length or visited[indices[i]] == 1:
                    continue
                active_points.append(indices[i])
                visited[indices[i]] = 1
            active_index += 1
        # If the cluster size is within the specified range, the cluster is logged
        if max_cluster_size > len(active_points) >= min_cluster_size:
            clusters.append(active_points)
    return clusters

def concatenate_points(points):
    # Concatenates multiple arrays of points into a single array
    for k in range(len(points)):
        if k ==0:
            points_concatenate = points[0]
        else:
            points_concatenate  = np.concatenate((points[k],points_concatenate),axis=0)
    return points_concatenate

def cultivate_length(points):
    # Calculate the total distance between points
    points_dist = []
    for i in range(len(points)):
        if i != 0:
            dist = np.linalg.norm(np.array(points[i]) - np.array(points[i - 1]))
            points_dist.append(dist)
    dist_sum = np.sum(points_dist)
    return dist_sum

def make_graph(pcd, radius):
    # Construct a 3D graph using graph theory
    graph = dijkstra.Graph()
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    target_points = np.asarray(pcd.points)
    for i in range(len(target_points)):
        # Use a radius-based method to construct connections between points
        [k, idx, d] = pcd_tree.search_radius_vector_3d(target_points[i], radius)
        for j in range(k):
            graph.add_edge(str(target_points[i]), str(target_points[idx[j]]), d[j])
    return graph

def get_path_points(start_point, end_point, graph):
    dijkstra_graph = dijkstra.DijkstraSPF(graph, str(start_point))
    # Retrieve the shortest path from the start point to the end point
    get_point = dijkstra_graph.get_path(str(end_point))
    dijkstra_path = []
    # Iterate through each point in the path
    for i in range(len(get_point)):
        remove_str = get_point[i].strip('[]').split()
        # Convert a list of strings to a list of floating-point numbers
        numb = list(map(float, remove_str))
        dijkstra_path.append(numb)
    return dijkstra_path

def find_nearest_point(branch_points, all_points, graph):
    # Initialize a list to store the distance from each branch point to its nearest point
    points_dist = []
    pcd_trunk = o3d.geometry.PointCloud()
    pcd_trunk.points = o3d.utility.Vector3dVector(np.asarray(all_points))
    pcd_tree = o3d.geometry.KDTreeFlann(pcd_trunk)
    # Iterate over each branch point to find its nearest point in the point cloud
    for i in range(len(branch_points)):
        [k, idx, d] = pcd_tree.search_knn_vector_3d(branch_points[i], 1)
        dist = np.linalg.norm(np.asarray(branch_points[i]) - np.asarray(pcd_trunk.points)[idx][0])
        points_dist.append(dist)
    # Find the index of the smallest distance and return the corresponding branch point
    index = np.where(points_dist == np.amin(points_dist))[0]
    branch_point = branch_points[index[0]]
    return branch_point

def remove_points_not_need(points_all, points_not_need):
    # Initialize a list to store the points that need to be kept
    points_need = []
    for i in range(len(points_all)):
        if (np.round(points_all[i], 4) == np.round(points_not_need, 4)).all(1).any():
            pass
        else:
            points_need.append(points_all[i])
    # Return the list of needed points
    return points_need

def remove_affect_points(array_affect):
    mask = np.ones(len(array_affect), dtype=bool)
    # Iterate through all paths, checking if each path is a subset of any other path
    for i, arr in enumerate(array_affect):
        other_arrays = np.delete(array_affect, i, axis=0)
        for other in other_arrays:
            if np.all(np.isin(arr, other)):
                mask[i] = False
                break
    filtered_array = array_affect[mask]
    return filtered_array

def find_outside_points(pcd,input_radius,trunk_diameter):
    points = np.asarray(pcd.points)
    min_distance, max_distance, mean_distance = analyze_point_distances(pcd)
    tree = cKDTree(points)
    points_get = []
    # Iterate through the set of points to find the outermost edge points
    for i in range(len(points)):
        index = tree.query_ball_point(points[i],0.02)
        point_need1 = points[np.where(points[index][:,2]>points[i][2])]
        point_need2 = points[np.where(points[index][:,2]<points[i][2])]
        if len(point_need1)==0 or len(point_need2)==0:
            points_get.append(points[i])
    z_points = sorted(points, key=(lambda z: z[2]))
    min_pts = z_points[0]
    if input_radius == None:
        results = False
        k = 0
        array_affect = []
        while not results:
            current_radius = (2*(k+1)) * trunk_diameter
            print("current_radius",current_radius)

            try:
                graph = make_graph(pcd,current_radius)
                array_affect.clear()
                # For each edge point, find the path to the lowest point
                for i in range(len(points_get)):
                    path = get_path_points(points_get[i],min_pts,graph)
                    array_affect.append(path)
                results = True
            except KeyError:
                print("failed",current_radius)
                k += 1
        # Use a shortest path algorithm to remove incorrectly identified edge points
        iltered_array = remove_affect_points(np.asarray(array_affect))
        out = []
        for i in range(len(iltered_array)):
            out.append(iltered_array[i][0])
        return out,graph,current_radius
    else:
        array_affect = []
        graph = make_graph(pcd,input_radius)
        for i in range(len(points_get)):
            path = get_path_points(points_get[i],min_pts,graph)
            array_affect.append(path)
        iltered_array = remove_affect_points(np.asarray(array_affect))
        out = []
        for i in range(len(iltered_array)):
            out.append(iltered_array[i][0])
        return out,graph,None

def clusters_to_pcd(pcd, dist, num):
    # Cluster the point cloud using the Euclidean distance clustering algorithm
    clusters = cluster_points(pcd, dist, min_cluster_size=num, max_cluster_size=100000)
    points = np.asarray(pcd.points)
    point_list = []
    # Iterate through each cluster, retrieve the points within, and add them to the list
    for i in clusters:
        point = points[i]
        point_list.append(point)
    concatenated_clusters = concatenate_points(point_list)
    pcd_trans = o3d.geometry.PointCloud()
    pcd_trans.points = o3d.utility.Vector3dVector(np.asarray(concatenated_clusters))
    return pcd_trans

def remove_support_structure_graph(pcd,trunk_diameter):
    pcd_trans = clusters_to_pcd(pcd, 0.5, 5)
    # graph = make_graph(pcd_trans, 0.5)
    outside_points,graph,current_radius = find_outside_points(pcd_trans, None,trunk_diameter)
    z_points = sorted(np.asarray(pcd_trans.points), key=(lambda z: z[2]))
    min_pts = z_points[0]
    points = []
    # Get the shortest paths from the lowest point to all outermost points
    for i in range(len(outside_points)):
        path = get_path_points(min_pts, np.asarray(outside_points)[i], graph)
        points.append(path)
    concatenated_path = concatenate_points(points)
    concatenated_path_unique = np.unique(concatenated_path, axis=0)
    # Save the point cloud after removing the support structures
    pcd_remove = o3d.geometry.PointCloud()
    pcd_remove.points = o3d.utility.Vector3dVector(np.asarray(concatenated_path_unique))
    return pcd_remove, pcd_trans,current_radius

def remove_support_structure_intensity(las_path,lidar_skeleton,lidar_pcd,intensity_threshold,trunk_radius):
    las = laspy.read(las_path)
    intensity = las.intensity
    # Calculate the average intensity value of the entire point cloud
    intensity_mean_all = np.mean(intensity)
    points = np.asarray(lidar_skeleton.points)
    pcd_tree = o3d.geometry.KDTreeFlann(lidar_pcd)
    support_structure_points = []
    points_intensity_value = []
    # Compare the intensity value of each point with the average intensity of the entire point cloud
    for i in range(len(points)):
        [k, idx, d] = pcd_tree.search_radius_vector_3d(lidar_skeleton.points[i],trunk_radius)
        points_intensity = intensity[idx]
        intensity_mean = np.mean(points_intensity)
        points_intensity_value.append(intensity_mean)
        if intensity_mean>intensity_mean_all * intensity_threshold:#recommend:1.1-1.3
            support_structure_points.append(points[i])
    if len(support_structure_points) > 0:
        pcd_intensity = o3d.geometry.PointCloud()
        pcd_intensity.points = o3d.utility.Vector3dVector(np.asarray(support_structure_points))
        pcd_intensity.paint_uniform_color([1, 0.0, 0.0])
        return pcd_intensity,points_intensity_value
    elif len(support_structure_points) == 0:
        return None

def intensity_value_normalization(points_intensity_value, lidar_skeleton_pcd):
    cool_cmap = plt.get_cmap('cividis')
    min_intensity = np.min(points_intensity_value)
    max_intensity = np.max(points_intensity_value)
    # Create a color array based on normalized intensity values
    colors = cool_cmap((points_intensity_value - min_intensity) / (max_intensity - min_intensity))
    # Set the point cloud colors
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(lidar_skeleton_pcd.points))
    cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
    # Visualize the point cloud using Open3D
    o3d.visualization.draw_geometries([cloud], window_name="visualization_points",
                                      width=800, height=800, left=50, top=50,
                                      mesh_show_back_face=False)

def extract_two_main_branches(pcd, current_radius, trunk_diameter):
    outside_points,graph,_ = find_outside_points(pcd,current_radius,trunk_diameter)
    print("outside_points",outside_points)
    print("len(outside_points)", len(outside_points))
    z_points = sorted(np.asarray(pcd.points), key=(lambda z: z[2]))

    min_pts, max_pts = z_points[0], z_points[-1]
    max_path = get_path_points(min_pts, max_pts, graph)
    points = []
    # outside_points = find_outside_points(pcd, current_radius, trunk_diameter)
    for i in range(len(outside_points)):
        get_path = get_path_points(min_pts, np.asarray(outside_points)[i], graph)
        need_points = remove_points_not_need(get_path, max_path)
        points.append(need_points)
    get_points = []
    end_points = []
    start_points = []
    list_node_point = []
    for i in range(len(points)):
        if len(points[i]) > 0:
            end_points.append(points[i][-1])
            start_points.append(points[i][0])
            get_points.append(points[i])
    # Obtain the point clouds for the branch groups
    min_pts1 = np.amin(start_points, axis=0)[2]
    min_pts2 = np.asarray(start_points)[np.where(np.asarray(start_points)[:, 2] > min_pts1)]

    z_points = sorted(min_pts2, key=(lambda z: z[2]))[0]
    index1 = np.where(np.asarray(start_points)[:, 2] == min_pts1)
    index2 = np.where(np.asarray(start_points)[:, 2] == z_points[2])
    points_length = []
    for i in range(len(index2[0])):
        length = cultivate_length(get_points[index2[0][i]])
        points_length.append(length)
    path = np.asarray(get_points,dtype=object)[index2[0][np.where(points_length == np.max(points_length))[0]]]
    points_length1 = []
    for i in range(len(index1[0])):
        length1 = cultivate_length(get_points[index1[0][i]])
        points_length1.append(length1)
    path1 = np.asarray(get_points,dtype=object)[index1[0][np.where(points_length1 == np.max(points_length1))[0]]]
    trunk_branches_point = np.concatenate((max_path, path[0], path1[0]), axis=0)
    trunk_branches_list = [max_path, path[0], path1[0]]
    branch_point = find_nearest_point(max_path,path[0],graph)
    branch_point1 = find_nearest_point(max_path,path1[0],graph)
    list_node_point.append(branch_point)
    list_node_point.append(branch_point1)
    return trunk_branches_point, outside_points, max_path, trunk_branches_list, list_node_point

def combine_intensity_and_graph(pcd_remove, pcd_all, trunk_diameter, current_radius,log_func=print):
    if pcd_all != None:

        trunk_branches_point, _, max_path, _, _ = extract_two_main_branches(pcd_remove,current_radius,trunk_diameter,log_func=log_func)
        # Get the branches misidentified as support structures
        points_need = remove_points_not_need(pcd_all.points, trunk_branches_point)
        if len(points_need) > 0:
            points_list = []
            pcd_need = o3d.geometry.PointCloud()
            pcd_need.points = o3d.utility.Vector3dVector(np.asarray(points_need))
            # Use Euclidean distance clustering
            ec = cluster_points(pcd_need, radius=0.1)
            for i in range(len(ec)):
                ind = ec[i]
                clusters_cloud = pcd_need.select_by_index(ind)
                if len(clusters_cloud.points) > 20:
                    points_list.append(np.asarray(clusters_cloud.points))
            if len(points_list) > 0:
                concatenated = concatenate_points(points_list)
                points = remove_points_not_need(np.asarray(pcd_remove.points), concatenated)
                pcd_finall = o3d.geometry.PointCloud()
                pcd_finall.points = o3d.utility.Vector3dVector(np.asarray(points))
            elif len(points_list) == 0:
                pcd_finall = pcd_remove
        elif len(points_need) == 0:
            pcd_finall = pcd_remove
    else:
        pcd_finall = pcd_remove
    return pcd_finall, max_path

def smooth_skeleton(start_point, end_point, current_point, iterations):
    # Smooth step size
    step_size = 0.1
    for i in range(iterations):
        direction_adjustment = (start_point - current_point) / 2 + (end_point - current_point) / 2
        # Adjust the position of the current point
        current_point = current_point + step_size * direction_adjustment
    return current_point

def repair_local(pcd_start,pcd1,max_path,radius, current_radius,trunk_height):
    # Voxel downsampling
    pcd_down_sample = pcd1.voxel_down_sample(voxel_size = 0.03)
    pcd = clusters_to_pcd(pcd_start,radius,4)
    clusters = cluster_points(pcd,radius,min_cluster_size=4, max_cluster_size=100000)
     # Get each broken branch
    length = []
    for i in range(len(clusters)):
        length.append(len(clusters[i]))
    max_length = np.where(length==np.max(length))[0]
    clusters.remove(clusters[max_length[0]])
    pcd_max = o3d.geometry.PointCloud()
    pcd_max.points = o3d.utility.Vector3dVector(np.asarray(max_path))
    pcd_tree_max = o3d.geometry.KDTreeFlann(pcd_max)
    z_points = sorted(np.asarray(pcd_down_sample.points), key=(lambda z: z[2]))
    min_pts = sorted(z_points, key=(lambda z: z[2]))[0]
    nearest_points_list = []
    point_list = []
    # The nearest point to the most populous skeleton is the fracture point for each branch
    for i in range(len(clusters)):
        distance = []
        point = np.asarray(pcd.points)[clusters[i]]
        for j in range(len(point)):
            [k, idx, d] = pcd_tree_max.search_knn_vector_3d(point[j],1)
            distance.append(np.sqrt(d)[0])
        get_point = point[np.where(distance == np.amin(distance))][0]
        tree = cKDTree(point)
        point_index = tree.query_ball_point(get_point,0.02)
        point_need1 = point[np.where(point[point_index][:,2]>get_point[2])]
        point_need2 = point[np.where(point[point_index][:,2]<get_point[2])]
        if len(point_need1)==0 or len(point_need2)==0 :
            nearest_points_list.append(get_point)
            point_list.append(point)
    if len(nearest_points_list) > 0 :
        pcd_min_point = o3d.geometry.PointCloud()
        pcd_min_point.points = o3d.utility.Vector3dVector(np.asarray(nearest_points_list))
        # Select points from the downsampled point cloud for repair
        pcd_all = pcd_min_point + pcd_down_sample
        graph = make_graph(pcd_all,current_radius)
        points_path_all = []
        for i in range(len(nearest_points_list)):
            path = get_path_points(np.asarray(nearest_points_list)[i],min_pts,graph)
            point_middle = np.mean(point_list[i],axis=0)
            middle_dist = np.linalg.norm(np.array(point_middle)-np.array(path[1]))
            middle_dist1 = np.linalg.norm(np.array(point_middle)-np.array(path[2]))
            if middle_dist1 > middle_dist:
                points_remain = remove_points_not_need(np.asarray(pcd.points),np.asarray(point_list[i]))
                pcd_remove_points = o3d.geometry.PointCloud()
                pcd_remove_points.points = o3d.utility.Vector3dVector(np.asarray(points_remain))
                pcd_tree_remove = o3d.geometry.KDTreeFlann(pcd_remove_points)
                points_path = []
                for l in range(len(path)):
                    [k, idx, d] = pcd_tree_remove.search_knn_vector_3d(path[l],1)
                    nearest_point =  np.asarray(pcd_remove_points.points)[idx][0]
                    [k1, idx1, d1] = pcd_tree_max.search_knn_vector_3d(path[l],1)
                    [k2, idx2, d2] = pcd_tree_max.search_knn_vector_3d(nearest_point,1)
                    nearest_dist = np.sqrt(d1) - np.sqrt(d2)
                    if (np.sqrt(d) > 0.05)  &(path[l][2]>trunk_height):
                        points_path.append(path[l])
                    elif (np.sqrt(d) < 0.05) & (path[l][2] > nearest_point[2]) & (nearest_dist > 0) :
                        points_path.append(path[l])
                    else:
                        break
                if len(points_path)>0:
                    points_path_all.append(points_path)
            else:
                continue
        if len(points_path_all)>0:
            # Smooth the selected point cloud
            points_smooth = []
            for i in range(len(points_path_all)):
                points_smooth1 = []
                for j in range(len(points_path_all[i])):
                    if len(points_path_all[i]) > 2:
                            if j ==1:
                                p = smooth_skeleton(np.array(points_path_all[i][j-1]),np.array(points_path_all[i][j+1]),np.array(points_path_all[i][j]),10)
                                points_smooth.append(p)
                                points_smooth1.append(p)
                            elif j == len(points_path_all[i])-1:
                                break
                            elif j > 1:
                                p = smooth_skeleton(np.array(points_path_all[i][j-1]),np.array(points_path_all[i][j+1]),p,10)
                                points_smooth.append(p)
            if len(points_smooth)>0:
                pcd_smooth = o3d.geometry.PointCloud()
                pcd_smooth.points = o3d.utility.Vector3dVector(np.asarray(points_smooth))
                #Add the smoothed point cloud to the skeleton point cloud
                pcd_whole = pcd_smooth + pcd
            else:
                pcd_whole = pcd
        else:
            pcd_whole = pcd
    else:
        pcd_whole = pcd
    return pcd_whole

def repair_tree_dijkstra(pcd_whole, trunk_diameter):
    points_all = []
    z_points = sorted(np.asarray(pcd_whole.points), key=(lambda z: z[2]))
    min_pts = z_points[0]
    outside_points, graph, current_radius = find_outside_points(pcd_whole, None, trunk_diameter)
    pcd_whole2 = o3d.geometry.PointCloud()
    pcd_whole2.points = o3d.utility.Vector3dVector(np.asarray(outside_points))

    # Use linear interpolation to complete
    min_distance, max_distance, mean_distance = analyze_point_distances(pcd_whole)

    step = mean_distance
    for i in range(len(np.asarray(outside_points))):
        points_insert = []
        path = get_path_points(np.asarray(min_pts), np.asarray(outside_points)[i], graph)

        for j in range(len(path)):
            if j != 0:
                dist = np.linalg.norm(np.array(path[j]) - np.array(path[j - 1]))
                # If the distance is greater than mean_distance, then perform interpolation
                if dist >= mean_distance:
                    number = max(1, int(np.ceil(dist / step)))
                    for k in range(1, number):

                        t = k / number
                        point_need = np.asarray(path[j - 1]) * (1 - t) + np.asarray(path[j]) * t
                        points_insert.append(point_need)

        if len(points_insert) > 0:
            points_all.append(points_insert)
        if len(path) > 0:
            points_all.append(path)


    concatenated = concatenate_points(points_all)
    points_concatenate_remove = np.unique(concatenated, axis=0)
    # Obtain the completed point cloud
    pcd_finall_repair = o3d.geometry.PointCloud()
    pcd_finall_repair.points = o3d.utility.Vector3dVector(np.asarray(points_concatenate_remove))
    pcd_finall_repair.paint_uniform_color([1, 0.0, 0.0])
    return pcd_finall_repair, mean_distance

def get_main_branch(pcd, mean_distance, current_radius, trunk_diameter):
    branch_path_points = []
    pcd_all = o3d.geometry.PointCloud()
    pcd_all.points = o3d.utility.Vector3dVector(np.asarray(pcd.points))
    trunk_points,outside_need,_,trunk_branches_path,branch_nodes = extract_two_main_branches(pcd,None, trunk_diameter)
    points_need = remove_points_not_need(np.asarray(pcd_all.points),trunk_points)
    pcd_remove_trunk = o3d.geometry.PointCloud()
    pcd_remove_trunk.points = o3d.utility.Vector3dVector(np.asarray(points_need))
    pcd_trunk = o3d.geometry.PointCloud()
    pcd_trunk.points = o3d.utility.Vector3dVector(np.asarray(trunk_points))
    pcd_tree = o3d.geometry.KDTreeFlann(pcd_trunk)
    graph = make_graph(pcd_remove_trunk,current_radius)
    # Cluster the remaining parts of the pear tree point cloud
    clusters = cluster_points(pcd_remove_trunk,mean_distance)
    for j in range(len(clusters)):
        points_distance = []
        points_path = []
        outside_point = []
        clusters_cloud = pcd_remove_trunk.select_by_index(clusters[j])
        if len(clusters_cloud.points)>1:
            points_trunk = np.asarray(clusters_cloud.points)
            for k in range(len(points_trunk)):
                if (np.round(points_trunk[k],4) == np.round(outside_need,4)).all(1).any():
                    outside_point.append(points_trunk[k])
            # Only one outermost point as a first-level branch
            if len(outside_point) == 1:
                nearest_point = find_nearest_point(points_trunk,trunk_points,graph)
                branch_path = get_path_points(nearest_point,outside_point[0],graph)
                [_, idx, _] = pcd_tree.search_knn_vector_3d(nearest_point, 1)
                node_point = np.asarray(trunk_points)[idx[0]]
                if len(branch_path)>30:
                    branch_path_points.append(branch_path)
                    branch_nodes.append(node_point)
            # The longest branch with multiple outermost points as a first-level branch
            elif len(outside_point) > 1:
                nearest_point1 = find_nearest_point(points_trunk,trunk_points,graph)
                [_, idx1, _] = pcd_tree.search_knn_vector_3d(nearest_point1, 1)
                node_point1 = np.asarray(trunk_points)[idx1[0]]
                for i in range(len(outside_point)):
                    path = get_path_points(np.asarray(nearest_point1),np.asarray(outside_point[i]),graph)
                    points_path.append(path)
                    dist = cultivate_length(path)
                    points_distance.append(dist)
                index = np.where(points_distance == np.amax(points_distance))[0]
                path_max = points_path[index[0]]
                if len(path_max)>30:
                    branch_path_points.append(path_max)
                    branch_nodes.append(node_point1)
    branch_number = len(branch_path_points) + 3
    print(branch_number)
    tree_points = np.concatenate((branch_path_points, trunk_branches_path), axis=0)
    return branch_nodes,branch_number,tree_points,outside_point


