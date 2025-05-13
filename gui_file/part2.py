# Import Necessary Libraries
import shapefile
import rasterio
import laspy
import cv2
import math
import numpy as np
from rasterio.control import GroundControlPoint as GCP
from rasterio.transform import from_gcps
import matplotlib.pyplot as plt
import pandas as pd
import whitebox
import os
import numpy as np
from scipy import ndimage
from sklearn.cluster import KMeans
from skimage import io, morphology, measure, color, filters
from sklearn.neighbors import KDTree
from scipy import spatial
import re

# Clear KMeans warning
os.environ['OMP_NUM_THREADS'] = '1'

# CHM Alignment
def transform_chm(chm_path, rtk_points_path):
    with rasterio.open(chm_path) as chm:
        profile = chm.profile
        chm_dir, chm_name = os.path.split(chm_path)

        sf = shapefile.Reader(rtk_points_path, 'rb')
        shapes = sf.shapes()
        rtk_points = [shape.points[0] for shape in shapes]
        # Retrieves the pixel coordinates of CHM from geographic coordinates
        pixel_indices = []
        for x, y in rtk_points:
            row, col = chm.index(x, y)
            pixel_indices.append((row, col))
        print(pixel_indices)

        ul_row, ul_col = pixel_indices[0]
        ll_row, ll_col = pixel_indices[1]
        lr_row, lr_col = pixel_indices[2]
        ur_row, ur_col = pixel_indices[3]

        num_rows = max(
            np.int64(math.hypot(ul_row - ll_row, ul_col - ll_col)),
            np.int64(math.hypot(ur_row - lr_row, ur_col - lr_col))
        )
        num_cols = max(
            np.int64(math.hypot(ul_row - ur_row, ul_col - ur_col)),
            np.int64(math.hypot(ll_row - lr_row, ll_col - lr_col))
        )

        print(f"num_rows: {num_rows}, num_cols: {num_cols}")
        print(f"RTK points: {rtk_points}")
        print(f"Pixel indices: {pixel_indices}")
        # make Ground Control Points (GCPs)
        gcps = [
            GCP(0, 0, *rtk_points[0]),
            GCP(0, num_cols, *rtk_points[3]),
            GCP(num_rows, 0, *rtk_points[1]),
            GCP(num_rows, num_cols, *rtk_points[2])
        ]
        # Generate affine transformation matrix from GCPs
        transform = from_gcps(gcps)
        print(f"Transform: {transform}")

        chm_array = chm.read(1)
        print(f"CHM array shape: {chm_array.shape}")

        src_points = np.float32([
            [ul_col, ul_row], [ur_col, ur_row],
            [ll_col, ll_row], [lr_col, lr_row]
        ])
        dst_points = np.float32([
            [0, 0], [num_cols, 0],
            [0, num_rows], [num_cols, num_rows]
        ])
        # Calculate perspective transformation matrix
        perspective_transform = cv2.getPerspectiveTransform(src_points, dst_points)
        print(f"Perspective transform shape: {perspective_transform.shape}")

        # Apply perspective transformation to CHM
        warped_chm = cv2.warpPerspective(chm_array, perspective_transform, (num_cols, num_rows))
        print(f"Warped CHM shape: {warped_chm.shape}")

        profile.update(width=num_cols, height=num_rows, transform=transform)

        transformed_path = os.path.join(chm_dir, "transformed_chm.tif")
        with rasterio.open(transformed_path, 'w', **profile) as dst:
            dst.write(warped_chm, 1)

    return transformed_path

def extract_bbox(image_path):
    # Read the image
    image = io.imread(image_path)

    # Binarization using Otsu's threshold method
    thresh = filters.threshold_otsu(image)
    binary = image > thresh

    # Morphological closing to close canopy areas
    closed = ndimage.binary_closing(
        binary, structure=morphology.disk(5), iterations=6)

    # Remove small isolated noise points
    cleaned = morphology.remove_small_objects(closed, min_size=100)

    # Connected component analysis
    labels = measure.label(cleaned, connectivity=2)

    # Label connected components and display bounding boxes
    rgb_labels = color.label2rgb(labels, bg_label=0)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(binary)

    centroid_list = []
    bbox_list = []
    for region in measure.regionprops(labels):
        # Draw bounding box
        minr, minc, maxr, maxc = region.bbox
        rect = plt.Rectangle((minc, minr), maxc - minc, maxr - minr,
                             fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)
        label = region.label
        ax.text(minc, minr, str(label), fontsize=12, color='yellow',
                ha='center', va='center')

        # Get centroid coordinates and bounding box coordinates
        centroid = region.centroid
        bbox = region.bbox
        centroid_list.append(centroid)
        bbox_list.append(bbox)

    ax.set_axis_off()
    plt.tight_layout()

    # Save result image to IMG folder
    output_dir = os.path.join(os.path.dirname(image_path), 'IMG')
    os.makedirs(output_dir, exist_ok=True)

    # Save the binary image
    binary_path = os.path.join(output_dir, 'binary.png')
    io.imsave(binary_path, binary.astype(np.uint8) * 255)

    # Save the image after closing operation
    closed_path = os.path.join(output_dir, 'closed.png')
    io.imsave(closed_path, closed.astype(np.uint8) * 255)

    # Save the image after removing noise
    cleaned_path = os.path.join(output_dir, 'cleaned.png')
    io.imsave(cleaned_path, cleaned.astype(np.uint8) * 255)

    # Save connected component analysis image
    labels_path = os.path.join(output_dir, 'rgb_labels.png')
    io.imsave(labels_path, (rgb_labels * 255).astype(np.uint8))

    # Save the image with labeled connected components
    result_path = os.path.join(output_dir, 'result.png')
    plt.savefig(result_path, dpi=300, bbox_inches='tight')
    # Generate image of connected component centroids on a white background with black dots

    plt.close(fig)

    return centroid_list, bbox_list,binary

def create_single_tree_shapefile(tree_points_geo, shapefile_path):
    with shapefile.Writer(shapefile_path, shapeType=shapefile.POLYGON) as w:
        w.field('name', 'C')
        w.poly([tree_points_geo])
        w.record('polygon')


# generate trees bouning box
def generate_tree_shapefiles(border_list, chm_path):
    chm = rasterio.open(chm_path)
    geo_pixels = []
    shapefile_paths = []
    col_geo_pixels = []
    for tree_index, tree in enumerate(border_list):
        if len(tree) == 4:
                # Convert the bounding box tuple into a list of coordinate points, and be aware that the direction of the generated .shp file,
                # whether clockwise or counter-clockwise, affects the results of subsequent segmentation. The code is in counter-clockwise direction
                # tree_points = [[tree[0], tree[1]], [tree[2], tree[1]], [tree[2], tree[3]], [tree[0], tree[3]]]
            tree_points = [[tree[0], tree[1]], [tree[0], tree[3]], [tree[2], tree[3]], [tree[2], tree[1]]]
            tree_points_geo = [rasterio.transform.xy(chm.transform, point[0], point[1], offset='center')
                                   for point in tree_points]
            col_geo_pixels.append(tree_points_geo)
            shapefile_name = f"polygon_{tree_index}.shp"
            shapefile_dir = os.path.join(os.path.dirname(chm_path), "Result_dir", "single_tree_shapefiles")
            os.makedirs(shapefile_dir, exist_ok=True)
            shapefile_path = os.path.join(shapefile_dir, shapefile_name)
            shapefile_paths.append(shapefile_path)
            create_single_tree_shapefile(tree_points_geo, shapefile_path)
    geo_pixels.append(col_geo_pixels)
    return geo_pixels, shapefile_paths

def segment_single_tree(las_path, shapefile_paths):
    las_dir, _ = os.path.split(las_path)
    output_dir = os.path.join(las_dir, "Result_dir", "origin_single_tree_las")
    os.makedirs(output_dir, exist_ok=True)

    wbt = whitebox.WhiteboxTools()
    wbt.set_verbose_mode(False)

    for shapefile_path in shapefile_paths:
        # Extract the sequence number from the filename
        file_name = os.path.basename(shapefile_path)
        tree_id = file_name.split('_')[1][:-4]

        output_las_name = f"{tree_id}.las"
        output_las_path = os.path.join(output_dir, output_las_name)

        wbt.clip_lidar_to_polygon(
            i=las_path,
            polygons=shapefile_path,
            output=output_las_path
        )

def denoise_pear_tree(input_folder, output_folder, sigma=1, K=10):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # Iterate over all LAS files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".las") or filename.endswith(".LAS"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            las = laspy.read(input_path)
            x, y, z = las["X"], las["Y"], las["Z"]
            lasdata = (np.stack((x, y, z), axis=0)).transpose()
            tree = spatial.KDTree(lasdata)

            k_dist = np.zeros_like(x)
            for i in range(len(x)):
                dist, index = tree.query(np.array([x[i], y[i], z[i]]), K)
                k_dist[i] = np.sum(dist)

            # Determine the maximum threshold of noise
            max_distance = np.mean(k_dist) + sigma * np.std(k_dist)

            # Index of noise
            outer_index = np.where(k_dist > max_distance)
            # print(f'Outer points index array for {filename}: {outer_index}')

            sor_filter = k_dist <= max_distance

            out_las = laspy.LasData(las.header)
            out_las.points = las.points[sor_filter]
            out_las.write(output_path)

def process(lidar_chm_path,lidar_rtk_path,lidar_las_path,lidar_single_tree_folder,lidar_denosied_tree_folder):

    lidar_transformed_chm_path = transform_chm(lidar_chm_path, lidar_rtk_path)
    lidar_centroid_list, lidar_bbox_list, lidar_binary = extract_bbox(lidar_transformed_chm_path)
    lidar_geo_pixels, lidar_shapefile_paths = generate_tree_shapefiles(lidar_bbox_list, lidar_transformed_chm_path)
    segment_single_tree(lidar_las_path, lidar_shapefile_paths)
    denoise_pear_tree(lidar_single_tree_folder, lidar_denosied_tree_folder, sigma=5, K=10)

def project_and_calcu_center(centroid_path, centroid_list, num_clusters_x, num_clusters_y):
    # Project centroid points onto the x and y axes
    x_coords = np.array([point[0] for point in centroid_list])
    y_coords = np.array([point[1] for point in centroid_list])

    # Cluster coordinates separately on the x and y axes
    kmeans_x = KMeans(n_clusters=num_clusters_x, n_init='auto')
    kmeans_y = KMeans(n_clusters=num_clusters_y, n_init='auto')
    x_clusters = kmeans_x.fit_predict(x_coords.reshape(-1, 1))
    y_clusters = kmeans_y.fit_predict(y_coords.reshape(-1, 1))

    # Calculate the center of each cluster on the x and y axes
    x_cluster_centers = []
    y_cluster_centers = []
    for cluster_id in range(num_clusters_x):
        x_cluster_points = x_coords[x_clusters == cluster_id]
        x_cluster_center = np.mean(x_cluster_points)
        x_cluster_centers.append(x_cluster_center)
    for cluster_id in range(num_clusters_y):
        y_cluster_points = y_coords[y_clusters == cluster_id]
        y_cluster_center = np.mean(y_cluster_points)
        y_cluster_centers.append(y_cluster_center)

    x_cluster_centers.sort()
    y_cluster_centers.sort()

    # Calculate midpoints between adjacent clusters on the x and y axes
    x_midpoints = []
    y_midpoints = []
    for i in range(num_clusters_x - 1):
        x_midpoint = (x_cluster_centers[i] + x_cluster_centers[i + 1]) / 2
        x_midpoints.append(x_midpoint)
    for i in range(num_clusters_y - 1):
        y_midpoint = (y_cluster_centers[i] + y_cluster_centers[i + 1]) / 2
        y_midpoints.append(y_midpoint)

    # Add image boundaries as midpoints
    lidar_x, lidar_y = io.imread(centroid_path).shape
    x_midpoints.extend([0, lidar_x])
    x_midpoints.sort()
    y_midpoints.extend([0, lidar_y])
    y_midpoints.sort()

    return x_cluster_centers, x_midpoints, y_cluster_centers, y_midpoints

def sort_key(name):
    # Extract the letter and number parts
    match = re.match(r'([a-zA-Z]*)(\d*)', name)
    if match:
        letters, number = match.groups()
        return (letters, int(number) if number else 0)
    return (name, 0)

def get_centroid(las_path):
    # las_dir, _ = os.path.split(las_path)
    centroid_list = []
    # output_dir = os.path.join(las_dir, "Result_dir", "denoised_single_tree_las")
    base_names = [os.path.splitext(name)[0] for name in os.listdir(las_path) if name.endswith('.las')]
    sorted_base_names = sorted(base_names, key=sort_key)
    output_dir_las = [os.path.join(las_path, f"{i}.las") for i in sorted_base_names]
    for las_path in output_dir_las:
        las = laspy.read(las_path)
        x, y, z = las.x, las.y, las.z
        # Calculate the threshold for the lowest 10%.
        threshold = np.percentile(z, 10)
        # print("Threshold:", threshold)
        # Find the indices in z that are below the threshold.
        low_z_indices = z <= threshold
        # Extract the corresponding x, y, z portions
        x_low = x[low_z_indices]
        y_low = y[low_z_indices]
        tree_centroid = np.array([np.mean(x_low), np.mean(y_low)])
        centroid_list.append(tree_centroid)
    return centroid_list,output_dir_las

def read_point(txt_path):
    with open(txt_path) as file:
        content = file.readlines()
        pickle_point = []
        for line in content:
            items = line.strip().split(',')[1:-1]
            pickle_point.append([float(items[0]), float(items[1])])
    return pickle_point

def find_pixel_coordinate(img_path, pickle_point):
    chm = rasterio.open(img_path)
    coordinate_list = []
    for i in pickle_point:
        coordinates = chm.index(
                        x=i[0],
                        y=i[1])
        coordinate_list.append(coordinates)
    return coordinate_list

def get_transform_matrix(lidar_points, uav_points):
    # Calculate the transformation matrix
    src_points = np.float32([[x, y] for x,y in lidar_points])
    dst_points = np.float32([[x, y] for x,y in uav_points])
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    return M

def centroid_sorted(ver_list, centroid_list, tree_centroid_list):
    sorted_centroid = []
    sorted_border = []
    combined = zip(centroid_list, tree_centroid_list)
    for i in range(len(ver_list) - 1):
        threshold = [ver_list[i], ver_list[i + 1]]
        filtered_data = [[centroid, border] for centroid, border in zip(centroid_list, tree_centroid_list)
                         if (centroid[1] < threshold[1] and centroid[1] > threshold[0])]
        sorted_data = sorted(filtered_data, key=lambda x: x[0][0], reverse=True)
        sorted_centroid_list, sorted_tree_centroid_list = list(zip(*sorted_data))
        sorted_centroid.append(list(sorted_centroid_list))
        sorted_border.append(list(sorted_tree_centroid_list))
    return sorted_centroid, sorted_border

def rename_file(before_sorted_points, after_sorted_points, output_dir_las):
    las_file,_ = os.path.split(output_dir_las[0])
    file_list = os.listdir(las_file)
    if not any(filename.startswith('a') for filename in file_list):
        for col_index, col in enumerate(after_sorted_points):
            for tree_index, tree in enumerate(col):
                index_need = np.where(np.all(before_sorted_points == tree, axis=1))[0]
                old_name = output_dir_las[index_need[0]]
                name = f"{chr(97 + col_index)}{tree_index+1}.las"
                new_name = os.path.join(las_file,name)
                os.rename(old_name,new_name)
    return None

def draw_centroid(img,centroid_list,output_dir):
    centroid_img = np.zeros_like(img, dtype=np.uint8)
    centroid_img.fill(255)  # Set background to white
    # Set the radius of centroid points
    radius = int(min(centroid_img.shape[:2]) * 0.005)
    for centroid in centroid_list:
        y, x = int(centroid[0]), int(centroid[1])
        # Draw circular centroid points using cv2.circle() function
        cv2.circle(centroid_img, (x, y), radius, 0, -1)
    # Save the image of connected component centroids
    centroid_path = os.path.join(output_dir, 'centroid.png')
    io.imsave(centroid_path, centroid_img)
    return centroid_img

def if_exist(img, ver_list, hor_list):
    x = ver_list
    y = hor_list[::-1]
    # Extract coordinates for each box
    all_cols = []
    for i in range(len(x)-1):
        single_col = []
        first_x = x[i]
        sec_x = x[i+1]
        for j in range(len(y)-1):
            first_y = y[j]
            sec_y = y[j+1]
            square = [[first_x, first_y], [sec_x, first_y], [sec_x, sec_y], [first_x, sec_y]]
            single_col.append(square)
        all_cols.append(single_col)
    # Check if a tree exists in each box
    all_result = []
    for col in all_cols:
        col_result = []
        for square in col:
            # Top-left coordinate
            x1, y1 = square[3]
            # Bottom-right coordinate
            x2, y2 = square[1]
            # Get elements within the rectangular box
            matrix = img[int(y1):int(y2)+1, int(x1):int(x2)+1]
            # Check if elements in the matrix are 0
            result = np.any(255 - matrix)
            col_result.append(result)
        all_result.append(col_result)
    return all_result, all_cols

def compare_list(lidar_exist, uav_exist):
    # Compare two lists
    # 1. Equal: Both have marking 1, both don't have marking 2
    # 2. Not equal: Drone has it but radar doesn't, indicating drone has extra, mark 3;
    #Drone doesn't have it but radar does, indicating drone is missing it, mark -1
    marked = np.zeros_like(uav_exist)
    marked = marked.astype(int)
    lidar_centroid_coordnites = []
    uav_centroid_coordnites = []
    for i in range(len(uav_exist)):
        for j in range(len(uav_exist[i])):
            if uav_exist[i][j] == lidar_exist[i][j] and lidar_exist[i][j] == True:
                marked[i][j] = 1
                lidar_centroid_coordnites.append(lidar_exist[i][j])
                uav_centroid_coordnites.append(uav_exist[i][j])
            elif uav_exist[i][j] == lidar_exist[i][j] and lidar_exist[i][j] == False:
                marked[i][j] = 2
            elif uav_exist[i][j] == True and lidar_exist[i][j] == False:
                marked[i][j] = 3
            else:
                marked[i][j] = -1
    return marked,lidar_centroid_coordnites,uav_centroid_coordnites


def modified_centroid_sorted(lidar_centroid_list,uav_centroid_list, marked, M):
    # 1: Both have; 2: Neither has; 3: Drone has extra; -1: Drone is missing
    # Fill in missing centroid points in the sorted centroid list, focusing on the missing parts
    miss_points_find = []
    miss_position = []
    uav = uav_centroid_list[:]
    lidar = lidar_centroid_list[:]
    for i in range(len(marked)):
        for j in range(len(marked[i])):
            if marked[i][j] == 2:
                uav[i].insert(j, ["Both don't have"])
            elif marked[i][j] == -1:
#                 print("i:{}, j:{}".format(i, j))
                print("i: {} out of {} range".format(i, len(lidar)))
                print("j: {} out of {} range".format(j, len(lidar[i])))
                miss_position.append([i,j])
                miss_point = lidar[i][j]
                uav_miss_point = np.dot(M, [miss_point[0], miss_point[1], 1])
                miss_points_find.append((uav_miss_point[0],uav_miss_point[1]))
                print("uav_miss_point",uav_miss_point)
                print("miss_position",miss_position)
    return miss_points_find,miss_position

def search_points(marked,lidar_txt,uav_txt,lidar_transformed,uav_transformed,lidar_sorted_centroid,uav_sorted_centroid,uav_centroids):
    if np.any(marked == -1):
        lidar_pickle_point = read_point(lidar_txt)
        uav_dormancy_pickle_point = read_point(uav_txt)
        # Find the corresponding pixel coordinates
        lidar_pixel_list = find_pixel_coordinate(lidar_transformed, lidar_pickle_point)
        uav_dormancy_pixel_list = find_pixel_coordinate(uav_transformed, uav_dormancy_pickle_point)
        # Calculate the transformation matrix
        M = get_transform_matrix(lidar_pixel_list, uav_dormancy_pixel_list)
        miss_points_find,miss_position = modified_centroid_sorted(lidar_sorted_centroid,uav_sorted_centroid, marked, M)
        chm = rasterio.open(uav_transformed)
        tree_points_geo = [rasterio.transform.xy(chm.transform, point[0], point[1], offset='center') for point in miss_points_find]
        print("tree_points_geo", tree_points_geo)
        for i, (row, col) in enumerate(miss_position):
            uav_centroids[row].insert(col, np.array(tree_points_geo[i]))
        return uav_centroids


def process_tree_sorted(uav_dormancy_las_path,uav_dormancy_transformed_chm_path,lidar_las_path,lidar_transformed_chm_path):
    # Hyperplane
    # Uav Dormancy Stage
    uav_dormancy_centroid_list, uav_dormancy_bbox_list, uav_dormancy_binary = extract_bbox(
            uav_dormancy_transformed_chm_path)
    uav_dormancy_centroids, uav_dormancy_output_dir_las = get_centroid(uav_dormancy_las_path)
    uav_dormancy_coordinate_list = find_pixel_coordinate(uav_dormancy_transformed_chm_path, uav_dormancy_centroids)
    uav_dormancy_x_cluster_centers, uav_dormancy_x_midpoints, uav_dormancy_y_cluster_centers, uav_dormancy_y_midpoints = project_and_calcu_center(
        uav_dormancy_transformed_chm_path, uav_dormancy_coordinate_list, 8, 9)
    uav_dormancy_sorted_centroid, uav_dormancy_sorted_tree_centroids = centroid_sorted(uav_dormancy_y_midpoints,
                                                                                       uav_dormancy_coordinate_list,
                                                                                       uav_dormancy_centroids)
    uav_dormancy_output_dir, _ = os.path.split(uav_dormancy_transformed_chm_path)
    uav_dormancy_centroid_img = draw_centroid(uav_dormancy_binary, uav_dormancy_coordinate_list,
                                              uav_dormancy_output_dir)
    uav_dormancy_exists, uav_dormancy_cols = if_exist(uav_dormancy_centroid_img, uav_dormancy_y_midpoints,
                                                      uav_dormancy_x_midpoints)
    # rename
    rename_file(uav_dormancy_centroids, uav_dormancy_sorted_tree_centroids, uav_dormancy_output_dir_las)


    # BBox Detection
    lidar_centroid_list, lidar_bbox_list, lidar_binary = extract_bbox(lidar_transformed_chm_path)

    # Lidar Dormancy Stage
    lidar_tree_centroids, lidar_output_dir_las = get_centroid(lidar_las_path)
    lidar_coordinate_list = find_pixel_coordinate(lidar_transformed_chm_path, lidar_tree_centroids)
    lidar_x_cluster_centers, lidar_x_midpoints, lidar_y_cluster_centers, lidar_y_midpoints = project_and_calcu_center(
        lidar_transformed_chm_path, lidar_coordinate_list, 8, 9)
    lidar_sorted_centroid, lidar_sorted_tree_centroids = centroid_sorted(lidar_y_midpoints, lidar_coordinate_list,
                                                                         lidar_tree_centroids)
    lidar_output_dir, _ = os.path.split(lidar_transformed_chm_path)
    lidar_centroid_img = draw_centroid(lidar_binary, lidar_coordinate_list, lidar_output_dir)
    lidar_exists, lidar_cols = if_exist(lidar_centroid_img, lidar_y_midpoints, lidar_x_midpoints)
    marked, lidar_centroid_coordnites, uav_centroid_coordnites = compare_list(lidar_exists, uav_dormancy_exists)

    # rename
    rename_file(lidar_tree_centroids, lidar_sorted_tree_centroids, lidar_output_dir_las)
    #search_points
    lidar_txt = r"D:\data_lidar.txt"
    uav_txt = r"D:\data_dormancy_uav.txt"
    uav_dormancy_centroids = search_points(marked, lidar_txt, uav_txt, lidar_transformed_chm_path,
                                           uav_dormancy_transformed_chm_path, lidar_sorted_centroid,
                                           uav_dormancy_sorted_centroid, uav_dormancy_sorted_tree_centroids)
