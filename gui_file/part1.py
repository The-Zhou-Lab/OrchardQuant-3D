import os
import shapefile
import geopandas as gpd
import laspy
from scipy import spatial
import whitebox
import CSF
from osgeo import osr, ogr
import rasterio
from skimage import io
import numpy as np
from matplotlib import pyplot as plt


# Region of interest extraction
def make_polygon(rtk_shapefile_path, image_file_path=None, epsg=None):
    if image_file_path is not None:
        # Retrieve EPSG from Image File
        with rasterio.open(image_file_path) as src:
            epsg = src.crs.to_epsg()
    elif epsg is None:
        raise ValueError("Either 'image_file_path' or 'epsg' must be provided.")

    # Read RTK File
    sf = shapefile.Reader(rtk_shapefile_path)
    shapes = sf.shapes()
    rtk_points = [shape.points[0] for shape in shapes]

    rtk_dir, rtk_name = os.path.split(rtk_shapefile_path)
    polygon_shapefile_path = os.path.join(rtk_dir, 'polygon.shp')

    # Create Polygon Shapefile
    driver = ogr.GetDriverByName("ESRI Shapefile")
    data_source = driver.CreateDataSource(polygon_shapefile_path)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    layer = data_source.CreateLayer("polygon", srs, ogr.wkbMultiPolygon)

    # Read the corresponding feature type from the layer and create a feature
    feature = ogr.Feature(layer.GetLayerDefn())
    wkt = 'POLYGON (({} {},{} {},{} {},{} {},{} {}))'.format(
        rtk_points[0][0], rtk_points[0][1],
        rtk_points[1][0], rtk_points[1][1],
        rtk_points[2][0], rtk_points[2][1],
        rtk_points[3][0], rtk_points[3][1],
        rtk_points[0][0], rtk_points[0][1])
    polygon = ogr.CreateGeometryFromWkt(wkt)
    feature.SetGeometry(polygon)
    layer.CreateFeature(feature)

    feature = None
    data_source = None

    # Visualize ROI Points and Polygons
    sf_m = gpd.read_file(rtk_shapefile_path)
    sf_po = gpd.read_file(polygon_shapefile_path)

    fig, axes = plt.subplots(1, 2, figsize=(10, 10))
    ax = axes.ravel()
    sf_m.plot(color='red', markersize=50, ax=ax[0])
    ax[0].set_title('ROI Points')
    sf_po.boundary.plot(color="red", zorder=10, ax=ax[1])
    ax[1].set_title('ROI Polygon')
    plt.tight_layout()
    plt.show()

    return polygon_shapefile_path, rtk_points


# point cloud denosing(Statistical Outlier Removal，SOR)
def denoising(input_las_file_path, polygon_shapefile_path):
    las_dir, las_name = os.path.split(input_las_file_path)
    roi_las_name = "roi_" + las_name
    roi_las_path = os.path.join(las_dir, roi_las_name)

    wbt = whitebox.WhiteboxTools()
    wbt.clip_lidar_to_polygon(
        i=input_las_file_path,
        polygons=polygon_shapefile_path,
        output=roi_las_path
    )

    # Read and Process Point Cloud Data in Chunks
    sigma = 5
    K = 20
    chunk_size = 5000000
    k_dist = []
    with laspy.open(roi_las_path) as las:
        num_points = las.header.point_count
        total_chunks = (num_points + chunk_size - 1) // chunk_size
        for i in range(0, num_points, chunk_size):
            chunk_number = i // chunk_size + 1
            print(f"Processing chunk {chunk_number} / {total_chunks}")
            chunk = las.read_points(chunk_size)
            x, y, z = chunk.x, chunk.y, chunk.z
            lasdata = np.vstack((x, y, z)).transpose()

            tree = spatial.KDTree(lasdata)
            dist, _ = tree.query(lasdata, K, workers=4)
            k_dist.extend(np.sum(dist, axis=1))

    # Determine the Maximum Threshold for Noise
    max_distance = np.mean(k_dist) + sigma * np.std(k_dist)

    # Index of Noise
    outer_index = np.where(np.array(k_dist) > max_distance)
    sor_filter = np.array(k_dist) <= max_distance

    denoised_las_file_path = os.path.join(las_dir, 'sor_' + roi_las_name)
    with laspy.open(roi_las_path) as las:
        points = las.read_points(las.header.point_count)
        denoised_points = points[sor_filter]
        with laspy.open(denoised_las_file_path, mode='w', header=las.header) as out_las:
            out_las.write_points(denoised_points)

    return denoised_las_file_path, roi_las_name


# Ground Filtering
def ground_filtering(denoised_las_file_path, resolution, threshold):
    las_dir, sor_name = os.path.split(denoised_las_file_path)
    with laspy.open(denoised_las_file_path) as las_sor:
        points = las_sor.read_points(las_sor.header.point_count)
        xyz_sor = np.vstack((points.x, points.y, points.z)).transpose()

    csf = CSF.CSF()
    csf.params.bSloopSmooth = False
    csf.params.cloth_resolution = resolution
    csf.params.class_threshold = threshold
    csf.setPointCloud(xyz_sor)
    ground = CSF.VecInt()
    non_ground = CSF.VecInt()
    csf.do_filtering(ground, non_ground)

    def save_points(points, file_name):
        file_path = os.path.join(las_dir, file_name)
        with laspy.open(denoised_las_file_path) as las_sor:
            point_data = las_sor.read_points(las_sor.header.point_count)
            filtered_points = point_data[list(points)]
            with laspy.open(file_path, mode='w', header=las_sor.header) as out_las:
                out_las.write_points(filtered_points)
        return file_path

    ground_points_file_path = save_points(ground, 'ground_' + sor_name)
    above_ground_points_file_path = save_points(non_ground, 'above_ground_' + sor_name)

    return ground_points_file_path,above_ground_points_file_path


# Generate Canopy Height Model (CHM)
def make_chm(ground_points_file_path, denoised_las_file_path, las_name, polygon_shapefile_path):
    wbt = whitebox.WhiteboxTools()
    las_dir, _ = os.path.split(denoised_las_file_path)

    with laspy.open(ground_points_file_path) as las_ground:
        ground_points = las_ground.read_points(las_ground.header.point_count)
        z_ground = ground_points.z

    above_ground_las_path = os.path.join(las_dir, 'sor_above_' + las_name)
    # point cloud slice
    wbt.lidar_elevation_slice(
        i=denoised_las_file_path,
        output=above_ground_las_path,
        minz=z_ground.min(),
        maxz=None,
        cls=False
    )

    above_ground_tif_path = os.path.join(las_dir, 'sor_above_' + las_name[:-4] + '.tif')
    # point cloud tin griding
    wbt.lidar_tin_gridding(i=above_ground_las_path,
                           output=above_ground_tif_path,
                           parameter="elevation",
                           returns="all",
                           resolution=0.01,
                           minz=None,
                           maxz=None,
                           max_triangle_edge_length=None)
    ground_tif_path = os.path.join(las_dir, 'ground_' + las_name[:-4] + '.tif')
    wbt.lidar_tin_gridding(
        i=ground_points_file_path,
        output=ground_tif_path,
        parameter="elevation",
        returns="all",
        resolution=0.01,
        minz=None,
        maxz=None,
        max_triangle_edge_length=None
    )
    # make polygon
    wbt.clip_raster_to_polygon(
        i=above_ground_tif_path,
        polygons=polygon_shapefile_path,
        output=above_ground_tif_path
    )
    wbt.clip_raster_to_polygon(
        i=ground_tif_path,
        polygons=polygon_shapefile_path,
        output=ground_tif_path
    )

    with rasterio.open(ground_tif_path) as src:
        dem_im = src.read(1, masked=True)
        print(f"DEM: {src.meta}")

    with rasterio.open(above_ground_tif_path) as src:
        dsm_im = src.read(1, masked=True)
        dsm_meta = src.profile
        print(f"DSM: {src.meta}")
    # make CHM
    chm = dsm_im - dem_im
    nodatavalue = chm.min()
    chm_fi = np.ma.filled(chm, fill_value=nodatavalue)

    chm_meta = dsm_meta.copy()
    chm_meta.update({'nodata': nodatavalue})

    canopy_height_model_file_path = os.path.join(las_dir, 'chm_' + las_name[:-4] + '.tif')
    with rasterio.open(canopy_height_model_file_path, 'w', **chm_meta) as ff:
        ff.write(chm_fi, 1)

    return canopy_height_model_file_path


# CHM Visualization
def visualize_chm(canopy_height_model_file_path):
    img = io.imread(canopy_height_model_file_path)

    fig, axes = plt.subplots(1, 2, figsize=(10, 10))
    axes[0].imshow(img, cmap=plt.cm.gray)
    axes[0].set_title('CHM (Grayscale)')
    axes[1].imshow(img, cmap=plt.cm.jet)
    axes[1].set_title('CHM (Jet)')
    plt.tight_layout()