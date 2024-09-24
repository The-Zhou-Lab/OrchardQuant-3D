from Skeleton_utils import process_point_clouds

def main():
    
    lidar_pcd_folder = r"E:\Pear\Lidar\Dormancy Stage\Result_dir\pcd"
    lidar_result_folder = r"E:\Pear\Lidar\Dormancy Stage\Result_dir"

    # Process Point Clouds
    process_point_clouds(lidar_pcd_folder, lidar_result_folder, 0.15)

if __name__ == '__main__':
    main()