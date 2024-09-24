from Skeleton_utils import process_point_clouds

def main():
    
    uav_flower_pcd_folder = r"E:\Pear\UAV\Flowering Stage\Result_dir\pcd"
    uav_flower_result_folder = r"E:\Pear\UAV\Flowering Stage\Result_dir"

    # Process Point Clouds
    process_point_clouds(uav_flower_pcd_folder, uav_flower_result_folder, 0.5)

if __name__ == '__main__':
    main()