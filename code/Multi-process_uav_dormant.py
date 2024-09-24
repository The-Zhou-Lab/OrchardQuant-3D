from Skeleton_utils import process_point_clouds

def main():
    
    uav_dormant_pcd_folder = r"E:\Pear\UAV\Dormancy Stage\Result_dir\pcd"
    uav_dormant_result_folder = r"E:\Pear\UAV\Dormancy Stage\Result_dir"

    # Process Point Clouds
    process_point_clouds(uav_dormant_pcd_folder, uav_dormant_result_folder, 0.5)

if __name__ == '__main__':
    main()