import os
import open3d as o3d
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import numpy as np
import whitebox
import pandas as pd
import shutil
from gui_file.part1 import make_chm
from gui_file.part1 import make_polygon
from gui_file.part1 import denoising
from gui_file.part1 import ground_filtering
from gui_file.part2 import transform_chm
from gui_file.part2 import extract_bbox
from gui_file.part2 import get_centroid
from gui_file.part2 import find_pixel_coordinate
from gui_file.part2 import centroid_sorted
from gui_file.part2 import project_and_calcu_center
from gui_file.part2 import generate_tree_shapefiles
from gui_file.part2 import rename_file
from gui_file.part3 import las_to_pcd
from gui_file.part3 import extract_trunk_points
from gui_file.part3 import extract_skeleton
from gui_file.part5 import calculate_height
from gui_file.part5 import calculate_surfacearea_and_volumn
from gui_file.part5 import out_2d_area
from gui_file.part2 import denoise_pear_tree
from gui_file.part4 import write_to_pcd
from gui_file.part4 import repair_local
from gui_file.part4 import repair_tree_dijkstra
from gui_file.part4 import get_main_branch
from gui_file.part4 import combine_intensity_and_graph
from gui_file.part4 import remove_support_structure_intensity
from gui_file.part4 import remove_support_structure_graph
from gui_file.part4 import get_diameter
try:
    from pc_skeletor.laplacian import SLBC
    from multiprocessing import Pool, cpu_count

    SKELETOR_AVAILABLE = True
except ImportError:
    SKELETOR_AVAILABLE = False
    print("Warning: pc_skeletor package not found. Skeleton extraction features will be limited.")

# Clear KMeans warning
os.environ['OMP_NUM_THREADS'] = '1'


class RedirectText:
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.buffer = ""

    def write(self, string):
        self.buffer += string
        self.text_widget.configure(state="normal")
        self.text_widget.insert(tk.END, string)
        self.text_widget.see(tk.END)
        self.text_widget.configure(state="disabled")
        self.text_widget.update()

    def flush(self):
        pass


class LidarProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("OrchardQuant-3D")
        # self.root.geometry("800x700")
        # self.root.minsize(600, 550)
        self.root.maxsize(620, 453)
        self.status_var = tk.StringVar(value="")
        # Set application icon
        try:
            # Set application icon (make sure icon file exists in the correct path)
            icon_path = "assets/fruit_tree_icon.ico"  # Replace with your icon file path
            if os.path.exists(icon_path):
                self.root.iconbitmap(icon_path)
            else:
                print(f"Warning: Icon file not found at {icon_path}")
        except Exception as e:
            print(f"Error setting icon: {str(e)}")

        # Store current processing state
        self.processing = False

        # File path variables - store LiDAR and UAV data separately
        self.lidar_las_file = tk.StringVar()  # LiDAR LAS file

        self.lidar_rtk_file = tk.StringVar()  # LiDAR RTK file

        self.output_folder = tk.StringVar()

        # Variables for subsequent processing
        self.lidar_file = tk.StringVar()
        self.rtk_file = tk.StringVar()
        self.epsg_code = tk.StringVar(value="4938")  # Default EPSG code
        # LAS file variables for subsequent processing
        self.lidar_file = tk.StringVar()

        # Processing type selection - add multiple selection functionality
        self.process_lidar = tk.BooleanVar(value=True)  # Process LiDAR data

        # Parameter variables
        # Preprocessing parameters
        self.sor_sigma = tk.DoubleVar(value=5.0)
        self.sor_k = tk.IntVar(value=20)
        self.csf_resolution = tk.DoubleVar(value=0.1)
        self.csf_threshold = tk.DoubleVar(value=0.5)
        self.chm_resolution = tk.DoubleVar(value=0.01)

        # Skeleton extraction parameters
        self.slice_distance = tk.DoubleVar(value=0.001)
        self.trunk_radius = tk.DoubleVar(value=0.05)
        self.semantic_weight = tk.DoubleVar(value=5.0)
        self.down_sample = tk.DoubleVar(value=0.01)

        # Branch extraction parameters
        self.cluster_radius = tk.DoubleVar(value=0.05)
        self.intensity_threshold = tk.DoubleVar(value=1.1)

        # Voxel volume parameters
        self.voxel_size = tk.DoubleVar(value=0.1)
        self.show_plots = tk.BooleanVar(value=False)

        # Initialize results storage paths
        self.results_path = {}
        self.lidar_results_path = {}

        # Create interface
        self.create_widgets()
        # Initialize whitebox tools
        self.wbt = whitebox.WhiteboxTools()

    def create_widgets(self):
        # Create tab control
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Create four tabs
        self.tab1 = ttk.Frame(self.notebook)  # File input
        self.tab2 = ttk.Frame(self.notebook)  # Parameter settings
        self.tab3 = ttk.Frame(self.notebook)  # Preprocessing and structure extraction
        self.tab4 = ttk.Frame(self.notebook)  # Feature extraction

        # Add tabs to notebook
        self.notebook.add(self.tab1, text="1. Input files")
        self.notebook.add(self.tab2, text="2. Set parameters")
        self.notebook.add(self.tab3, text="3. Preprocess data, segment & analyze tree")
        self.notebook.add(self.tab4, text="4. Output tree-level traits")

        # Create contents for each tab
        self.create_file_input_tab()
        self.create_parameters_tab()
        self.create_processing_tab()
        self.create_traits_tab()

    def create_file_input_tab(self):
        """Create file input tab content"""
        # Create main frame
        main_frame = ttk.Frame(self.tab1, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Create data input section
        input_frame = ttk.LabelFrame(main_frame, text="Data input", padding="10")
        input_frame.pack(fill=tk.X, pady=(0, 12))

        # Create subframes to separate LiDAR and UAV data
        lidar_frame = ttk.LabelFrame(input_frame, text="LiDAR files", padding="10")
        lidar_frame.pack(fill=tk.X, pady=(0, 12))

        # ======= LiDAR data input =======
        # LiDAR file selection
        ttk.Label(lidar_frame, text="LiDAR point cloud file (LAS/LAZ):").grid(row=0, column=0, sticky=tk.W,
                                                                              pady=(7, 7))
        ttk.Entry(lidar_frame, textvariable=self.lidar_las_file, width=35).grid(row=0, column=1, sticky=tk.W + tk.E,
                                                                                padx=5, pady=7)
        ttk.Button(lidar_frame, text="Browse",
                   command=lambda: self.browse_file(self.lidar_las_file, "LiDAR LAS File", "las laz")).grid(row=0,
                                                                                                            column=2,
                                                                                                            pady=7)

        # LiDAR RTK file selection
        ttk.Label(lidar_frame, text="RTK shapefile:").grid(row=1, column=0, sticky=tk.W,
                                                           pady=(7, 7))
        ttk.Entry(lidar_frame, textvariable=self.lidar_rtk_file, width=35).grid(row=1, column=1, sticky=tk.W + tk.E,
                                                                                padx=5, pady=7)
        ttk.Button(lidar_frame, text="Browse",
                   command=lambda: self.browse_file(self.lidar_rtk_file, "LiDAR RTK File", "shp")).grid(row=1,
                                                                                                        column=2,
                                                                                                        pady=7)

        # Default setting to process LiDAR data, but don't show checkbox
        self.process_lidar.set(True)

        # CHANGED: Moved output folder selection out of the input frame
        # Create a separate frame for output settings
        output_frame = ttk.LabelFrame(main_frame, text="Results output", padding="10")
        output_frame.pack(fill=tk.X, pady=(0, 12))

        # Here use Grid layout to place the label in the first column, input box in the second column, browse button in the third column
        ttk.Label(output_frame, text="Select output folder:").grid(row=0, column=0, sticky=tk.W, pady=8)
        ttk.Entry(output_frame, textvariable=self.output_folder, width=35).grid(row=0, column=1, sticky=tk.W + tk.E,
                                                                                padx=5, pady=8)
        ttk.Button(output_frame, text="Browse",
                   command=lambda: self.browse_folder(self.output_folder, "Output Folder")).grid(row=0, column=2,
                                                                                                 padx=5,
                                                                                                 pady=8)
        # Add next button
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(fill=tk.X, pady=12)

        # Use lambda function to jump directly to the next tab
        ttk.Button(buttons_frame, text="Next",
                   command=lambda: self.notebook.select(1)).pack(side=tk.RIGHT)

    def create_parameters_tab(self):
        """Create parameter settings tab content"""
        # Create main frame
        main_frame = ttk.Frame(self.tab2, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        parameter_help = {
            "SOR sigma:": "The multiplier of the standard deviation, used to determine whether a point should be considered an outlier.",
            "SOR k": "Number of neighboring points to consider when filtering outliers.",
            "CSF resolution": "Cloth Simulation Filter resolution for ground filtering.",
            "CSF threshold": "Height threshold used in CSF algorithm. Points above this threshold from the cloth surface are considered non-ground.",
            "CHM resolution": "Resolution of the Canopy Height Model in meters.",
            "Z-axis sampling distance": "Slice distance in the Z direction for trunk extraction.",
            "Distance cluster radius": "Distance parameter for Euclidean clustering algorithm.",
            "Semantic weight": "Weighting factor for semantic information in skeleton extraction. ",
            "Voxel downsampling": "Voxel grid size for voxel downsampling.",
            "Distance cluster radius": "Distance parameter for Euclidean clustering algorithm.",
            "LiDAR intensity threshold": "LiDAR intensity threshold for distinguishing branches from supporting structures."
        }

        def show_parameter_help(param_name):
            help_text = parameter_help.get(param_name, "No help available for this parameter.")
            messagebox.showinfo(f"{param_name}", help_text)

        # Create validation function to limit maximum value - fixed for DoubleVar and IntVar handling
        def create_validator(var):
            def validate_max_value(*args):
                try:
                    value = var.get()  # For DoubleVar and IntVar, this directly returns number type
                    # Check if value exceeds 1000
                    if value > 1000:
                        messagebox.showerror("Error", f"Input value error,please enter a reasonable value.")
                        var.set(1000)  # Reset to maximum allowed value
                except:
                    # Handle any possible errors, such as variable having no value
                    pass

            return validate_max_value

        # Preprocessing parameters
        preprocess_frame = ttk.LabelFrame(main_frame, text="Preprocessing parameters", padding="8")
        preprocess_frame.pack(fill=tk.X, pady=(0, 12))

        def create_label_with_help(parent, text, param_name, row, col, **kwargs):
            frame = ttk.Frame(parent)
            frame.grid(row=row, column=col, sticky=tk.W, **kwargs)

            ttk.Label(frame, text=text).pack(side=tk.LEFT)

            canvas_size = 16
            canvas = tk.Canvas(frame, width=canvas_size, height=canvas_size,
                               bg=self.root.cget('background'),
                               highlightthickness=0)
            canvas.pack(side=tk.LEFT, padx=(4, 0))

            radius = canvas_size // 2 - 1
            center = canvas_size // 2
            canvas.create_oval(center - radius, center - radius,
                               center + radius, center + radius,
                               fill="#3498db", outline="")

            canvas.create_text(center, center, text="?", fill="white",
                               font=("Arial", 9, "bold"))

            canvas.bind("<Button-1>", lambda event: show_parameter_help(param_name))

            canvas.bind("<Enter>", lambda event: canvas.config(cursor="hand2"))
            canvas.bind("<Leave>", lambda event: canvas.config(cursor=""))

            return frame

        # Add validation tracking for each parameter
        self.sor_sigma.trace_add("write", create_validator(self.sor_sigma))
        self.sor_k.trace_add("write", create_validator(self.sor_k))
        self.csf_resolution.trace_add("write", create_validator(self.csf_resolution))
        self.csf_threshold.trace_add("write", create_validator(self.csf_threshold))
        self.chm_resolution.trace_add("write", create_validator(self.chm_resolution))
        self.slice_distance.trace_add("write", create_validator(self.slice_distance))
        self.trunk_radius.trace_add("write", create_validator(self.trunk_radius))
        self.semantic_weight.trace_add("write", create_validator(self.semantic_weight))
        self.down_sample.trace_add("write", create_validator(self.down_sample))
        self.cluster_radius.trace_add("write", create_validator(self.cluster_radius))
        self.intensity_threshold.trace_add("write", create_validator(self.intensity_threshold))

        # Noise filtering parameters
        create_label_with_help(preprocess_frame, "SOR sigma:", "SOR sigma:", 0, 0, padx=(0, 5), pady=(5, 3))
        ttk.Entry(preprocess_frame, textvariable=self.sor_sigma, width=8).grid(row=0, column=1, sticky=tk.W,
                                                                               pady=(5, 3))

        create_label_with_help(preprocess_frame, "SOR k:", "SOR k", 0, 2, padx=(15, 5), pady=(5, 3))
        ttk.Entry(preprocess_frame, textvariable=self.sor_k, width=8).grid(row=0, column=3, sticky=tk.W, pady=(5, 3))

        # Ground point filtering parameters
        create_label_with_help(preprocess_frame, "CSF resolution:", "CSF resolution", 1, 0, padx=(0, 5), pady=5)
        ttk.Entry(preprocess_frame, textvariable=self.csf_resolution, width=8).grid(row=1, column=1, sticky=tk.W,
                                                                                    pady=5)

        create_label_with_help(preprocess_frame, "CSF threshold:", "CSF threshold", 1, 2, padx=(15, 5), pady=5)
        ttk.Entry(preprocess_frame, textvariable=self.csf_threshold, width=8).grid(row=1, column=3, sticky=tk.W, pady=5)

        # CHM generation parameters
        create_label_with_help(preprocess_frame, "CHM resolution:", "CHM resolution", 2, 0, padx=(0, 5), pady=(3, 5))
        ttk.Entry(preprocess_frame, textvariable=self.chm_resolution, width=8).grid(row=2, column=1, sticky=tk.W,
                                                                                    pady=(3, 5))

        # Skeleton extraction parameters
        skeleton_frame = ttk.LabelFrame(main_frame, text="3D tree skeleton parameters", padding="8")
        skeleton_frame.pack(fill=tk.X, pady=(0, 12))

        create_label_with_help(skeleton_frame, "Z-axis sampling distance:", "Z-axis sampling distance", 0, 0, padx=(0, 5),
                               pady=(5, 3))
        ttk.Entry(skeleton_frame, textvariable=self.slice_distance, width=8).grid(row=0, column=1, sticky=tk.W,
                                                                                  pady=(5, 3))

        create_label_with_help(skeleton_frame, "Distance cluster radius:", "Distance cluster radius", 0, 2, padx=(15, 5),
                               pady=(5, 3))
        ttk.Entry(skeleton_frame, textvariable=self.trunk_radius, width=8).grid(row=0, column=3, sticky=tk.W,
                                                                                pady=(5, 3))

        create_label_with_help(skeleton_frame, "Semantic weight:", "Semantic weight", 1, 0, padx=(0, 5), pady=(3, 5))
        ttk.Entry(skeleton_frame, textvariable=self.semantic_weight, width=8).grid(row=1, column=1, sticky=tk.W,
                                                                                   pady=(3, 5))

        create_label_with_help(skeleton_frame, "Voxel downsampling:", "Voxel downsampling", 1, 2, padx=(15, 5), pady=(3, 5))
        ttk.Entry(skeleton_frame, textvariable=self.down_sample, width=8).grid(row=1, column=3, sticky=tk.W,
                                                                               pady=(3, 5))

        # Branch extraction parameters
        branch_frame = ttk.LabelFrame(main_frame, text="Tree Branch parameters", padding="8")
        branch_frame.pack(fill=tk.X, pady=(0, 12))

        create_label_with_help(branch_frame, "Distance cluster radius:", "Distance cluster radius", 0, 0, padx=(0, 5), pady=5)
        ttk.Entry(branch_frame, textvariable=self.cluster_radius, width=8).grid(row=0, column=1, sticky=tk.W, pady=5)

        create_label_with_help(branch_frame, "LiDAR intensity threshold:", "LiDAR intensity threshold", 0, 2, padx=(15, 5),
                               pady=5)
        ttk.Entry(branch_frame, textvariable=self.intensity_threshold, width=8).grid(row=0, column=3, sticky=tk.W,
                                                                                     pady=5)

        # Add next button
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(fill=tk.X, pady=10)

        ttk.Button(buttons_frame, text="Previous",
                   command=lambda: self.notebook.select(0)).pack(side=tk.LEFT)
        ttk.Button(buttons_frame, text="Next",
                   command=lambda: self.notebook.select(2)).pack(side=tk.RIGHT)

    def create_processing_tab(self):
        """Create preprocessing and structure extraction tab content"""
        # Create main frame
        main_frame = ttk.Frame(self.tab3, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Create preprocessing panel
        preprocess_frame = ttk.LabelFrame(main_frame, text="Step 1. Data preprocessing", padding="10")
        preprocess_frame.pack(fill=tk.X, pady=(0, 10))

        # Preprocessing buttons
        process_btn_frame = ttk.Frame(preprocess_frame)
        process_btn_frame.pack(fill=tk.X, pady=5)

        self.preprocess_btn = ttk.Button(process_btn_frame, text="Data preprocessing",
                                         command=self.start_preprocessing)
        self.preprocess_btn.pack(side=tk.LEFT, padx=5)

        self.preprocess_stop_btn = ttk.Button(process_btn_frame, text="Stop",
                                              command=self.stop_preprocessing,
                                              state=tk.DISABLED)
        self.preprocess_stop_btn.pack(side=tk.LEFT, padx=5)

        ttk.Button(process_btn_frame, text="View preprocessing results",
                   command=self.view_preprocess_results).pack(side=tk.LEFT, padx=5)

        # Preprocessing progress bar
        self.preprocess_progress = ttk.Progressbar(preprocess_frame, mode='determinate')
        self.preprocess_progress.pack(fill=tk.X, pady=5)

        # Single tree segmentation panel
        segment_frame = ttk.LabelFrame(main_frame, text="Step 2. Tree segmentation", padding="10")
        segment_frame.pack(fill=tk.X, pady=(0, 10))

        # Single tree segmentation buttons
        segment_btn_frame = ttk.Frame(segment_frame)
        segment_btn_frame.pack(fill=tk.X, pady=5)

        self.segment_btn = ttk.Button(segment_btn_frame, text="Tree segmentation",
                                      command=self.start_segmentation)
        self.segment_btn.pack(side=tk.LEFT, padx=5)

        self.segment_stop_btn = ttk.Button(segment_btn_frame, text="Stop",
                                           command=self.stop_segmentation,
                                           state=tk.DISABLED)
        self.segment_stop_btn.pack(side=tk.LEFT, padx=5)

        ttk.Button(segment_btn_frame, text="View segmentation results",
                   command=self.view_segment_results).pack(side=tk.LEFT, padx=5)

        # Single tree segmentation progress bar
        self.segment_progress = ttk.Progressbar(segment_frame, mode='determinate')
        self.segment_progress.pack(fill=tk.X, pady=5)

        # Skeleton extraction panel
        skeleton_frame = ttk.LabelFrame(main_frame, text="Step 3. 3D skeletonization of all the segmented trees",
                                        padding="10")
        skeleton_frame.pack(fill=tk.X, pady=(0, 10))

        # Skeleton extraction buttons
        skeleton_btn_frame = ttk.Frame(skeleton_frame)
        skeleton_btn_frame.pack(fill=tk.X, pady=5)

        self.skeleton_btn = ttk.Button(skeleton_btn_frame, text="3D skeletonization",
                                       command=self.start_skeleton_extraction)
        self.skeleton_btn.pack(side=tk.LEFT, padx=5)

        self.skeleton_stop_btn = ttk.Button(skeleton_btn_frame, text="Stop",
                                            command=self.stop_skeleton_extraction,
                                            state=tk.DISABLED)
        self.skeleton_stop_btn.pack(side=tk.LEFT, padx=5)

        ttk.Button(skeleton_btn_frame, text="View skeletonization results",
                   command=self.view_skeleton_results).pack(side=tk.LEFT, padx=5)

        # Skeleton extraction progress bar
        self.skeleton_progress = ttk.Progressbar(skeleton_frame, mode='determinate')
        self.skeleton_progress.pack(fill=tk.X, pady=5)

        # Add navigation buttons
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(fill=tk.X, pady=10)

        ttk.Button(buttons_frame, text="Previous",
                   command=lambda: self.notebook.select(1)).pack(side=tk.LEFT)
        ttk.Button(buttons_frame, text="Next",
                   command=lambda: self.notebook.select(3)).pack(side=tk.RIGHT)

    def create_traits_tab(self):
        """Create feature extraction tab content"""
        # Create main frame
        main_frame = ttk.Frame(self.tab4, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Branch extraction panel
        branch_frame = ttk.LabelFrame(main_frame, text="Quantify branch-level traits", padding="10")
        branch_frame.pack(fill=tk.X, pady=(0, 15))

        # Branch extraction buttons
        branch_btn_frame = ttk.Frame(branch_frame)
        branch_btn_frame.pack(fill=tk.X, pady=8)

        self.branch_btn = ttk.Button(branch_btn_frame, text="Quantify branch traits",
                                     command=self.start_branch_extraction)
        self.branch_btn.pack(side=tk.LEFT, padx=5)

        self.branch_stop_btn = ttk.Button(branch_btn_frame, text="Stop",
                                          command=self.stop_branch_extraction,
                                          state=tk.DISABLED)
        self.branch_stop_btn.pack(side=tk.LEFT, padx=5)

        ttk.Button(branch_btn_frame, text="Download results",
                   command=self.view_branch_results).pack(side=tk.LEFT, padx=5)

        # Branch extraction progress bar
        self.branch_progress = ttk.Progressbar(branch_frame, mode='determinate')
        self.branch_progress.pack(fill=tk.X, pady=8)

        # Tree feature statistics panel
        traits_frame = ttk.LabelFrame(main_frame, text="Quantify tree-level crown traits", padding="10")
        traits_frame.pack(fill=tk.X, pady=(0, 15))

        # Feature statistics buttons
        traits_btn_frame = ttk.Frame(traits_frame)
        traits_btn_frame.pack(fill=tk.X, pady=8)

        self.traits_btn = ttk.Button(traits_btn_frame, text="Quantify crown traits ",
                                     command=self.start_traits_analysis)
        self.traits_btn.pack(side=tk.LEFT, padx=5)

        self.traits_stop_btn = ttk.Button(traits_btn_frame, text="Stop",
                                          command=self.stop_traits_analysis,
                                          state=tk.DISABLED)
        self.traits_stop_btn.pack(side=tk.LEFT, padx=5)

        ttk.Button(traits_btn_frame, text="Download results",
                   command=self.view_traits_results).pack(side=tk.LEFT, padx=5)

        # Feature statistics progress bar
        self.traits_progress = ttk.Progressbar(traits_frame, mode='determinate')
        self.traits_progress.pack(fill=tk.X, pady=8)

        # Add navigation buttons
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(fill=tk.X, pady=15)

        ttk.Button(buttons_frame, text="Previous",
                   command=lambda: self.notebook.select(2)).pack(side=tk.LEFT)
    # Helper functions
    def browse_file(self, string_var, title, filetypes):
        """Generic file browsing function"""
        filetypes_list = [(f"{ext.upper()} Files", f"*.{ext}") for ext in filetypes.split()]
        filetypes_list.append(("All Files", "*.*"))

        file_path = filedialog.askopenfilename(
            title=f"Select {title}",
            filetypes=filetypes_list
        )
        if file_path:
            string_var.set(file_path)
            self.update_file_log(f"Selected {title}: {file_path}")

    def browse_folder(self, string_var, title):
        """Generic folder browsing function"""
        folder_path = filedialog.askdirectory(
            title=f"Select {title}"
        )
        if folder_path:
            string_var.set(folder_path)
            self.update_file_log(f"Selected {title}: {folder_path}")

    def update_file_log(self, message):
        """Update file tab log"""
        # Since file_log_text was commented out in the original code, we'll update the status bar instead
        self.update_status(message)
        # If you need to use a log in practice, you can uncomment the code below
        # self.file_log_text.config(state="normal")
        # self.file_log_text.insert(tk.END, message + "\n")
        # self.file_log_text.see(tk.END)
        # self.file_log_text.config(state="disabled")

    def update_process_log(self, message):
        """Update processing log (now just prints to console and status bar)"""

        print(f"[Process] {message}")
        self.update_status(message)

    def update_traits_log(self, message):
        """Update traits log (now just prints to console and status bar)"""

        print(f"[Traits] {message}")
        self.update_status(message)


    def update_status(self, message):
        """Update status bar message"""
        self.status_var.set(message)
        self.root.update_idletasks()

    def validate_inputs(self):
        """Validate input files and folders"""
        errors = []

        # Get selected processing types
        process_lidar = self.process_lidar.get()

        # If no processing type is selected, set to default LiDAR
        if not process_lidar :
            process_lidar = True
            self.process_lidar.set(True)
            self.update_file_log("No processing type selected, defaulting to LiDAR data processing")

        # Check required fields based on processing type
        if process_lidar:
            # Check LiDAR data
            if not self.lidar_las_file.get():
                errors.append("LiDAR LAS file not selected")
            elif not os.path.exists(self.lidar_las_file.get()):
                errors.append(f"LiDAR LAS file does not exist: {self.lidar_las_file.get()}")

            if not self.lidar_rtk_file.get():
                errors.append("LiDAR RTK file not selected")
            elif not os.path.exists(self.lidar_rtk_file.get()):
                errors.append(f"LiDAR RTK file does not exist: {self.lidar_rtk_file.get()}")

        # Check common required data
        if not self.output_folder.get():
            errors.append("Output folder not selected")

        # Show errors or continue
        if errors:
            error_msg = "\n".join(errors)
            messagebox.showerror("Input validation error", error_msg)
            return False
        else:
            self.update_file_log("All inputs validated successfully!")

            # Update main LAS and RTK file variables based on selected processing type
            if process_lidar:
                self.lidar_file.set(self.lidar_las_file.get())
                self.rtk_file.set(self.lidar_rtk_file.get())

            # Update processing type
            self.update_file_log(
                f"Will process: {'LiDAR' if process_lidar else ''}data")

            self.notebook.select(1)  # Jump to parameter settings tab
            return True

    def create_folder_structure(self, show_message=False):
        """Create project folder structure"""
        if not self.output_folder.get():
            if show_message:  # Only show error if explicitly requested
                messagebox.showerror("Error", "Please select an output folder first")
            return False

        try:
            base_dir = self.output_folder.get()
            if not os.path.exists(base_dir):
                os.makedirs(base_dir)
                self.update_file_log(f"Created main directory: {base_dir}")

            # Create main directory structure, separate LiDAR and UAV data
            dirs = [
                # LiDAR data folders
                os.path.join(base_dir, "LiDAR", "1_Preprocessing"),
                os.path.join(base_dir, "LiDAR", "2_Segmentation", "single_trees"),
                os.path.join(base_dir, "LiDAR", "2_Segmentation", "denoised_trees"),
                os.path.join(base_dir, "LiDAR", "3_Skeletonization", "trunk"),
                os.path.join(base_dir, "LiDAR", "3_Skeletonization", "branch"),
                os.path.join(base_dir, "LiDAR", "3_Skeletonization", "mesh"),
                os.path.join(base_dir, "LiDAR", "4_Branch traits"),
                os.path.join(base_dir, "LiDAR", "5_Crown traits"),
                os.path.join(base_dir, "LiDAR", "temp", "pcd"),

            ]

            for dir_path in dirs:
                os.makedirs(dir_path, exist_ok=True)
                if show_message:  # Only log if explicitly requested
                    self.update_file_log(f"Created directory: {dir_path}")

            # Update results path dictionary - create separate path dictionaries for each data type
            # LiDAR paths
            self.lidar_results_path = {
                "preprocessing": os.path.join(base_dir, "LiDAR", "1_Preprocessing"),
                "segmentation": os.path.join(base_dir, "LiDAR", "2_Segmentation"),
                "single_trees": os.path.join(base_dir, "LiDAR", "2_Segmentation", "single_trees"),
                "denoised_trees": os.path.join(base_dir, "LiDAR", "2_Segmentation", "denoised_trees"),
                "skeleton": os.path.join(base_dir, "LiDAR", "3_Skeletonization"),
                "trunk": os.path.join(base_dir, "LiDAR", "3_Skeletonization", "trunk"),
                "branch": os.path.join(base_dir, "LiDAR", "3_Skeletonization", "branch"),
                "mesh": os.path.join(base_dir, "LiDAR", "3_Skeletonization", "mesh"),
                "branches": os.path.join(base_dir, "LiDAR", "4_Branch traits"),
                "traits": os.path.join(base_dir, "LiDAR", "5_Crown traits"),
                "temp_pcd": os.path.join(base_dir, "LiDAR", "temp", "pcd")
            }

            # Determine default results path based on selected processing type
            if self.process_lidar.get():
                self.results_path = self.lidar_results_path
            else:
                self.results_path = self.lidar_results_path

            if show_message:  # Only show success message if explicitly requested
                messagebox.showinfo("Success", "Project folder structure created successfully!")

            return True

        except Exception as e:
            if show_message:  # Only show error if explicitly requested
                messagebox.showerror("Error", f"Error creating folder structure: {str(e)}")
            self.update_process_log(f"Error creating folder structure: {str(e)}")
            return False

    def save_parameters(self):
        """Save current parameter settings to file"""
        if not self.output_folder.get():
            messagebox.showerror("Error", "Please select an output folder first")
            return

        try:
            file_path = filedialog.asksaveasfilename(
                title="Save parameter file",
                defaultextension=".json",
                filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
                initialdir=self.output_folder.get()
            )

            if not file_path:
                return

            # Collect all parameters
            parameters = {
                # Preprocessing parameters
                "sor_sigma": self.sor_sigma.get(),
                "sor_k": self.sor_k.get(),
                "csf_resolution": self.csf_resolution.get(),
                "csf_threshold": self.csf_threshold.get(),
                "chm_resolution": self.chm_resolution.get(),

                # Skeleton extraction parameters
                "slice_distance": self.slice_distance.get(),
                "trunk_radius": self.trunk_radius.get(),
                "semantic_weight": self.semantic_weight.get(),
                "down_sample": self.down_sample.get(),

                # Branch extraction parameters
                "cluster_radius": self.cluster_radius.get(),
                "intensity_threshold": self.intensity_threshold.get(),

                # Voxel calculation parameters
                "voxel_size": self.voxel_size.get(),
                "show_plots": self.show_plots.get()
            }

            # Save to JSON file
            import json
            with open(file_path, 'w') as f:
                json.dump(parameters, f, indent=4)

            messagebox.showinfo("Success", f"Parameters saved to: {file_path}")

        except Exception as e:
            messagebox.showerror("Error", f"Error saving parameters: {str(e)}")

    def load_parameters(self):
        """Load parameter settings from file"""
        try:
            file_path = filedialog.askopenfilename(
                title="Load parameter file",
                filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
                initialdir=self.output_folder.get() if self.output_folder.get() else None
            )

            if not file_path:
                return

            # Load parameters from JSON file
            import json
            with open(file_path, 'r') as f:
                parameters = json.load(f)

            # Set parameter variables
            # Preprocessing parameters
            if "sor_sigma" in parameters:
                self.sor_sigma.set(parameters["sor_sigma"])
            if "sor_k" in parameters:
                self.sor_k.set(parameters["sor_k"])
            if "csf_resolution" in parameters:
                self.csf_resolution.set(parameters["csf_resolution"])
            if "csf_threshold" in parameters:
                self.csf_threshold.set(parameters["csf_threshold"])
            if "chm_resolution" in parameters:
                self.chm_resolution.set(parameters["chm_resolution"])

            # Skeleton extraction parameters
            if "slice_distance" in parameters:
                self.slice_distance.set(parameters["slice_distance"])
            if "trunk_radius" in parameters:
                self.trunk_radius.set(parameters["trunk_radius"])
            if "semantic_weight" in parameters:
                self.semantic_weight.set(parameters["semantic_weight"])
            if "down_sample" in parameters:
                self.down_sample.set(parameters["down_sample"])

            # Branch extraction parameters
            if "cluster_radius" in parameters:
                self.cluster_radius.set(parameters["cluster_radius"])
            if "intensity_threshold" in parameters:
                self.intensity_threshold.set(parameters["intensity_threshold"])

            # Voxel calculation parameters
            if "voxel_size" in parameters:
                self.voxel_size.set(parameters["voxel_size"])
            if "show_plots" in parameters:
                self.show_plots.set(parameters["show_plots"])

            messagebox.showinfo("Success", f"Parameters loaded from {file_path}")

        except Exception as e:
            messagebox.showerror("Error", f"Error loading parameters: {str(e)}")

    # Preprocessing and structure extraction related functions
    def start_preprocessing(self):
        """Start preprocessing operation - for all selected data types"""
        # Get selected processing types
        process_lidar = self.process_lidar.get()

        # Check if output folder is selected
        if not self.output_folder.get():
            messagebox.showerror("Error", "Please select an output folder first")
            return

        # Disable button, enable stop button
        self.preprocess_btn.config(state=tk.DISABLED)
        self.preprocess_stop_btn.config(state=tk.NORMAL)
        self.preprocessing = True

        # Reset progress bar
        self.preprocess_progress["value"] = 0

        # Update status
        self.update_status("Preprocessing in progress...")
        self.update_process_log("Start preprocessing...")

        # Start processing thread
        self.preprocess_thread = threading.Thread(target=self.process_preprocessing)
        self.preprocess_thread.daemon = True
        self.preprocess_thread.start()

    def stop_preprocessing(self):
        """Stop preprocessing operation"""
        if hasattr(self, 'preprocessing') and self.preprocessing:
            self.preprocessing = False
            self.update_status("Preprocessing stopped by user")
            self.update_process_log("Preprocessing stopped")
            self.preprocess_btn.config(state=tk.NORMAL)
            self.preprocess_stop_btn.config(state=tk.DISABLED)

    def process_preprocessing(self):
        """Execute preprocessing operation - modified to only process LiDAR data"""
        try:
            # Create folder structure if needed
            if not self.results_path:
                if not self.create_folder_structure():
                    self.update_process_log("Error: Failed to create folder structure")
                    return
                self.update_process_log("Created folder structure automatically")

            # Force to only process LiDAR data regardless of what was selected
            process_types = []
            if self.lidar_las_file.get() and os.path.exists(self.lidar_las_file.get()):
                process_types.append(("lidar", self.lidar_las_file.get(), self.lidar_rtk_file.get(),
                                      self.lidar_results_path))
            else:
                self.update_process_log("Error: LiDAR LAS file not found or not selected")
                messagebox.showerror("Error", "LiDAR LAS file not found or not selected")
                return

            total_steps = 4  # 4 steps for LiDAR processing
            current_step = 0

            # Process LiDAR data
            data_type, las_file, rtk_file, results_path = process_types[0]

            # self.update_process_log(f"Processing {data_type} data...")

            # Check necessary inputs
            if not las_file or not os.path.exists(las_file):
                # self.update_process_log(f"Error: {data_type} LAS file does not exist")
                return

            if not rtk_file or not os.path.exists(rtk_file):
                # self.update_process_log(f"Error: {data_type} RTK file does not exist")
                return

            # Step 1: Create region of interest polygon
            self.update_status(f"Step {current_step + 1}/{total_steps}: Creating ROI polygon ({data_type})...")
            # self.update_process_log(f"Creating ROI polygon ({data_type})...")
            self.preprocess_progress["value"] = (current_step / total_steps) * 100
            current_step += 1

            # Get EPSG code
            epsg = int(self.epsg_code.get())
            # self.update_process_log(f"rtk_file ({rtk_file})...")
            # self.update_process_log(f"rtk_file ({rtk_file})...")
            # Use set RTK file
            polygon_file, rtk_points = make_polygon(rtk_file, epsg=4938)
            # Store path to current data type's results path dictionary
            results_path["polygon_file"] = polygon_file
            results_path["rtk_points"] = rtk_points

            # Check if should stop
            if not self.preprocessing:
                return

            # Step 2: Point cloud denoising
            self.update_status(f"Step {current_step + 1}/{total_steps}: Point cloud denoising ({data_type})...")
            # self.update_process_log(f"Point cloud denoising ({data_type})...")
            self.preprocess_progress["value"] = (current_step / total_steps) * 100
            current_step += 1

            denoised_file, roi_las_name = denoising(las_file, polygon_file)
            # Store path to current data type's results path dictionary
            results_path["denoised_file"] = denoised_file

            # Check if should stop
            if not self.preprocessing:
                return

            # Step 3: Ground point filtering
            self.update_status(f"Step {current_step + 1}/{total_steps}: Ground point filtering ({data_type})...")
            # self.update_process_log(f"Ground point filtering ({data_type})...")
            self.preprocess_progress["value"] = (current_step / total_steps) * 100
            current_step += 1

            resolution = self.csf_resolution.get()
            threshold = self.csf_threshold.get()
            ground_file, above_ground_file = ground_filtering(denoised_file, resolution, threshold)
            # Store path to current data type's results path dictionary
            results_path["ground_file"] = ground_file
            results_path["above_ground_file"] = above_ground_file
            # Check if should stop
            if not self.preprocessing:
                return

            # Step 4: Generate canopy height model
            self.update_status(
                f"Step {current_step + 1}/{total_steps}: Generating canopy height model ({data_type})...")
            self.update_process_log(f"Generating canopy height model ({data_type})...")
            self.preprocess_progress["value"] = (current_step / total_steps) * 100
            current_step += 1

            chm_file = make_chm(ground_file, denoised_file, roi_las_name, polygon_file)
            # Store path to current data type's results path dictionary
            results_path["chm_file"] = chm_file

            # Save results to corresponding results path dictionary
            self.lidar_results_path = results_path

            # Ensure CHM file exists
            if os.path.exists(chm_file):
                self.update_process_log(f"{data_type} data processing complete, CHM file saved to: {chm_file}")
            else:
                self.update_process_log(f"Warning: {data_type} CHM file creation failed or path incorrect: {chm_file}")

            self.copy_preprocessed_files(data_type, results_path)

            # Update progress and status
            self.preprocess_progress["value"] = 100
            self.update_status("Preprocessing complete")
            self.update_process_log("LiDAR data preprocessing complete")

            # Show success message after processing
            self.root.after(0, lambda: messagebox.showinfo("Success",
                                                           "Preprocessing complete! LiDAR data has been processed."))

        except Exception as e:
            self.update_status(f"Error: {str(e)}")
            self.update_process_log(f"Error during preprocessing: {str(e)}")
            import traceback
            self.update_process_log(f"Error details: {traceback.format_exc()}")
            messagebox.showerror("Processing error", f"Error during preprocessing:\n{str(e)}")

        finally:
            # Restore button status
            self.root.after(0, lambda: self.preprocess_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.preprocess_stop_btn.config(state=tk.DISABLED))
            self.preprocessing = False

    def copy_preprocessed_files(self, data_type, results_path):

        target_dir = results_path["preprocessing"]
        os.makedirs(target_dir, exist_ok=True)

        # 收集所有需要复制的文件
        preprocessed_files = [
            results_path.get("polygon_file", ""),
            results_path.get("denoised_file", ""),
            results_path.get("ground_file", ""),
            results_path.get("above_ground_file", ""),
            results_path.get("chm_file", "")
        ]


        preprocessed_files = [f for f in preprocessed_files if f and os.path.exists(f)]

        if not preprocessed_files:
            self.update_process_log(f"No {data_type} preprocessed files found to copy")
            return


        for file in preprocessed_files:
            try:
                file_name = os.path.basename(file)
                dest_path = os.path.join(target_dir, file_name)

                if os.path.exists(dest_path):
                    os.remove(dest_path)
                shutil.copy2(file, target_dir)
            except Exception as e:
                self.update_process_log(f"Failed to copy file {file}: {str(e)}")


    def start_segmentation(self):
        """Start single tree segmentation operation - for all selected data types"""
        # Get selected processing types
        process_lidar = self.process_lidar.get()

        # Check if at least one data type is selected
        if not (process_lidar):
            messagebox.showerror("Error", "Please select at least one data processing type")
            return

        # Check if output folder is selected
        if not self.output_folder.get():
            messagebox.showerror("Error", "Please select an output folder first")
            return

        # Disable button, enable stop button
        self.segment_btn.config(state=tk.DISABLED)
        self.segment_stop_btn.config(state=tk.NORMAL)
        self.segmentation = True

        # Reset progress bar
        self.segment_progress["value"] = 0

        # Update status
        self.update_status("Performing single tree segmentation...")
        self.update_process_log("Starting single tree segmentation...")

        # Start processing thread
        self.segment_thread = threading.Thread(target=self.process_segmentation)
        self.segment_thread.daemon = True
        self.segment_thread.start()

    def stop_segmentation(self):
        """Stop single tree segmentation operation"""
        if hasattr(self, 'segmentation') and self.segmentation:
            self.segmentation = False
            self.update_status("Single tree segmentation stopped by user")
            self.update_process_log("Single tree segmentation stopped")
            self.segment_btn.config(state=tk.NORMAL)
            self.segment_stop_btn.config(state=tk.DISABLED)

    def process_segmentation(self):
        """Execute single tree segmentation operation - modified to process all selected data types"""
        try:
            # Create folder structure if needed
            if not self.results_path:
                if not self.create_folder_structure():
                    self.update_process_log("Error: Failed to create folder structure")
                    return
                # self.update_process_log("Created folder structure automatically")

            # Get selected processing types and corresponding result paths
            process_infos = []
            if self.process_lidar.get():
                process_infos.append(("lidar", self.lidar_results_path))

            total_steps = len(process_infos) * 4  # 4 steps for each type
            current_step = 0

            # Process each data type
            for data_type, results_path in process_infos:
                if not self.segmentation:
                    return

                self.update_process_log(f"Processing {data_type} data for single tree segmentation...")

                # Check necessary inputs and add detailed logs
                chm_file = results_path.get("chm_file")
                self.update_process_log(f"chm_path: {chm_file}")

                self.update_process_log(f"{data_type} CHM file path: {chm_file}")

                if not chm_file:
                    self.update_process_log(
                        f"Error: {data_type} CHM file path not set, please complete preprocessing first")
                    continue

                if not os.path.exists(chm_file):
                    self.update_process_log(f"Error: {data_type} CHM file does not exist: {chm_file}")
                    continue

                if not results_path.get("rtk_points"):
                    self.update_process_log(
                        f"Error: {data_type} RTK points do not exist, please complete preprocessing first")
                    continue

                denoised_file = results_path.get("denoised_file")
                self.update_process_log(f"{data_type} denoised point cloud file path: {denoised_file}")

                if not denoised_file:
                    self.update_process_log(
                        f"Error: {data_type} denoised point cloud file path not set, please complete preprocessing first")
                    continue

                if not os.path.exists(denoised_file):
                    self.update_process_log(
                        f"Error: {data_type} denoised point cloud file does not exist: {denoised_file}")
                    continue

                # Step 1: Transform CHM
                self.update_status(f"Step {current_step + 1}/{total_steps}: Transforming CHM ({data_type})...")
                self.update_process_log(f"Transforming CHM ({data_type})...")
                self.segment_progress["value"] = (current_step / total_steps) * 100
                current_step += 1

                # Create RTK shapefile path
                rtk_shapefile = self.lidar_rtk_file.get()
                self.update_process_log(f"rtk_shapefile ({rtk_shapefile})...")

                if not os.path.exists(rtk_shapefile):
                    self.update_process_log(f"Error: {data_type} Cannot find RTK shapefile: {rtk_shapefile}")
                    if "polygon_file" in results_path and os.path.exists(results_path["polygon_file"]):
                        rtk_shapefile = results_path["polygon_file"]
                        self.update_process_log(f"Trying to use polygon file instead: {rtk_shapefile}")
                    else:
                        continue

                transformed_chm_path = transform_chm(results_path["chm_file"], rtk_shapefile)
                results_path["transformed_chm"] = transformed_chm_path
                self.update_process_log(f"{data_type} transformed CHM saved to: {transformed_chm_path}")

                # Check if stop
                if not self.segmentation:
                    return

                # Step 2: Extract bounding boxes
                self.update_status(f"Step {current_step + 1}/{total_steps}: Extracting bounding boxes ({data_type})...")
                self.update_process_log(f"Extracting bounding boxes ({data_type})...")
                self.segment_progress["value"] = (current_step / total_steps) * 100
                current_step += 1

                centroid_list, bbox_list, binary = extract_bbox(transformed_chm_path)
                self.update_process_log(f"{data_type} found {len(bbox_list)} tree candidates")

                # Check if stop
                if not self.segmentation:
                    return

                # Step 3: Generate tree boundary files
                self.update_status(
                    f"Step {current_step + 1}/{total_steps}: Generating tree boundary files ({data_type})...")
                self.update_process_log(f"Generating tree boundary files ({data_type})...")
                self.segment_progress["value"] = (current_step / total_steps) * 100
                current_step += 1

                # Ensure directory for storing shapefiles exists
                shapefile_dir = os.path.join(os.path.dirname(transformed_chm_path), "shapefiles")
                os.makedirs(shapefile_dir, exist_ok=True)

                geo_pixels, shapefile_paths = generate_tree_shapefiles(bbox_list, transformed_chm_path)
                self.update_process_log(f"{data_type} generated {len(shapefile_paths)} tree boundary files")

                # Check if stop
                if not self.segmentation:
                    return

                # Step 4: Segment individual trees and denoise
                self.update_status(
                    f"Step {current_step + 1}/{total_steps}: Segmenting individual trees and denoising ({data_type})...")
                self.update_process_log(f"Segmenting individual trees and denoising ({data_type})...")
                self.segment_progress["value"] = (current_step / total_steps) * 100
                current_step += 1

                # Ensure output directory exists
                os.makedirs(results_path["single_trees"], exist_ok=True)

                # Segment individual trees for each shapefile
                for i, shapefile_path in enumerate(shapefile_paths):
                    if not self.segmentation:
                        break

                    # Extract tree ID
                    file_name = os.path.basename(shapefile_path)
                    tree_id = file_name.split('_')[1][:-4]

                    # Set output path
                    output_las_name = f"{tree_id}.las"
                    output_las_path = os.path.join(results_path["single_trees"], output_las_name)

                    self.update_process_log(f"{data_type} segmenting tree {i + 1}/{len(shapefile_paths)}: {tree_id}")

                    # Execute point cloud clipping
                    self.wbt.clip_lidar_to_polygon(
                        # i=results_path["denoised_file"],
                        i=results_path["above_ground_file"],

                        polygons=shapefile_path,
                        output=output_las_path
                    )

                # Denoise single tree point clouds
                self.update_process_log(f"{data_type} denoising single tree point clouds...")

                sigma = self.sor_sigma.get()
                k = self.sor_k.get()

                # Ensure output directory exists
                os.makedirs(results_path["denoised_trees"], exist_ok=True)

                # Execute denoising
                denoise_pear_tree(results_path["single_trees"], results_path["denoised_trees"], sigma, k)

                # Store the output_dir_las in lidar_las_path
                lidar_tree_centroids, lidar_output_dir_las = get_centroid(results_path["denoised_trees"])

                lidar_coordinate_list = find_pixel_coordinate(transformed_chm_path, lidar_tree_centroids)
                lidar_x_cluster_centers, lidar_x_midpoints, lidar_y_cluster_centers, lidar_y_midpoints = project_and_calcu_center(
                    transformed_chm_path, lidar_coordinate_list, 8, 9)
                lidar_sorted_centroid, lidar_sorted_tree_centroids = centroid_sorted(lidar_y_midpoints,
                                                                                     lidar_coordinate_list,
                                                                                     lidar_tree_centroids)

                # rename
                rename_file(lidar_tree_centroids, lidar_sorted_tree_centroids, lidar_output_dir_las)

                # Check denoising results
                if os.path.exists(results_path["denoised_trees"]):
                    denoised_files = [f for f in os.listdir(results_path["denoised_trees"]) if f.endswith('.las')]
                    self.update_process_log(f"{data_type} successfully denoised {len(denoised_files)} trees")

                self.update_process_log(f"{data_type} single tree segmentation complete")

                # Update progress and status
            self.segment_progress["value"] = 100
            self.update_status("Single tree segmentation complete")
            self.update_process_log("All data types single tree segmentation complete")

            # Show success message after processing
            self.root.after(0, lambda: messagebox.showinfo("Success",
                                                           "Single tree segmentation complete! All selected data types have been processed."))

        except Exception as e:
            self.update_status(f"Error: {str(e)}")
            self.update_process_log(f"Error during single tree segmentation: {str(e)}")
            import traceback
            self.update_process_log(f"Error details: {traceback.format_exc()}")
            messagebox.showerror("Processing error", f"Error during single tree segmentation:\n{str(e)}")

        finally:
            # Restore button status
            self.root.after(0, lambda: self.segment_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.segment_stop_btn.config(state=tk.DISABLED))
            self.segmentation = False

    def start_skeleton_extraction(self):
        """Start skeleton extraction operation - for all selected data types"""
        # Check if pc_skeletor is available
        if not SKELETOR_AVAILABLE:
            messagebox.showerror("Missing dependency",
                                 "pc_skeletor package is required for skeleton extraction. Please install it using pip install pc_skeletor.")
            return

        # Get selected processing types
        process_lidar = self.process_lidar.get()

        # Check if output folder is selected
        if not self.output_folder.get():
            messagebox.showerror("Error", "Please select an output folder first")
            return

        # Disable button, enable stop button
        self.skeleton_btn.config(state=tk.DISABLED)
        self.skeleton_stop_btn.config(state=tk.NORMAL)
        self.skeleton_extraction = True

        # Reset progress bar
        self.skeleton_progress["value"] = 0

        # Update status
        self.update_status("Performing skeleton extraction...")
        self.update_process_log("Starting skeleton extraction...")

        # Start processing thread
        self.skeleton_thread = threading.Thread(target=self.process_skeleton_extraction)
        self.skeleton_thread.daemon = True
        self.skeleton_thread.start()

    def stop_skeleton_extraction(self):
        """Stop skeleton extraction operation"""
        if hasattr(self, 'skeleton_extraction') and self.skeleton_extraction:
            self.skeleton_extraction = False
            self.update_status("Skeleton extraction stopped by user")
            self.update_process_log("Skeleton extraction stopped")
            self.skeleton_btn.config(state=tk.NORMAL)
            self.skeleton_stop_btn.config(state=tk.DISABLED)

    def process_skeleton_extraction(self):
        """Execute skeleton extraction operation - modified to process all selected data types"""
        try:
            # Create folder structure if needed
            if not self.results_path:
                if not self.create_folder_structure():
                    self.update_process_log("Error: Failed to create folder structure")
                    return
                self.update_process_log("Created folder structure automatically")

            # Get selected processing types and corresponding result paths
            process_infos = []
            if self.process_lidar.get():
                process_infos.append(("lidar", self.lidar_results_path))

            # Get parameters
            slice_distance = self.slice_distance.get()
            radius = self.trunk_radius.get()
            semantic_weight = self.semantic_weight.get()
            down_sample = self.down_sample.get()

            total_data_types = len(process_infos)
            current_data_type = 0

            for data_type, results_path in process_infos:
                if not self.skeleton_extraction:
                    return

                self.update_process_log(f"Processing {data_type} data for skeleton extraction...")

                # Check necessary inputs
                if not results_path.get("denoised_trees") or not os.path.exists(results_path.get("denoised_trees")):
                    self.update_process_log(
                        f"Error: {data_type} denoised single tree point clouds do not exist, please complete single tree segmentation first")
                    continue

                # Ensure temporary PCD directory exists
                temp_pcd_folder = results_path.get("temp_pcd")
                os.makedirs(temp_pcd_folder, exist_ok=True)

                # Step 1: Convert LAS files to PCD format
                self.update_status(
                    f"Data type {current_data_type + 1}/{total_data_types} - Step 1/3: Converting LAS files to PCD format ({data_type})...")
                self.update_process_log(f"Converting LAS files to PCD format ({data_type})...")
                self.skeleton_progress["value"] = (current_data_type / total_data_types) * 30

                input_folder = results_path["denoised_trees"]
                pcd_file_paths, offset_list = las_to_pcd(input_folder, temp_pcd_folder)

                if not self.skeleton_extraction:
                    return

                self.update_process_log(f"{data_type} converted {len(pcd_file_paths)} files to PCD format")

                # Check PCD file generation
                if not pcd_file_paths:
                    self.update_process_log(f"{data_type} no PCD files generated. Please check your input files.")
                    continue

                # Step 2: Extract trunk and branch point clouds
                self.update_status(
                    f"Data type {current_data_type + 1}/{total_data_types} - Step 2/3: Extracting trunk and branch point clouds ({data_type})...")
                self.update_process_log(f"Extracting trunk and branch point clouds ({data_type})...")
                self.skeleton_progress["value"] = (current_data_type / total_data_types) * 30 + 30 / total_data_types

                # Process each PCD file
                total_files = len(pcd_file_paths)
                for i, pcd_path in enumerate(pcd_file_paths):
                    if not self.skeleton_extraction:
                        break

                    pcd_file = os.path.basename(pcd_path)
                    self.update_process_log(f"{data_type} processing point cloud ({i + 1}/{total_files}): {pcd_file}")
                    self.update_status(f"{data_type} processing {pcd_file}...")

                    try:
                        pcd = o3d.io.read_point_cloud(pcd_path)

                        # Extract trunk and branch point clouds
                        try:
                            trunk_pcd, branch_pcd = extract_trunk_points(pcd, slice_distance, radius)
                        except UnboundLocalError:
                            self.update_process_log(f"{data_type} using fallback radius 0.5 for {pcd_file}")
                            trunk_pcd, branch_pcd = extract_trunk_points(pcd, slice_distance, 0.5)

                        # Create output folders
                        tree_name = pcd_file[:-4]  # remove .pcd suffix
                        trunk_folder = os.path.join(results_path["trunk"], tree_name)
                        branch_folder = os.path.join(results_path["branch"], tree_name)
                        mesh_folder = os.path.join(results_path["mesh"], tree_name).replace("\\", "/")

                        os.makedirs(trunk_folder, exist_ok=True)
                        os.makedirs(branch_folder, exist_ok=True)
                        os.makedirs(mesh_folder, exist_ok=True)

                        # Save trunk and branch point clouds
                        trunk_path = os.path.join(trunk_folder, f"trunk_{pcd_file}").replace("\\", "/")
                        branch_path = os.path.join(branch_folder, f"branch_{pcd_file}").replace("\\", "/")

                        o3d.io.write_point_cloud(trunk_path, trunk_pcd)
                        o3d.io.write_point_cloud(branch_path, branch_pcd)

                        # Extract skeleton
                        if len(np.asarray(trunk_pcd.points)) > 0:
                            try:
                                slbc = extract_skeleton(trunk_pcd, branch_pcd, semantic_weight, down_sample)
                                slbc.save(mesh_folder)
                            except Exception as e:
                                self.update_process_log(
                                    f"{data_type} error extracting skeleton for {pcd_file}: {str(e)}")
                        else:
                            self.update_process_log(
                                f"{data_type} skipping skeleton extraction for {pcd_file}: empty trunk point cloud")

                    except Exception as e:
                        self.update_process_log(f"{data_type} error processing {pcd_file}: {str(e)}")

                    # Update progress
                    file_progress = (i + 1) / total_files
                    self.skeleton_progress["value"] = (current_data_type / total_data_types) * 30 + (
                            30 + file_progress * 40) / total_data_types

                # Step 3: Complete processing
                self.update_status(
                    f"Data type {current_data_type + 1}/{total_data_types} - Step 3/3: Skeleton extraction complete ({data_type})")
                self.update_process_log(f"{data_type} skeleton extraction complete")

                # Check results
                mesh_folder = results_path["mesh"]
                if os.path.exists(mesh_folder):
                    tree_folders = [d for d in os.listdir(mesh_folder) if
                                    os.path.isdir(os.path.join(mesh_folder, d))]
                    self.update_process_log(
                        f"{data_type} successfully extracted skeletons for {len(tree_folders)} trees")
                else:
                    self.update_process_log(f"{data_type} warning: no skeleton meshes generated")

                current_data_type += 1

            # Update progress and status
            self.skeleton_progress["value"] = 100
            self.update_status("Skeleton extraction complete")
            self.update_process_log("All data types skeleton extraction complete")

            # Show success message after processing
            self.root.after(0, lambda: messagebox.showinfo("Success",
                                                           "Skeleton extraction complete! All selected data types have been processed."))

        except Exception as e:
            self.update_status(f"Error: {str(e)}")
            self.update_process_log(f"Error during skeleton extraction: {str(e)}")
            import traceback
            self.update_process_log(f"Error details: {traceback.format_exc()}")
            messagebox.showerror("Processing error", f"Error during skeleton extraction:\n{str(e)}")

        finally:
            # Restore button status
            self.root.after(0, lambda: self.skeleton_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.skeleton_stop_btn.config(state=tk.DISABLED))
            self.skeleton_extraction = False

    # Feature extraction related functions, also modified to support multiple data sources
    def start_branch_extraction(self):
        """Start branch extraction operation - for all selected data types"""
        # Get selected processing types
        process_lidar = self.process_lidar.get()


        # Check if at least one data type is selected
        if not (process_lidar):
            messagebox.showerror("Error", "Please select at least one data processing type")
            return

        # Check if output folder is selected
        if not self.output_folder.get():
            messagebox.showerror("Error", "Please select an output folder first")
            return

        # Disable button, enable stop button
        self.branch_btn.config(state=tk.DISABLED)
        self.branch_stop_btn.config(state=tk.NORMAL)
        self.branch_extraction = True

        # Reset progress bar
        self.branch_progress["value"] = 0

        # Update status
        self.update_status("Performing branch extraction...")
        self.update_traits_log("Starting branch extraction...")

        # Start processing thread
        self.branch_thread = threading.Thread(target=self.process_branch_extraction)
        self.branch_thread.daemon = True
        self.branch_thread.start()

    def stop_branch_extraction(self):
        """Stop branch extraction operation"""
        if hasattr(self, 'branch_extraction') and self.branch_extraction:
            self.branch_extraction = False
            self.update_status("Branch extraction stopped by user")
            self.update_traits_log("Branch extraction stopped")
            self.branch_btn.config(state=tk.NORMAL)
            self.branch_stop_btn.config(state=tk.DISABLED)

    def process_branch_extraction(self):
        """Execute branch extraction operation - modified to process all selected data types"""
        try:
            # Create folder structure if needed
            if not self.results_path:
                if not self.create_folder_structure():
                    self.update_traits_log("Error: Failed to create folder structure")
                    return
                self.update_traits_log("Created folder structure automatically")

            # Get selected processing types and corresponding result paths
            process_infos = []
            if self.process_lidar.get():
                process_infos.append(("lidar", self.lidar_results_path))

            # Get parameters
            cluster_radius = self.cluster_radius.get()
            intensity_threshold = self.intensity_threshold.get()

            total_data_types = len(process_infos)
            current_data_type = 0

            for data_type, results_path in process_infos:
                if not self.branch_extraction:
                    return

                self.update_traits_log(f"Processing {data_type} data for branch extraction...")

                # Check necessary inputs
                if not results_path.get("mesh") or not os.path.exists(results_path.get("mesh")):
                    self.update_traits_log(
                        f"Error: {data_type} skeleton extraction results do not exist, please complete skeleton extraction first")
                    continue

                if not results_path.get("denoised_trees") or not os.path.exists(results_path.get("denoised_trees")):
                    self.update_traits_log(
                        f"Error: {data_type} denoised single tree point clouds do not exist, please complete single tree segmentation first")
                    continue

                # Ensure branch output directory exists
                branches_dir = results_path.get("branches")
                os.makedirs(branches_dir, exist_ok=True)

                # Get file lists
                skeleton_files = [f for f in os.listdir(results_path["mesh"])
                                  if os.path.isdir(os.path.join(results_path["mesh"], f))]
                original_files = [f for f in os.listdir(results_path["denoised_trees"])
                                  if f.endswith(('.las', '.LAS'))]

                self.update_traits_log(f"{data_type} found {len(skeleton_files)} skeleton folders")
                self.update_traits_log(f"{data_type} found {len(original_files)} original point cloud files")

                # Ensure there are files to process
                if not skeleton_files or not original_files:
                    self.update_traits_log(f"{data_type} error: no input files found")
                    continue

                # Calculate trunk diameter
                trunk_path = results_path["temp_pcd"]
                pcd_files = [os.path.join(trunk_path, f) for f in os.listdir(trunk_path) if f.endswith('.pcd')]
                list_trunk = []
                for pcd_file in pcd_files:
                    pcd_read = o3d.io.read_point_cloud(pcd_file)
                    trunk_diameter_single = get_diameter(pcd_read)
                    list_trunk.append(trunk_diameter_single)
                trunk_diameter = np.mean(list_trunk) if list_trunk else 0.1
                trunk_radius = trunk_diameter/2
                self.update_traits_log(f"{trunk_radius} trunk_radius...")
                # Process each tree
                total_trees = len(skeleton_files)
                branch_counts = {}

                for i, skeleton_folder in enumerate(skeleton_files):
                    if not self.branch_extraction:
                        self.update_traits_log(f"{data_type} processing stopped by user")
                        break

                    # Find corresponding files
                    try:
                        skeleton_path = os.path.join(results_path["mesh"], skeleton_folder, "02_skeleton_SLBC.ply")

                        # Find corresponding original point cloud file
                        tree_id = skeleton_folder
                        las_file = next((f for f in original_files if tree_id in f), None)

                        if not las_file:
                            self.update_traits_log(f"{data_type} warning: cannot find LAS file for {tree_id}")
                            continue

                        las_path = os.path.join(results_path["denoised_trees"], las_file)

                        if not os.path.exists(skeleton_path):
                            self.update_traits_log(f"{data_type} warning: cannot find skeleton file: {skeleton_path}")
                            continue

                        if not os.path.exists(las_path):
                            self.update_traits_log(f"{data_type} warning: cannot find LAS file: {las_path}")
                            continue

                        self.update_traits_log(f"{data_type} processing tree {i + 1}/{total_trees}: {tree_id}")
                        self.update_status(f"{data_type} processing tree {i + 1}/{total_trees}: {tree_id}")

                        # Read skeleton point cloud
                        skeleton_pcd = o3d.io.read_point_cloud(skeleton_path)

                        # Convert LAS file to PCD
                        pcd_lidar, offset = write_to_pcd(las_path)

                        # Remove support structure using graph theory
                        self.update_traits_log(f"{data_type} removing support structure using graph theory...")
                        pcd_remove, skeleton_pcd_final, current_radius = remove_support_structure_graph(
                            skeleton_pcd, trunk_diameter
                        )

                        # Remove support structure using intensity values
                        self.update_traits_log(f"{data_type} removing support structure using intensity values...")
                        pcd_intensity, points_intensity_value = remove_support_structure_intensity(
                            las_path, skeleton_pcd_final, pcd_lidar, intensity_threshold, trunk_radius
                        )

                        # Combine results from both methods
                        self.update_traits_log(f"{data_type} combining intensity and graph methods...")
                        pcd_final, max_path = combine_intensity_and_graph(
                            pcd_remove, pcd_intensity, 0.1, current_radius, log_func=self.update_traits_log
                        )

                        # Repair skeleton
                        self.update_traits_log(f"{data_type} repairing skeleton...")
                        # Get trunk height
                        trunk_file = next((f for f in os.listdir(results_path["trunk"]) if tree_id in f), None)
                        if trunk_file:
                            trunk_folder = os.path.join(results_path["trunk"], trunk_file)
                            trunk_file_path = next((os.path.join(trunk_folder, f) for f in os.listdir(trunk_folder)),
                                                   None)
                            if trunk_file_path:
                                pcd_trunk = o3d.io.read_point_cloud(trunk_file_path)
                                trunk_length = np.max(np.asarray(pcd_trunk.points)[:, 2])
                            else:
                                trunk_length = 0.45  # Default value
                        else:
                            trunk_length = 0.45  # Default value

                        pcd_whole = repair_local(
                            pcd_final, pcd_lidar, max_path, cluster_radius, current_radius, trunk_length
                        )

                        # Further repair using Dijkstra algorithm
                        self.update_traits_log(f"{data_type} applying Dijkstra repair...")
                        pcd_final_repair, mean_distance = repair_tree_dijkstra(pcd_whole, trunk_diameter)

                        # Save repaired skeleton point cloud
                        points_final_repair = np.asarray(pcd_final_repair.points)
                        pcd_final_ply = o3d.geometry.PointCloud()
                        pcd_final_ply.points = o3d.utility.Vector3dVector(np.asarray(points_final_repair))
                        output_path_re = os.path.join(results_path["branches"], tree_id)
                        os.makedirs(output_path_re, exist_ok=True)
                        output_path = os.path.join(output_path_re, f"{tree_id}.ply")
                        o3d.io.write_point_cloud(output_path, pcd_final_ply)

                        # Calculate branches
                        self.update_traits_log(f"{data_type} calculating branch count...")
                        branch_nodes, branch_number, tree_points, outside_point = get_main_branch(
                            pcd_final_repair, mean_distance, current_radius, trunk_diameter
                        )

                        branch_counts[f"{data_type}_{tree_id}"] = branch_number
                        self.update_traits_log(f"{data_type} tree {tree_id} has {branch_number} branches")

                    except Exception as e:
                        self.update_traits_log(f"{data_type} error processing tree {i + 1}: {str(e)}")
                        import traceback
                        self.update_traits_log(f"Error details: {traceback.format_exc()}")

                    # Update progress
                    progress_per_data_type = 100 / total_data_types
                    tree_progress = (i + 1) / total_trees * progress_per_data_type
                    self.branch_progress["value"] = current_data_type * progress_per_data_type + tree_progress

                # Save branch counts to file
                if branch_counts:
                    branch_counts_file = os.path.join(results_path["branches"], "branch_counts.txt")
                    with open(branch_counts_file, 'w') as f:
                        for tree_id, count in branch_counts.items():
                            f.write(f"{tree_id}: {count}\n")
                    self.update_traits_log(f"{data_type} branch counts saved to {branch_counts_file}")

                    # Calculate average branch count
                    current_data_type_branches = {k: v for k, v in branch_counts.items() if
                                                  k.startswith(f"{data_type}_")}
                    if current_data_type_branches:
                        avg_branches = sum(current_data_type_branches.values()) / len(current_data_type_branches)
                        self.update_traits_log(f"{data_type} average branches per tree: {avg_branches:.1f}")

                self.update_traits_log(f"{data_type} branch extraction successfully completed")
                current_data_type += 1

            # Update progress and status
            self.branch_progress["value"] = 100
            self.update_status("Branch extraction complete")
            self.update_traits_log("All data types branch extraction complete")

            # Show success message
            self.root.after(0, lambda: messagebox.showinfo("Success",
                                                           "Branch extraction complete! All selected data types have been processed."))

        except Exception as e:
            self.update_status(f"Error: {str(e)}")
            self.update_traits_log(f"Error during branch extraction: {str(e)}")
            import traceback
            self.update_traits_log(f"Error details: {traceback.format_exc()}")
            messagebox.showerror("Processing error", f"Error during branch extraction:\n{str(e)}")

        finally:
            # Restore button status
            self.root.after(0, lambda: self.branch_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.branch_stop_btn.config(state=tk.DISABLED))
            self.branch_extraction = False

    def start_traits_analysis(self):
        """Start feature statistics operation - for all selected data types"""
        # Get selected processing types
        process_lidar = self.process_lidar.get()

        # Check if at least one data type is selected
        if not (process_lidar):
            messagebox.showerror("Error", "Please select at least one data processing type")
            return

        # Check if output folder is selected
        if not self.output_folder.get():
            messagebox.showerror("Error", "Please select an output folder first")
            return

        # Disable button, enable stop button
        self.traits_btn.config(state=tk.DISABLED)
        self.traits_stop_btn.config(state=tk.NORMAL)
        self.traits_analysis = True

        # Reset progress bar
        self.traits_progress["value"] = 0

        # Update status
        self.update_status("Performing feature statistics...")
        self.update_traits_log("Starting feature statistics...")

        # Start processing thread
        self.traits_thread = threading.Thread(target=self.process_traits_analysis)
        self.traits_thread.daemon = True
        self.traits_thread.start()

    def stop_traits_analysis(self):
        """Stop feature statistics operation"""
        if hasattr(self, 'traits_analysis') and self.traits_analysis:
            self.traits_analysis = False
            self.update_status("Feature statistics stopped by user")
            self.update_traits_log("Feature statistics stopped")
            self.traits_btn.config(state=tk.NORMAL)
            self.traits_stop_btn.config(state=tk.DISABLED)

    def process_traits_analysis(self):
        try:

            if not self.results_path:
                if not self.create_folder_structure():
                    self.update_traits_log("Error: Failed to create folder structure")
                    return
                self.update_traits_log("Created folder structure automatically")

            results_path = self.lidar_results_path

            self.update_traits_log("Starting to calculate LiDAR data canopy traits...")
            self.traits_progress["value"] = 5


            temp_pcd_folder = results_path.get("temp_pcd")
            if not temp_pcd_folder or not os.path.exists(temp_pcd_folder):
                self.update_traits_log(
                    "Error: Cannot find temporary PCD files directory, please complete preprocessing and segmentation steps first")
                messagebox.showerror("Error",
                                     "Cannot find temporary PCD files directory, please complete preprocessing and segmentation steps first")
                return

            traits_output_dir = results_path.get("traits")
            if not os.path.exists(traits_output_dir):
                os.makedirs(traits_output_dir, exist_ok=True)

            pcd_files = [f for f in os.listdir(temp_pcd_folder) if f.endswith('.pcd')]

            if not pcd_files:
                self.update_traits_log("Error: No pcd files found in the temporary directory")
                messagebox.showerror("Error", "No pcd files found in the temporary directory")
                return


            crown_traits_list = []
            total_files = len(pcd_files)

            self.update_traits_log(f"Found {total_files} pcd files, starting to calculate canopy traits...")
            parent_dir = os.path.dirname(results_path.get("temp_pcd"))

            mesh_file = os.path.join(parent_dir, "mesh")
            os.makedirs(mesh_file, exist_ok=True)
            for i, pcd_file in enumerate(pcd_files):
                try:
                    tree_id = pcd_file[:-4]
                    pcd_path = os.path.join(temp_pcd_folder, pcd_file)

                    self.update_traits_log(f"Processing tree {i + 1}/{total_files}: {tree_id}")


                    pcd = o3d.io.read_point_cloud(pcd_path)

                    # Calculate tree height
                    tree_height = calculate_height(pcd)

                    # Calculate surface area and volume
                    surface_area, volume = calculate_surfacearea_and_volumn(pcd,mesh_file,i)

                    # Calculate 2D horizontal projection area
                    area_2d = out_2d_area(pcd)

                    # Add results to the list
                    crown_traits_list.append({
                        'Tree_ID': tree_id,
                        'Height(m)': tree_height,
                        'Surface_Area(m2)': surface_area,
                        'Volume(m3)': volume,
                        'Projection_Area_2D(m2)': area_2d,

                    })

                    progress = 5 + (i + 1) / total_files * 85
                    self.traits_progress["value"] = progress

                except Exception as e:
                    self.update_traits_log(f"Error processing tree{tree_id}: {str(e)}")
                    import traceback
                    self.update_traits_log(f"Detailed error: {traceback.format_exc()}")

            if crown_traits_list:

                crown_df = pd.DataFrame(crown_traits_list)
                # Save crown traits csv file
                crown_csv_path = os.path.join(traits_output_dir, "results.csv")
                crown_df.to_csv(crown_csv_path, index=False)
                self.update_traits_log(f"Saved crown trait data to:{crown_csv_path}")
                self.update_traits_log(f"Processing complete, analyzed {len(crown_traits_list)}trees")

            else:
                self.update_traits_log("Warning: No crown traits were calculated")


            self.traits_progress["value"] = 100
            self.update_status("Canopy trait calculation complete")
            self.update_traits_log("LiDAR data canopy trait calculation complete")

            self.root.after(0, lambda: messagebox.showinfo("Success",
                                                           "Canopy trait calculation complete! LiDAR data has been processed."))

        except Exception as e:
            self.update_status(f"Error: {str(e)}")
            self.update_traits_log(f"Error occurred during canopy trait calculation:{str(e)}")
            import traceback
            self.update_traits_log(f"Error details: {traceback.format_exc()}")
            messagebox.showerror("Error", f"Error occurred during canopy trait calculation:\n{str(e)}")

        finally:

            self.root.after(0, lambda: self.traits_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.traits_stop_btn.config(state=tk.DISABLED))
            self.traits_analysis = False

    # Result viewing functions
    def view_preprocess_results(self):
        """View preprocessing results"""
        # Create folder structure if needed
        if not self.results_path:
            if not self.create_folder_structure():
                self.update_process_log("Error: Failed to create folder structure")
                return
            self.update_process_log("Created folder structure automatically")

        # Get selected processing types
        results_paths = []
        if self.process_lidar.get():
            results_paths.append(self.lidar_results_path["preprocessing"])

        if not results_paths:
            messagebox.showwarning("Warning", "Please select data types to view first")
            return

        if len(results_paths) == 1:
            # If only one path, open directly
            if os.path.exists(results_paths[0]):
                self.open_folder(results_paths[0])
            else:
                messagebox.showwarning("Warning", "No preprocessing results available to view")
        else:
            # If multiple paths, let user choose
            self.show_folder_selection_dialog("Select preprocessing results to view", results_paths)

    def view_segment_results(self):
        """View single tree segmentation results"""
        # Create folder structure if needed
        if not self.results_path:
            if not self.create_folder_structure():
                self.update_process_log("Error: Failed to create folder structure")
                return
            self.update_process_log("Created folder structure automatically")

        # Get selected processing types
        results_paths = []
        if self.process_lidar.get():
            results_paths.append(self.lidar_results_path["denoised_trees"])

        if not results_paths:
            messagebox.showwarning("Warning", "Please select data types to view first")
            return

        if len(results_paths) == 1:
            # If only one path, open directly
            if os.path.exists(results_paths[0]):
                self.open_folder(results_paths[0])
            else:
                messagebox.showwarning("Warning", "No single tree segmentation results available to view")
        else:
            # If multiple paths, let user choose
            self.show_folder_selection_dialog("Select single tree segmentation results to view", results_paths)

    def view_skeleton_results(self):
        """View skeleton extraction results"""
        # Create folder structure if needed
        if not self.results_path:
            if not self.create_folder_structure():
                self.update_process_log("Error: Failed to create folder structure")
                return
            self.update_process_log("Created folder structure automatically")

        # Get selected processing types
        results_paths = []
        if self.process_lidar.get():
            results_paths.append(self.lidar_results_path["mesh"])

        if not results_paths:
            messagebox.showwarning("Warning", "Please select data types to view first")
            return

        if len(results_paths) == 1:
            # If only one path, open directly
            if os.path.exists(results_paths[0]):
                self.open_folder(results_paths[0])
            else:
                messagebox.showwarning("Warning", "No skeleton extraction results available to view")
        else:
            # If multiple paths, let user choose
            self.show_folder_selection_dialog("Select skeleton extraction results to view", results_paths)

    def view_branch_results(self):
        """View branch extraction results"""
        # Create folder structure if needed
        if not self.results_path:
            if not self.create_folder_structure():
                self.update_traits_log("Error: Failed to create folder structure")
                return
            self.update_traits_log("Created folder structure automatically")

        # Get selected processing types
        results_paths = []
        if self.process_lidar.get():
            results_paths.append(self.lidar_results_path["branches"])

        if not results_paths:
            messagebox.showwarning("Warning", "Please select data types to view first")
            return

        if len(results_paths) == 1:
            # If only one path, open directly
            if os.path.exists(results_paths[0]):
                self.open_folder(results_paths[0])
            else:
                messagebox.showwarning("Warning", "No branch extraction results available to view")
        else:
            # If multiple paths, let user choose
            self.show_folder_selection_dialog("Select branch extraction results to view", results_paths)

    def view_traits_results(self):
        """View feature statistics results"""
        # Create folder structure if needed
        if not self.results_path:
            if not self.create_folder_structure():
                self.update_traits_log("Error: Failed to create folder structure")
                return
            self.update_traits_log("Created folder structure automatically")

        # Get selected processing types
        results_paths = []
        html_reports = []

        if self.process_lidar.get():
            traits_path = self.lidar_results_path["traits"]
            if os.path.exists(traits_path):
                results_paths.append(traits_path)
                html_report = os.path.join(traits_path, "lidar_trait_report.html")
                if os.path.exists(html_report):
                    html_reports.append(html_report)

        if not results_paths:
            messagebox.showwarning("Warning", "Please select data types to view or perform feature statistics first")
            return

        # If HTML reports exist, open HTML reports preferentially
        if html_reports:
            if len(html_reports) == 1:
                self.open_file(html_reports[0])
            else:
                self.show_file_selection_dialog("Select feature report to view", html_reports)
        else:
            # Otherwise open folders
            if len(results_paths) == 1:
                self.open_folder(results_paths[0])
            else:
                self.show_folder_selection_dialog("Select feature statistics results to view", results_paths)

    def show_folder_selection_dialog(self, title, folders):
        """Show folder selection dialog"""
        dialog = tk.Toplevel(self.root)
        dialog.title(title)
        dialog.geometry("500x300")
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(dialog, text="Please select a folder to view:").pack(pady=(10, 5))

        listbox = tk.Listbox(dialog, width=70, height=10)
        listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Add all folders to list
        for i, folder in enumerate(folders):
            folder_name = os.path.basename(os.path.dirname(folder)) + "/" + os.path.basename(folder)
            listbox.insert(tk.END, f"{i + 1}. {folder_name} ({folder})")

        # Add buttons
        buttons_frame = ttk.Frame(dialog)
        buttons_frame.pack(fill=tk.X, padx=10, pady=10)

        def on_select():
            selection = listbox.curselection()
            if selection:
                index = selection[0]
                selected_folder = folders[index]
                dialog.destroy()
                self.open_folder(selected_folder)

        ttk.Button(buttons_frame, text="Open selected folder", command=on_select).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Cancel", command=dialog.destroy).pack(side=tk.RIGHT, padx=5)

    def show_file_selection_dialog(self, title, files):
        """Show file selection dialog"""
        dialog = tk.Toplevel(self.root)
        dialog.title(title)
        dialog.geometry("500x300")
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(dialog, text="Please select a file to view:").pack(pady=(10, 5))

        listbox = tk.Listbox(dialog, width=70, height=10)
        listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Add all files to list
        for i, file in enumerate(files):
            file_name = os.path.basename(file)
            data_type = os.path.basename(os.path.dirname(file))
            listbox.insert(tk.END, f"{i + 1}. {data_type} - {file_name}")

        # Add buttons
        buttons_frame = ttk.Frame(dialog)
        buttons_frame.pack(fill=tk.X, padx=10, pady=10)

        def on_select():
            selection = listbox.curselection()
            if selection:
                index = selection[0]
                selected_file = files[index]
                dialog.destroy()
                self.open_file(selected_file)

        ttk.Button(buttons_frame, text="Open selected file", command=on_select).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Cancel", command=dialog.destroy).pack(side=tk.RIGHT, padx=5)

    def open_folder(self, folder_path):
        """Open specified folder"""
        try:
            normalized_path = os.path.normpath(folder_path)

            if os.name == 'nt':  # Windows
                os.startfile(normalized_path)
            elif os.name == 'posix':  # Linux/macOS
                os.system(f'xdg-open "{normalized_path}"')
            else:
                messagebox.showinfo("Information", f"Folder path: {normalized_path}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to open folder: {str(e)}")

    def open_file(self, file_path):
        """Open specified file"""
        try:
            normalized_path = os.path.normpath(file_path)

            if os.name == 'nt':  # Windows
                os.startfile(normalized_path)
            elif os.name == 'posix':  # Linux/macOS
                if file_path.endswith(('.csv', '.txt')):
                    os.system(f'xdg-open "{normalized_path}"')
                else:
                    os.system(f'xdg-open "{normalized_path}"')
            else:
                messagebox.showinfo("Information", f"File path: {normalized_path}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to open file: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = LidarProcessingApp(root)
    root.mainloop()