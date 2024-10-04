# OrchardQuant-3D - combining drone and LiDAR to perform scalable 3D phenotyping for characterising key canopy and floral traits in fruit orchards

Yunpeng Xia<sup>1*</sup>, Hanghang Li<sup>1</sup>, Gang Sun<sup>1</sup>, Ji Zhou<sup>1,2*</sup>

<sup>1</sup>Academy for Advanced Interdisciplinary Studies, The Crop Phenomics Research Center, Nanjing Agricultural University, Nanjing 210095, China;

<sup>2</sup>Data Sciences Department, Crop Science Centre (CSC), National Institute of Agricultural Botany (NIAB), Cambridge CB3 0LE, UK;

<sup>*</sup>Correspondence: xiayunpeng@stu.njau.edu.cn; Ji.Zhou@NJAU.edu.cn, Ji.Zhou@NIAB.com, or JZ655@cam.ac.uk

## Install Python, Anaconda and Libraries
If you wish to run OrchardQuant-3D from source code, you will need to set up Python on your operating system. 

1. Install Python releases:
   
   •	Read the beginner’s guide to Python if you are new to the language: 
   https://wiki.python.org/moin/BeginnersGuide
   
   •	For Windows users, Python 3 release can be downloaded via: 
   https://www.python.org/downloads/windows/
   
   •	For Mac OS users, Python 3 release can be downloaded via: 
   https://www.python.org/downloads/mac-osx/
   
   •	OrchardQuant-3D only supports Python 3 onwards

2. Install Anaconda Python distribution:
   
   •	Read the install instruction using the URL: https://docs.continuum.io/anaconda/install
   
   •	For Windows users, a detailed step-by-step installation guide can be found via: 
   https://docs.continuum.io/anaconda/install/windows 
   
   •	For Mac OS users, a detailed step-by-step installation guide can be found via:
   https://docs.continuum.io/anaconda/install/mac-os.html
   
   •	An Anaconda Graphical installer can be found via: 
   https://www.continuum.io/downloads

   •	We recommend users install the latest Anaconda Python distribution

3. Install packages:

   • OrchardQuant-3D uses a number of 3rd-party libraries that you may need to add to your conda environment.
   These include, but are not limited to:
   
       Laspy=2.5.1
       Whitebox==2.3.1
       GDAL=3.6.2
       Rasterio=1.3.9
       Pc-skeletor=1.0.0
       Dijkstra=0.2.1
       OpenCV-Python=4.8.1
       Open3d=0.18.0
       Scikit-image=0.20.0
       Scikit-learn=1.3.2
       Geopandas=0.13.2
       PyShp=2.3.1
       CSF=1.1.5
       Matplotlib =3.7.3
       Pandas=2.0.3
       Numpy=1.23.2
       Scipy=1.9.1
       Vtk=9.2.6
       mistree=1.2.0
## Running OrchardQuant-3D

After successfully installing the required third-party libraries, you can download the fruit tree point cloud test data and the necessary files for running the code from the compressed file (OrchardQuant-3D.zip) that we have provided. Then, please run the code in the latest version we provided to obtain the result data. To reduce the running time, we have also included a multi-process version of the code.
