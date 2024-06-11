# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 10:52:22 2024

@author: Ron Simenhois
"""
import os
from glob import glob
import json
import numpy as np
from osgeo import gdal
import richdem as rd
import pdal
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
from scipy.stats import kurtosis#, skew
import warnings
warnings.filterwarnings("ignore")


def las_to_dem_tif(in_laz_file,
                   out_dem_file):
    """
    This function take a path to a LAS/LAZ file and generate and save 
    a tif DEM from that file. The tig file is in UTM S13 (Colorado) projections.
    The steps for the convertion are as follow:
    Create the pipline transtale filter to convert the laz file to ground trueth laz file:
        1. Reproject the Point Cloud:
            The first object specifies the filters.reprojection command. 
            This reprojects the input point cloud. Reprojection of point 
            clouds is often necessary to ensure the points are referenced to a linear coordinate system. In this case, the input point cloud will be reprojected to NAD83 UTM Zone 13S.
        2. Reclassify all Points to 0:
            filters.assign gives all the points in the point cloud a 
            classification value of 0. Lidar specification defines a 
            classification of 0 as points that are created but never 
            classified. This command effectively resets the classification 
            of all points.
        3. Remove Low-Noise Points:
            The extended local minimum (ELM) algorithm identifies points that 
            have low noise and are likely not suitable for analysis. The PDAL 
            filter filters.elm implements the ELM algorithm. This command will 
            identify low-noise points with ELM and classify them with a value 
            of 7, which represents noise points in accordance with lidar specs.
        4. Remove Outliers:
            apply an outlier filter to identify points that don’t seem to 
            fit with points around them. Points identified as outliers are 
            given a classification of 7. You can read more about outlier 
            identification in the PDAL documentation 
            (https://pdal.io/en/2.7-maintenance/stages/filters.outlier.html#filters-outlier).
        5. Execute SMRF Ground Segmentation Algorithm:
            Until this point, we’ve just been setting up the point cloud and 
            removing noise (i.e. points that potentially contain errors). 
            The ground segmentation algorithm is what will actually give us 
            a point cloud of ground-only points.
            Here we use the Simple Morphological Filter (or SMRF) for ground 
            segmentation. This is a PDAL-implemented algorithm based on: 
                https://doi.org/10.1016/j.isprsjprs.2012.12.002
            Basically, SMRF has been optimized to identify ground points 
            that will create a detailed DEM.
            Here is a simple description of the SMRF parameters.
            > - "ignore" tells SMRF not to use points with classification of 7 in the segmentation.
            > - "slope" is a slope threshold used to identify points that are associated with objects (trees, building, etc) and not the ground
            > - "window" specifies the maximum distance/radius for the SMRF window to be applied within the point cloud
            > - "treshold" is an elevation threshold used to identify objects
            > - "scalar" is an elevation scaling factor.<br>
            
            The SMRF filter will identify ground points according to the input parameters and classify them with a value of 2.
        6. Extact Ground Points: 
            Now that we have identified ground points with SMRF we want to 
            extract only these points so they can be saved to a separate point 
            cloud. This is done with a PDAL range filter(filters.range).

    Parameters
    ----------
    in_laz_file : str
        A path to the origin LAS/LAZ file to conver to the DEM tif file.

    out_dem_file : str
        A path to save the DEM tif file.

    Returns
    -------
    None.

    """
    ground_filter = """{
                        "pipeline":[
                            {
                              "type":"filters.reprojection",
                              "out_srs":"EPSG:26913"
                            },
                            {
                              "type":"filters.assign",
                              "assignment":"Classification[:]=0"
                            },
                            {
                              "type":"filters.elm"
                            },
                            {
                              "type":"filters.outlier"
                            },
                            {
                              "type":"filters.smrf",
                              "ignore":"Classification[7:7]",
                              "slope":1.7,
                              "window":16,
                              "threshold":0.45,
                              "scalar":1.2
                            },
                            {
                              "type":"filters.range",
                              "limits":"Classification[2:2]"
                            },
                            {
                              "type":"writers.las",
                              "filename":ground_laz,
                              "minor_version":1.4,
                              "extra_dims":"all"
                            }
                        ]
    }"""



    with open('ground_filter.json', 'w') as gf:
        json.dump(ground_filter, gf)
        
    # Run filter to generate ground points laz file
    translate_cmd = f'pdal translate {in_laz_file} ground_points_laz.laz --json pipeline.json'
    os.system(translate_cmd)
    # Create a pdal pipeline to convert the ground points laz file to tif DEM
    tiff_writer = {
                "pipeline": [
                    "ground_points_laz.laz",
                    {
                        "filename":f"{out_dem_file}",
                        "gdaldriver":"GTiff",
                        "output_type":"all",
                        "resolution":"1.0",
                        "type": "writers.gdal"
                        }
                    ]
    }
    r = pdal.Pipeline(json.dumps(tiff_writer))
    r.execute()
    
    

def get_polygons_from_kml(kml_path,
                          save_shp=True,
                          shp_file="No file saved",
                          crs="EPSG:26913"):
    """
    This function reads kml file and convert its Polygons points and save it
    into a shp file   

    Parameters
    ----------
    kml_path : str
        Path to the kml file to parse.
    save_shp : Bool, optional
        To save the Polygon as shp file or not. The default is True
    shp_file : str
        shp file path, the default is "No file saved".
    crs : str, optional
        DESCRIPTION. The default is "EPSG:26913" for Colorado.

    Returns
    -------
    polys : list of shaply Polygons
        list of shaply Polygons.
    shp_file : str
        The saved shp file path.

    """
    
    import xml.etree.ElementTree as ET
    import geopandas as gpd
    from shapely.geometry import Polygon

    root = ET.parse(kml_path).getroot()
    p_cords = [c.text.strip() for c in list(root.iter()) if 'coordinates' in c.tag]
    polys = []
    for p_c in p_cords:
        p_c = [c.split(',') for c in p_c.split(' ')]
        poly = Polygon(np.array(p_c)[:, :-1].astype(float))
        polys.append(poly)
    if save_shp:
        features = np.arange(len(polys))
        gdr = gpd.GeoDataFrame({'feature': features, 'geometry': poly}, crs="EPSG:26913")
        if shp_file is None:
            shp_file = path.replace('kml', 'shp')    
        gdr.to_file(path.replace('kml', 'shp'))
    
    return polys, shp_file


def load_tif_as_Array(path='../data/Sisters/Sisters.tif'):
    """
    

    Parameters
    ----------
    path : TYPE, optional
        DESCRIPTION. The default is '../data/Sisters/Sisters.tif'.

    Returns
    -------
    band : TYPE
        DESCRIPTION.

    """
    
    gdal.UseExceptions()
    ds = gdal.Open(path)
    
    band = ds.GetRasterBand(1).ReadAsArray()
    ds = None
    return band


def crop_tif(in_path,
             out_path,
             polygon_path=None, # shp file, kml file ...
             minx=None, maxx=None,
             miny=None, maxy=None):
    """
    This function crope a GeoTif to a shp, kml file polygons or a bounding box

    Parameters
    ----------
    in_path : str
        path to the original tif file.
    out_path : str
        path to save the croped tif.
    polygon_path : str, optional
        path to the shp, kml file. The default is None. It needs to have str 
        value if the bounding box is None
    minx : float, optional
        The bounding box cordinate. The default is None.
    maxx : float, optional
        The bounding box cordinate. The default is None.
    miny : float, optional
        The bounding box cordinate. The default is None.
    maxy : float, optional
        The bounding box cordinate. The default is None.

    Returns
    -------
    dem : numpy array
        The croped tif raster band 1 as numpy array.

    """
    
    gdal.UseExceptions()
    assert polygon_path is not None or minx is not None, 'Geometry is missing...' 
    
    clip =  gdal.Warp(destNameOrDestDS = out_path, 
                      srcDSOrSrcDSTab  = in_path,
                      cutlineDSName    = polygon_path, 
                      cropToCutline    = True,
                      copyMetadata     = True,
                      dstNodata        = np.nan)
    
    dem = clip.GetRasterBand(1).ReadAsArray()
    clip = None
    return dem


def show_array(a, cmap='winter', title=''):
    """
    Display a 2D array as an image

    Parameters
    ----------
    a : 2D numpy array
        The data array to display.
   cmap : str, optional
       A color map. The default is 'winter_r'.
   title : str, optional
       The plot title. The default is ''.

    Returns
    -------
    None.

    """
    
    fig, ax = plt.subplots(figsize=(12,12))
    im = ax.imshow(a, cmap=cmap)
    plt.title(title)
    fig.colorbar(im, ax=ax)
    plt.show()
    
def plot_3d(arr, cmap='winter_r', title='', **kwargs):
    """
    This fuction generate a 3D plot from a 2D array

    Parameters
    ----------
    arr : 2D Numpy array
        A 2D array contain the data to polot.
    cmap : str, optional
        A color map. The default is 'winter_r'.
    title : str, optional
        The plot title. The default is ''.
    **kwargs : dict
        Azimuth ans elevation view to set the plot initial view.

    Returns
    -------
    None.

    """
    
    view_elev = kwargs.get('view elev', 20)
    view_azi = kwargs.get('view azi', 60)
    ax = plt.figure(figsize=(22,22)).add_subplot(projection='3d')
    height, width = arr.shape
    x = np.arange(width)
    y = np.arange(height)
    x, y = np.meshgrid(x, y)
    x, y = x.flatten(), y.flatten()
    ax.plot_trisurf(x, y, arr.flatten(), linewidth=0.2, antialiased=True, cmap='winter_r')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.view_init(elev=view_elev, azim=view_azi)
    ax.grid(False)
    plt.title(title)
    plt.show()
    
   
def get_slope_attributes(dem_path, *args):
    """
    This function gets a path to a DEM and a list of sloe attributes to 
    generate. It saves a tig files with these attriutes and returns a dict 
    where the keys are the paths to the tif files.  
    
    Parameters
    ----------
    dem_path : TYPE
        path to the dem to generate the attributs from.
    *args : list os str
        A list of the variables to gerate.
        Possibole options are: slope_riserun, slope_percentage, slope_degrees, 
                               slope_radians, aspect, curvature, 
                               planform_curvature, profile_curvature

    Returns
    -------
    attr_paths : list arrays contaning the attributes data
        DESCRIPTION.

    """
    
    gdal.UseExceptions()
    attributes = {'richdem': ['slope_riserun', 'slope_percentage', 'slope_degrees', 'slope_radians',
                              'aspect', 'curvature', 'planform_curvature', 'profile_curvature'],
                  'gdal': ['hillshade', 'slope', 'aspect', 'color-relief', 'TRI', 'TPI', 'Roughness']}

    attr_arrays = {}
    for attr in args:
        name = os.path.split(dem_path   )[1]
        name = name[:name.rfind(' ')] + f' {attr}'
        try:
            if attr in attributes['richdem']:
                dem = rd.LoadGDAL(dem_path)
                dem = rd.FillDepressions(dem, epsilon=False, in_place=False)
                attr_band = rd.TerrainAttribute(dem, attrib=attr)
            else:
                out_path = dem_path.replace('.tif', f'_{attr}.tif')
                #open(out_path, 'a').close() # Otherwise, DEMPsocessing will raise a RuntimeError: pszColorFilename == NULL.
                ds = gdal.Open(dem_path)
                attr_df = gdal.DEMProcessing(out_path, ds, attr, computeEdges=True)
                attr_band = attr_df.GetRasterBand(1).ReadAsArray()
            attr_arrays[name] = attr_band    
        except Exception as e:
            print(e)
    return attr_arrays
    

def plot_attr_vals_probability(attr, title, **paths):
    """
    This function plots a probability histogram and the corresponding 
    Kernel Density Estimators (kde) of the different values in a 2D array. 

    Parameters
    ----------
    attr : str
        Slope attribute, can be "aspect" "slope angle" "roughness"...
    title : str
        plot title.
    **paths : dict
        A dictionary where the values are the 2D arrays and the keys are 
        the array descriptions.

    Returns
    -------
    kurtosis_d : dict
        A dictionary with keys as the kurtosis of each array's values 
        distribution and keys as the descriptions of the distributions

    """
       
    clip_vals = {
                    'slope_degrees':
                    {
                        'min': 30, 
                        'max': 65,
                    },
                    'slope_riserun':
                    {
                        'min': 0.6, 
                        'max': 2,
                    },
                      'aspect':
                    {
                        'min': 0, 
                        'max': 360
                    }
                  }
    
    fig, ax = plt.subplots(figsize=(32,20))
    kurtosis_d = {}
    for path, arr in tqdm(paths.items(), total=len(paths), desc='Fetching start zones data'): 
        arr = arr[~np.isnan(arr)]
        unique, counts = np.unique(arr, return_counts=True)
        if counts.max()/arr.shape[0] > 0.1:
            arr = arr[arr != unique[counts.argmax()]]
        arr = arr[np.where(np.logical_and(arr>=clip_vals[attr]['min'], arr<=clip_vals[attr]['max']))]
        if attr == 'aspect':
            arr = (arr + 180) % 360
        ku = round(kurtosis(arr.flatten()), 2)
        kurtosis_d[path] = ku
        label = f'{path}, (kurtosis: {ku})'
        sns.histplot(arr.flatten(), stat='probability', bins=100, 
                     ax=ax, kde=True, line_kws={'lw': 3}, label=label, 
                     alpha=0.4)
    if attr == 'aspect':
        corected_xticks = (ax.get_xticks() + 180) % 360
        ax.set_xticklabels(corected_xticks.astype(int)) 
    ax.set_xlabel('Slope (%)', fontsize = 20)
    ax.set_ylabel('Probability', fontsize = 20)
    plt.title(title, fontsize=30)
    plt.legend(fontsize = 20)
    
    return kurtosis_d    

