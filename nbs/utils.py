"""Utils"""
from typing import Optional, Union
from pathlib import Path

import unidecode
import numpy as np

import rasterio as rio
from shapely.geometry import Polygon, mapping, shape
import geopandas as gpd
import shapely
import skimage


cache = {}


def get_bbox_and_footprint(dataset):
    """
    This function gets an Rasterio or RioXarray dataset and extracts the bounding box and corresponding footprint
    The Footprint is a Polygon with the dataset boundaries.
    """
    # create the bounding box it will depend if it comes from rasterio or rioxarray
    bounds = dataset.bounds

    if isinstance(bounds, rio.coords.BoundingBox):  # type: ignore
        bbox = [bounds.left, bounds.bottom, bounds.right, bounds.top]
    else:
        bbox = [float(f) for f in bounds()]

    # create the footprint
    footprint = Polygon(
        [[bbox[0], bbox[1]], [bbox[0], bbox[3]], [bbox[2], bbox[3]], [bbox[2], bbox[1]]]
    )

    return bbox, mapping(footprint)


def catalog_to_dataframe(catalog):
    """
    Create a Geopandas Dataframe with the footprints of the items in the catalog
    """
    # first, let's get catalog items
    items = list(catalog.get_all_items())

    # create a GeoDataFrame with the items to perform the intersection
    crs = "epsg:" + str(items[0].assets["DEM"].extra_fields["proj:epsg"])
    gdf = gpd.GeoDataFrame(
        index=[item.id for item in items],
        geometry=[shape(item.geometry) for item in items],
        crs=crs,
    )  # type: ignore

    return gdf


def create_fname(place):
    """Create a file name based on the place"""
    name = f"{unidecode.unidecode(place['municipio'])}_{place['uf']}_{place.name}"
    return Path(name.replace(' ', '_'))


def decode_name(name: str):
    """Eliminate symbols and strange characters from the name"""
    name = unidecode.unidecode(name).replace(' ', '_')
    return name


def create_folder(base_folder: Union[str, Path], name: str):
    """Create the folder to store the reports"""
    name = decode_name(name)
    folder = Path(base_folder)/name
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def search_catalog(catalog, geometry: Optional[shapely.Geometry] = None):
    """
    Search the catalog for items intersecting the given geometry. The intersection uses the asset footprint
    that is stored in the `item.geometry`.
    If no geometry is given, all items in the catalog are returned.
    """
    
    if 'catalog' in cache and catalog == cache['catalog']:
        gdf = cache['gdf']
        
    else:
        gdf = catalog_to_dataframe(catalog)
        
        cache['catalog'] = catalog
        cache['gdf'] = gdf

    gdf_items = gdf[gdf.intersects(geometry)]

    return [catalog.get_item(idx) for idx in gdf_items.index]  # type: ignore


def flood_areas_hand(flood_mask: np.ndarray, dem: np.ndarray, hand: np.ndarray, threshold=50):
    """
    Extrapolate the obtained flood mask according to the DEM and HAND of the area.
    This function operate on a pixel basis (raster), so all paramaters must be given in Numpy Arrays 
    with the same shape.
    """

    if flood_mask.shape == dem.shape == hand.shape:
        pass
    else:
        raise ValueError(f"Arrays must have the same shape: {flood_mask.shape}, {dem.shape}, {hand.shape}.")

    # first, we clean the area by removing very small regions
    flood_mask = skimage.morphology.area_opening(flood_mask, area_threshold=threshold)

    # isolate all the identified flood areas
    labels = skimage.measure.label(flood_mask)
    # print(f'Number of areas to flood: {labels.max()}')

    # create lists to store the floods for each label and the dem region for each label
    floods = []
    dem_steps = []

    # let's loop through each label (i.e., flood region)
    for label in range(1, labels.max()+1): #type: ignore
        # print(f'Processing label {label}')

        # get the flood for the corresponding label (area)
        # and set all other pixels to 0
        flood_step = labels.copy() #type: ignore
        flood_step[labels!=label] = 0

        # get the highest pixel within the area, but try to remove any outlier
        height = np.percentile(dem[labels==label], 95)
        # height = dem[labels==label].max()
        # print(f'Height={height}')

        # create a the DEM-fences with the calculated height
        # this guarantees the flood fill will not go uphill and will not cross boundaries
        dem_step = dem.copy()
        dem_step[dem_step<=height] = 0
        dem_step[dem_step>height] = 1


        # the problem with the last assumption is that the river goes down so the farthest from the fill point
        # a bigger area will be flooded. In this case, we add a second assumption considering the HAND value
        hand_height = np.percentile(hand[labels==label], 95)
        if not np.isnan(hand_height):

            # print(f'Hand height = {hand_height}')
            # dem_step[hand <= hand_height] = 0
            dem_step[hand > hand_height] = 1

            dem_steps.append(dem_step)

            # to flood-fill, we need a starting point, we can get the lowest point with the label
            xs, ys = np.where(labels==label)
            pos = dem[xs, ys].argmin()
            start = (xs[pos], ys[pos])

            # flood fill and get the extended flood for this label
            flood = skimage.morphology.flood_fill(dem_step, seed_point=start, new_value=-1)
            flood = np.where(flood==-1, 1, 0)

        else:
            print(f'No hand available')
            flood = np.where(labels == label, 1, 0)
            dem_steps.append(flood)

        floods.append(flood)

    return floods, labels, dem_steps, hand_height #type: ignore