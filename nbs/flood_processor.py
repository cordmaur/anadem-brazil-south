"""FloodProcessor module"""
import io
from pathlib import Path
from typing import Union, Tuple, List, Optional
import unidecode

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

import img2pdf
import skimage

import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray as xrio

from utils import create_fname
from icloud import get_file, icloud_login


class FloodProcessor:
    """FloodProcessor class"""

    VECTORS_FOLDER = "vectors"
    RASTERS_FOLDER = "rasters"
    REPORTS_FOLDER = "reports"

    def __init__(
        self,
        place: pd.Series,
        local_folder: Union[str, Path],
        icloud_folder: str,
        dem: Union[str, Path],
        hand: Union[str, Path],
    ) -> None:
        self.place = place.copy()
        self.icloud_folder = icloud_folder
        self.output_folder = FloodProcessor.create_target_folder(place, local_folder)
        self.session = icloud_login()

        # get the original report
        self.report = self.get_report(icloud_folder)

        self.vars = {}
        self.vars["aoi"] = self.get_aoi(icloud_folder)
        self.vars["floods"] = self.get_floods(icloud_folder)
        self.vars["ref"] = self["floods"]["ref"].where(self["floods"]["ref"] > 0)

        self.vars["dem"] = self.load_window(dem, self["ref"].rio.bounds())
        self.vars["hand"] = self.load_window(hand, self["ref"].rio.bounds())

        self.vars["max_flood"] = self.get_max_flood()

        # create the maximum flood
        self.reshape(["ref", "max_flood", "dem", "hand"], shape=self["dem"].shape)

        try:
            (
                self.vars["flooded_regions"],
                self.vars["labels"],
                self.vars["dem_steps"],
            ) = self.extrapolate_flood()
            floods = np.stack(self["flooded_regions"])
            self.vars["extrapolated_flood"] = self["dem"].copy()
            self.vars["extrapolated_flood"].data = floods.any(axis=0).astype("int")

            # .astype('float')

            # quantify attributes
            self.place["vulnerable_area"] = (
                np.count_nonzero(self["extrapolated_flood"]) * 30 * 30 * 1e-6
            )
            extrapolated_flood = self["ref"].copy()
            extrapolated_flood.data = self["extrapolated_flood"]

            urban_flood = extrapolated_flood.astype("float").rio.clip(
                self["aoi"].geometry, all_touched=True
            )
            urban_flood = urban_flood.where(
                urban_flood != urban_flood.attrs["_FillValue"]
            )
            self.place["urban_area"] = float(urban_flood.count()) * 30 * 30 * 1e-6
            self.place["urban_vulnerable"] = (
                float(urban_flood.where(urban_flood > 0).count()) * 30 * 30 * 1e-6
            )

            self.place["DEM-Status"] = "Ok"

        except Exception as e:  # pylint: disable=broad-except
            self.place["status"] = str(e)

    @staticmethod
    def flood_areas_hand(flood_mask: np.ndarray, dem: np.ndarray, hand: np.ndarray):
        """
        Extrapolate the obtained flood mask according to the DEM and HAND of the area.
        This function operate on a pixel basis (raster), so all paramaters must be given in Numpy Arrays
        with the same shape.
        """
        # isolate all the identified flood areas
        labels = skimage.measure.label(flood_mask)
        # print(f'Number of areas to flood: {labels.max()}')

        # create lists to store the floods for each label and the dem region for each label
        floods = []
        dem_steps = []

        # let's loop through each label (i.e., flood region)
        for label in range(1, labels.max() + 1):  # type: ignore
            # print(f'Processing label {label}')

            # get the flood for the corresponding label (area)
            # and set all other pixels to 0
            flood_step = labels.copy()  # type: ignore
            flood_step[labels != label] = 0

            # get the highest pixel within the area, but try to remove any outlier
            height = np.percentile(dem[labels == label], 95)
            # height = dem[labels==label].max()
            # print(f'Height={height}')

            # create a the DEM-fences with the calculated height
            # this guarantees the flood fill will not go uphill and will not cross boundaries
            dem_step = dem.copy()
            dem_step[dem_step <= height] = 0
            dem_step[dem_step > height] = 1

            # the problem with the last assumption is that the river goes down so the farthest from the fill point
            # a bigger area will be flooded. In this case, we add a second assumption considering the HAND value
            hand_height = np.percentile(hand[labels == label], 95)
            if not np.isnan(hand_height):
                # print(f'Hand height = {hand_height}')
                # dem_step[hand <= hand_height] = 0
                dem_step[hand > hand_height] = 1

                dem_steps.append(dem_step)

                # to flood-fill, we need a starting point, we can get the lowest point with the label
                xs, ys = np.where(labels == label)
                pos = dem[xs, ys].argmin()
                start = (xs[pos], ys[pos])

                # flood fill and get the extended flood for this label
                flood = skimage.morphology.flood_fill(
                    dem_step, seed_point=start, new_value=-1
                )
                flood = np.where(flood == -1, 1, 0)

            else:
                print("No hand available")
                flood = np.where(labels == label, 1, 0)
                dem_steps.append(flood)

            floods.append(flood)

        return floods, labels, dem_steps  # type: ignore

    @property
    def icloud_name(self) -> Path:
        """Return the base name for the icloud files"""
        fs = f"Flood_Report_{self.place.name}_{unidecode.unidecode(self.place['municipio'])}_{self.place['uf']}"
        return Path(fs)

    @staticmethod
    def create_target_folder(place: pd.Series, folder: Union[str, Path]) -> Path:
        """Create a folder to store the output files based on place name"""
        folder = Path(folder)
        place_folder = folder / create_fname(place)
        place_folder.mkdir(parents=True, exist_ok=True)

        return place_folder

    def reshape(
        self, vars_lst: List[str], shape: Tuple[int, int], crs: str = "epsg:4326"
    ):
        """Reproject all variables to the same shape"""
        for var in vars_lst:
            self.vars[var] = self[var].rio.reproject(dst_crs=crs, shape=shape)

    def load_window(self, geotiff_path: Union[str, Path], bounds: tuple):
        """Load a window from a geotiff file"""

        xmin, ymin, xmax, ymax = bounds

        tif = xrio.open_rasterio(geotiff_path).squeeze()  # type: ignore
        window = tif.sel(x=slice(xmin, xmax), y=slice(ymax, ymin)).compute()

        return window.where(window != window.attrs["_FillValue"])

    def save_vars(self, vars_lst: List[str]):
        """Save GeoTiff for each var in the list"""

        for var in vars_lst:
            if var in self.vars:
                tif = self[var]
                name = self.output_folder.stem + "_" + var + ".tif"
                tif.rio.to_raster(self.output_folder / name, compress="DEFLATE")

    def get_max_flood(self):
        """Get the maximum flood in the floods"""
        # create a series with the flood extension
        floods_series = (
            self["floods"]
            .drop("ref")
            .to_array(dim="layer")
            .sum(dim=["x", "y"])
            .to_series()
        )
        max_flood = self["floods"][floods_series.idxmax()]
        # max_flood = max_flood.astype('int').rio.reproject(dst_crs='epsg:4326') #, shape=dem.shape)
        return max_flood.where(max_flood > 0)

    def get_floods(self, icloud_folder: str) -> xr.Dataset:
        """Get the flood raster from the iCloud"""

        floods_name = self.icloud_name.with_suffix(".netcdf")
        remote_folder = self.session.drive[icloud_folder][FloodProcessor.RASTERS_FOLDER]  # type: ignore
        floods_path = get_file(remote_folder, floods_name, self.output_folder)

        floods = xr.load_dataset(floods_path)

        return floods.astype("int").rio.reproject(dst_crs="epsg:4326")

    def get_aoi(self, icloud_folder: str) -> gpd.GeoDataFrame:
        """Get the geojson AOI from the iCloud"""
        aoi_name = self.icloud_name.with_suffix(".geojson")
        remote_folder = self.session.drive[icloud_folder][FloodProcessor.VECTORS_FOLDER]  # type: ignore
        aoi_path = get_file(remote_folder, aoi_name, self.output_folder)  # type: ignore

        aoi = gpd.read_file(aoi_path).to_crs("epsg:4326")
        return aoi

    def get_report(self, icloud_folder: str) -> Path:
        """Get the PDF report for the specific location"""
        report_name = self.icloud_name.with_suffix(".pdf")
        remote_folder = self.session.drive[icloud_folder][FloodProcessor.REPORTS_FOLDER]  # type: ignore

        local_path = get_file(remote_folder, report_name, self.output_folder)
        return local_path

    def plot_var(self, var: str, ax: plt.Axes, **kwargs):
        """Plot a single var within the given axes"""
        self[var].plot(ax=ax, **kwargs)

    def plot_vars(self, ax: plt.Axes, dem: str = "dem") -> None:
        """Plot all variables in the same Axes"""
        self.plot_var(dem, ax=ax, cmap="gist_earth", vmin=0, add_colorbar=False)
        self.plot_var("ref", vmax=1, cmap="Blues", ax=ax, add_colorbar=False)
        self.plot_var("max_flood", vmax=1, cmap="brg", ax=ax, add_colorbar=False)
        self.plot_var("aoi", facecolor="none", edgecolor="white", ax=ax)

    def create_dem_page(self) -> io.BytesIO:
        """Create the PDF page with DEM and HAND to be appended for the report"""

        # create memory-like objects to store the PNG and PDF
        png = io.BytesIO()
        pdf = io.BytesIO()

        # the objective here is to save the figure, so we will be using the Agg backend
        current_backend = mpl.get_backend()

        mpl.use("agg")

        fig, ax = plt.subplots(2, 1, figsize=(12, 23), num=1)
        self.plot_vars(ax=ax[0])
        self.plot_vars(ax=ax[1], dem="hand")
        fig.savefig(png, dpi=150, format="png")
        png.seek(0)
        pdf.write(img2pdf.convert(png))  # type: ignore

        pdf.seek(0)
        plt.close(fig)

        # return the original backend
        mpl.use(current_backend)

        return pdf

    def plot_flood_summary(self, axs: Optional[np.ndarray] = None):
        """Create a plot with the maximum flood vulnerability"""
        if "extrapolated_flood" not in self.vars:
            raise ValueError(
                f"No flood extrapolated for place {self.place['municipio']}"
            )

        mpl.use("agg")

        if axs is None:
            _, axs = plt.subplots(2, 1, figsize=(12, 20))

        ax = axs[1]
        self.plot_var("aoi", ax=ax, facecolor="none", edgecolor="green")
        extrapolated_flood = self["dem"].where(self["extrapolated_flood"]).copy()
        extrapolated_flood.plot.imshow(
            ax=ax, cmap="Reds_r", add_colorbar=False, robust=True
        )
        self.plot_var("ref", ax=ax, cmap="Blues", vmax=1, add_colorbar=False)

        ax = axs[0]
        self.plot_var("dem", ax=ax, cmap="gist_earth", robust=True, add_colorbar=False)
        vulnerable_area = extrapolated_flood.copy()
        vulnerable_area.data[~vulnerable_area.isnull()] = 1
        vulnerable_area.plot(ax=ax, cmap="Wistia", add_colorbar=False)
        self.plot_var("max_flood", vmax=1, cmap="brg", ax=ax, add_colorbar=False)
        self.plot_var("ref", vmax=1, cmap="Blues", ax=ax, add_colorbar=False)
        self.plot_var("aoi", facecolor="none", edgecolor="white", ax=ax)

        return axs[0].figure

    def create_flood_page(self) -> io.BytesIO:
        """Create the PDF page with the extrapolated flood for the report"""

        # first, create the figure
        fig = self.plot_flood_summary()

        # create memory-like objects to store the PNG and PDF
        png = io.BytesIO()
        pdf = io.BytesIO()

        # the objective here is to save the figure, so we will be using the Agg backend
        current_backend = mpl.get_backend()

        mpl.use("agg")

        fig.savefig(png, dpi=150, format="png")
        png.seek(0)
        pdf.write(img2pdf.convert(png))  # type: ignore

        pdf.seek(0)
        plt.close(fig)

        # return the original backend
        mpl.use(current_backend)

        return pdf

    def extrapolate_flood(self, threshold: int = 25):
        """Extrapolate the floods according to the DEM and HAND"""

        # first, we clean the area by removing very small regions
        # our pixel is 30x30m. A threshold of 25 will ensure a minimum area of
        # 2.5ha for each flooded region
        max_flood = np.nan_to_num(self["max_flood"].data, nan=0)
        flood_mask = skimage.morphology.area_opening(
            max_flood, area_threshold=threshold
        )

        # check if there is at least 1 region to process
        if not flood_mask.any():
            raise ValueError(f"No flooded regions considering threshold={threshold}")

        (
            flooded_regions,
            labels,
            dem_steps,
        ) = FloodProcessor.flood_areas_hand(
            flood_mask, self["dem"].data, self["hand"].data
        )

        return flooded_regions, labels, dem_steps

    def urban_flood(self):
        """Calculate the Urban Flooded area"""
        # check if there is flood extrapolated for the current place
        if not hasattr(self, "extrapolated_flood"):
            raise ValueError(
                f"No flood extrapolated for place {self.place['municipio']}"
            )

    def __getitem__(self, idx):
        """Get a variable from the processor"""
        return self.vars[idx]

    def __repr__(self):
        s = "Flood Processor for place: \n"
        s += str(self.place)
        return s
