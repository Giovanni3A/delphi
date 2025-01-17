import numpy as np
import pandas as pd
import geopandas as gpd

from typing import List
from shapely.geometry import Polygon, MultiPolygon

from .time_discretization_utils import calculate_seasonality, apply_custom_time_events
from .squares import rectangle_discretization
from .h3_utils import generate_H3_discretization
from .add_regressors import addRegressorUniformDistribution


class DataAggregator():

    def __init__(self, crs: str = 'epsg:4326'):
        '''
        Class initialization.

        Parameters
        ----------
        crs : The value can be any accepted by pyproj.CRS.from_user_input()
            Coordinate Reference System to be assured and applied.
        '''
        self.crs = crs
        self.time_indexes = []
        self.geo_index = 'gdiscr'
        self.events_features = []
        self.geo_features = []

        self.max_borders = None
        self.events_data = None
        self.geo_discretization = None

    def add_max_borders(self, data: gpd.GeoDataFrame = None, method: str = None):
        '''
        Adds limit borders to internal map. Will be used on multiple
        methods to limit events and discretization reach.

        Parameters
        ----------
        data : geopandas.GeoDataFrame
            GeoDataFrame containing Polygon representing limit borders

        method : str
            Method of estimating borders using events data:

                rectangle => Calculates smallest rectangle containing all event points
                convex => Calculates aproximate of smallest convex polygon containing all event points
        '''

        if data is not None:
            # If no CRS set, will assume class set CRS
            if data.crs is None:
                print(f'''Limit borders data is not set with CRS information. Will assume "{self.crs}".''')
                self.max_borders = data

            # If CRS indetified, set new one if needed
            elif data.crs != self.crs:
                self.max_borders = data.to_crs(self.crs)

            else:
                self.max_borders = data

        elif method == 'rectangle':

            # if no events data was passed, raise error
            if not(hasattr(self, "events_data")):
                raise AttributeError('Please inform events data using `add_events_data` method.')

            # get limits and compute rectangle
            minx, miny, maxx, maxy = self.events_data.geometry.total_bounds
            pol_max_borders = Polygon([
                (minx, miny), (minx, maxy),
                (maxx, maxy), (maxx, miny)
            ])
            # replace self.max_borders
            self.max_borders = gpd.GeoDataFrame()
            self.max_borders['geometry'] = [pol_max_borders]
            self.max_borders = self.max_borders.set_crs(self.crs)

        elif method == 'convex':

            # if no events data was passed, raise error
            if not(hasattr(self, "events_data")):
                raise AttributeError('Please inform events data using `add_events_data` method.')

            x_std = self.events_data.geometry.x.std()*.1
            y_std = self.events_data.geometry.y.std()*.1
            # concatenate list of mini event-polygons
            d_polys = self.events_data\
                .drop(self.events_data.loc[self.events_data.geometry.x.isna()].index)\
                .geometry.apply(lambda pt: Polygon([
                    (pt.x - x_std, pt.y - y_std),
                    (pt.x - x_std, pt.y + y_std),
                    (pt.x + x_std, pt.y + y_std),
                    (pt.x + x_std, pt.y - y_std)
                ]) if pt.x >= -np.inf else pt)
            # replace self.max_borders
            self.max_borders = gpd.GeoDataFrame()
            self.max_borders['geometry'] = [MultiPolygon(list(d_polys.values)).convex_hull]
            self.max_borders = self.max_borders.set_crs(self.crs)

        else:
            raise ValueError('Please inform `data` or `method` parameter.')

    def add_events_data(
        self,
        events_data: pd.DataFrame,
        datetime_col: str = None,
        lat_col: str = 'lat',
        lon_col: str = 'lon',
        feature_cols: List[str] = [],
        datetime_format: str = None
    ):
        '''
        Processes and save events registers.

        Parameters
        ----------
        events_data : pandas.DataFrame, geopandas.GeoDataFrame
            DataFrame containing latitude longitude information
            -> If pandas dataframe, should contain two
            columns (lat_col, lon_col) with latitude and longitude
            coordinates
            -> If geopandas geodataframe, should contain "geometry" column

        datetime_col : str
            Column containing time information from events
            Can be not informed if not applicable

        lat_col : str
            Column containing latitude from events

        lon_col : str
            Column containing longitude from events

        feature_cols : List[str]
            Column(s) containing adicional information from events
            that will be considered in discretization process

        datetime_format : str
            The strftime to parse time, e.g. "%d/%m/%Y".
            See https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
            for reference
        '''
        # configure geodataframe
        # if pandas dataframe type, will create geometry columns
        if type(events_data) == pd.DataFrame:
            events_data['geometry'] = gpd.points_from_xy(
                events_data[lon_col],
                events_data[lat_col]
            )
            self.events_data = gpd.GeoDataFrame(events_data)\
                .set_crs('epsg:4326')\
                .to_crs(self.crs)
        # if geopandas object, will assure correct CRS
        else:
            if events_data.crs is None:
                self.events_data = events_data.set_crs(self.crs)
            else:
                self.events_data = events_data.to_crs(self.crs)

        # transform datetime_col data to datetime
        if datetime_col is not None:
            ts = self.events_data[datetime_col]
            if ts.dtype == 'O':
                self.events_data['ts'] = pd.to_datetime(ts, format=datetime_format, infer_datetime_format=True)
            else:
                self.events_data['ts'] = ts.copy()
            self.events_data.sort_values('ts', ascending=True, inplace=True)

        # filter only necessary cols
        self.events_features += feature_cols
        special_cols = ['geometry', 'ts'] if datetime_col is not None else ['geometry']
        self.events_data = self.events_data[
            feature_cols + special_cols
        ].copy()

    def add_time_discretization(
        self,
        seasonality_type: str,
        window: int = 1,
        frequency: int = 1
    ):
        '''
        Applies seasonality index as time discretization new column.

        Parameters
        ----------
        seasonality_type: str or DataFrame
            If string, represents seasonality frequency type
                Y -> year
                M -> month
                W -> week
                D -> day
                H -> hour
                m -> minute
                S -> second

            If dataframe, represents all necessary information for each event.

        window: (List[int], int)
            Seasonality window. Can be unique, passed as an int, or variable,
            passed as a list of int.

        frequency: int
            Seasonality frequency.

            eg: If seasonality is between 12 months in a year:
                seasonality_type = 'Y'
                window = 1
                frequency = 12

                If time discretization must consider the first 12 hours in a day as first index,
                next 8 hours as second and last 4 hours as third:
                seasonality_type = 'D'
                window = [12, 8, 4]
                frequency = 24
        '''
        # throw error if events_data is None
        if self.events_data is None:
            raise AttributeError('Please inform events data using `add_events_data` method.')

        # get new column name
        i_col = 1
        while True:
            t_col = f'tdiscr_{i_col}'
            if t_col in self.events_data.columns:
                i_col += 1
            else:
                break

        # if dataframe passed, apply custom events time discretization
        if type(seasonality_type) == pd.DataFrame:
            self.time_indexes.append(t_col)
            self.events_data[t_col] = apply_custom_time_events(
                ts=self.events_data['ts'],
                time_disc_df=seasonality_type,
                nan_idx=0
            )

        else:
            # throw error if frequency is not compatible with window
            if (type(window) is int) and (frequency % window != 0):
                raise ValueError('Parameter `frequency` must be a multiple of `window`.')
            if type(window) is list and sum(window) != frequency != 0:
                raise ValueError('Parameter `frequency` must be equal sum of `window`.')

            # add new column
            self.time_indexes.append(t_col)
            self.events_data[t_col] = calculate_seasonality(
                ts=self.events_data['ts'],
                seasonality_type=seasonality_type,
                window=window,
                frequency=frequency
            )

    def add_geo_discretization(
        self,
        discr_type: str,
        hex_discr_param: int = None,
        rect_discr_param_x: int = None,
        rect_discr_param_y: int = None,
        custom_data: gpd.GeoDataFrame = None
    ) -> gpd.GeoDataFrame:
        '''
        Calculates geographical discretization and apply index
        to events data.

        Parameters
        ----------
        discr_type : str
            Indicator of method of geographical discretization.

                R => Rectangular discretization. Parameters `rect_discr_param_x`
                and `rect_discr_param_y` are required.

                H => Hexagonal discretization using H3 package. Parameter
                `hex_discr_param` is required.

                C => Custom discretization using user GeoDataFrame. Parameter
                `custom_data` is required.

                G => Custom discretization using graph data. Geolocated nodes
                must be informed in dataframe format `custom_data`.

        hex_discr_param : int
            Granularity level for hexagonal discretization.

        rect_discr_param_x : int
            Granularity level for rectangular horizontal discretization.

        rect_discr_param_y : int
            Granularity level for rectangular vertical discretization.
        '''

        # if there is no max borders set throw error
        if self.max_borders is None:
            raise ValueError('Please inform geographical limits using `add_max_borders` method.')

        # square discretization
        if discr_type == 'R':
            self.geo_discretization = rectangle_discretization(
                gdf=self.max_borders,
                nx=rect_discr_param_x,
                ny=rect_discr_param_y
            )

        # hexagonal discretization
        elif discr_type == 'H':
            full_area = self.max_borders.copy()  # keep original
            full_area.geometry = full_area.geometry.convex_hull  # apply convex hull so no area is cutted out
            self.geo_discretization = generate_H3_discretization(self.max_borders, hex_discr_param)

            # cut hexagons borders that dont belong to original shape
            limited_discretization = gpd.overlay(
                self.geo_discretization[['id', 'geometry']],
                self.max_borders[['geometry']],
                how='intersection'
            ).dissolve(by='id').reset_index()
            self.geo_discretization = pd.merge(
                self.geo_discretization.drop('geometry', axis=1),
                limited_discretization,
                on='id',
                how='right'
            )
            # return type to geoDataFrame
            self.geo_discretization = gpd.GeoDataFrame(self.geo_discretization)

            # correct polygons index, ordering in neighbors list
            corr_index_dict = self.geo_discretization.reset_index().set_index('id')['index'].to_dict()
            neighbors_list = list(self.geo_discretization['neighbors'])
            for ind, n_list in enumerate(neighbors_list):
                new_list = []
                for n in n_list:
                    if corr_index_dict.get(n):
                        new_list.append(corr_index_dict[n])
                neighbors_list[ind] = new_list
            self.geo_discretization['neighbors'] = neighbors_list
            self.geo_discretization.drop('id', axis=1, inplace=True)
            self.geo_discretization.reset_index(inplace=True)
            self.geo_discretization.rename({'index': 'id'}, axis=1, inplace=True)

        elif discr_type == 'C':
            # check for custom_data
            if custom_data is None:
                raise ValueError('Please inform `custom_data` to be used as geographical discretization.')
            else:
                self.geo_discretization = custom_data[['geometry']].copy()
                self.geo_discretization['id'] = list(range(len(custom_data)))

                # If no CRS set, will assume class set CRS
                if self.geo_discretization.crs is None:
                    print(f'Custom data is not set with CRS information. Will assume "{self.crs}".')
                    self.geo_discretization = self.geo_discretization.set_crs(self.crs)

                # If CRS indetified, set new one if needed
                elif self.geo_discretization.crs != self.crs:
                    self.geo_discretization = self.geo_discretization.to_crs(self.crs)

            # compute neighbors
            neighbors = []
            for _, row in self.geo_discretization.iterrows():
                row_neighbors = list(self.geo_discretization[
                    ~self.geo_discretization.geometry.disjoint(row.geometry)
                ]['id'])
                row_neighbors = [n for n in row_neighbors if n != row['id']]
                neighbors.append(row_neighbors)
            self.geo_discretization['neighbors'] = neighbors

        elif discr_type == "G":
            # check for custom_data
            if custom_data is None:
                raise ValueError('Please inform `custom_data` to be used as grpah discretization.')
            else:
                self.geo_discretization = custom_data[['geometry']].copy()
                self.geo_discretization['id'] = list(range(len(custom_data)))

                # If no CRS set, will assume class set CRS
                if self.geo_discretization.crs is None:
                    print(f'Custom data is not set with CRS information. Will assume "{self.crs}".')
                    self.geo_discretization = self.geo_discretization.set_crs(self.crs)

                # If CRS indetified, set new one if needed
                elif self.geo_discretization.crs != self.crs:
                    self.geo_discretization = self.geo_discretization.to_crs(self.crs)

            # join events with nodes using nearest node
            self.events_data = gpd.sjoin_nearest(
                self.events_data.to_crs("epsg:29193"),
                self.geo_discretization.to_crs('epsg:29193')[["id", "geometry"]],
                how="left"
            )\
                .to_crs(self.crs)\
                .drop("index_right", axis=1)\
                .rename(columns={"id": "node_id"})

            # create map from node_id to gdiscr index
            node_idx_dict = dict(enumerate(self.events_data["node_id"].dropna().unique()))
            node_idx_dict = {v: k for k, v in node_idx_dict.items()}
            self.events_data["gdiscr"] = self.events_data["node_id"].map(node_idx_dict)
            self.events_data.drop("node_id", axis=1, inplace=True)

            # drop duplicated events
            self.events_data = self.events_data.reset_index()\
                .drop_duplicates(subset=["index"], keep="first")\
                .set_index("index")

            # not consider events outside max borders
            self.events_data.loc[
                ~self.events_data["geometry"].dropna().within(
                    self.max_borders["geometry"].values[0]
                ), "gdiscr"] = pd.NA

            # column gdiscr is alread copmuted so function must be terminated
            return None

        else:
            raise ValueError(f'Invalid `discr_type` value {discr_type}.')

        # fills center_lat and center_lon (using projected crs for distance precision)
        centroids = self.geo_discretization.to_crs('epsg:4088').geometry.centroid
        self.geo_discretization['center_lat'] = centroids.x
        self.geo_discretization['center_lon'] = centroids.y

        # apply index to events data
        self.events_data = gpd.sjoin(
            self.events_data.drop('gdiscr', axis=1, errors='ignore'),
            self.geo_discretization[['geometry', 'id']],
            how='left',
            op='within'
        ).drop('index_right', axis=1)\
            .rename({'id': 'gdiscr'}, axis=1)

    def add_geo_variable(self, data: gpd.GeoDataFrame):
        '''
        Merge information from external geographical data with computed
        geographical discretization.

        Parameters
        ----------
        data: geopandas.GeoDataFrame
            Geodataframe with desired variables.
        '''

        # If no CRS set, will assume class set CRS
        if data.crs is None:
            print(f'Regressor data is not set with CRS information. Will assume "{self.crs}".')
            regr_data = data.set_crs(self.crs)

        # If CRS indetified, set new one if needed
        else:
            regr_data = data.to_crs(self.crs)

        # add features to geo_features list
        self.geo_features += list(data.drop(['geometry'], axis=1).columns)

        # Calculate intersection between regressor data and geo discretization.
        # For math precision, if current crs is not projected,
        # will transform into proper (projected) CRS before using area metric
        # to calcultate intersections
        geo_disc = self.geo_discretization[['id', 'geometry']].copy()
        if not(self.geo_discretization.crs.is_projected):
            geo_disc = geo_disc.to_crs('epsg:4088')
            regr_data = regr_data.to_crs('epsg:4088')

        regr_intersection = addRegressorUniformDistribution(
            geo_disc,
            regr_data,
            discr_id_col='id'
        )
        self.geo_discretization = pd.merge(
            self.geo_discretization,
            regr_intersection.drop('geometry', axis=1),
            on='id',
            how='left'
        )
