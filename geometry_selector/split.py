import os
from datetime import datetime

from shapely.geometry import LineString, box, Polygon
from shapely.ops import unary_union, transform, split, linemerge
from shapely.ops import cascaded_union
from shapely import wkt
import pyproj

import pandas as pd
import geopandas as gpd

import multiprocessing
from multiprocessing import Manager

import math
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

POLYGON_BUFFER = 2.0


def convert_wgs84_to_utm(wkt_geometry):
    try:
        # Parse the WKT geometry
        geometry = wkt.loads(wkt_geometry)

        # Define the WGS84 and UTM coordinate systems
        wgs84 = pyproj.CRS("EPSG:4326")  # WGS84 coordinate system
        utm = pyproj.CRS("EPSG:32633")  # UTM coordinate system, Zone 33 for example

        # Create a transformer for the coordinate conversion
        transformer = pyproj.Transformer.from_crs(wgs84, utm, always_xy=True)

        # Use the transformer to convert the geometry
        transformed_geometry = transform(transformer.transform, geometry)

        return transformed_geometry

    except Exception as e:
        print(f"Error: {e}")
        return None

def extract_datetime_from_string(input_string):
    try:
        # Extract date and time parts from the string
        date_string = input_string.split('_')[1:4]
        time_string = input_string.split('_')[5:8]
        date_time_format = "%Y_%m_%d_%H_%M_%S"
        
        # Join the date and time parts into a formatted datetime string
        formatted_datetime_string = "_".join(date_string + time_string)
        
        # Convert the formatted datetime string to a datetime object
        extracted_datetime = datetime.strptime(formatted_datetime_string, date_time_format)
        
        return extracted_datetime

    except Exception as e:
        print(f"Error: {e}")
        return None

def read_and_create_geodf(csv_file_path, session_names, number_of_sessions=0):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file_path)

    # Convert the geometry column to shapely.geometry objects
    #df['geometry'] = df['geometry'].apply(wkt.loads)
    df['geometry'] = df['geometry'].apply(convert_wgs84_to_utm)
    # Create a GeoDataFrame from the pandas DataFrame
    geodf = gpd.GeoDataFrame(df, geometry='geometry')
    geodf['timestamp'] = geodf['sessionname'].apply(extract_datetime_from_string)
    geodf = geodf.sort_values(by=['timestamp'], ascending=False)
    
    if session_names is not None:
        # Get the unique session names
        filtered_geodf = geodf[geodf['sessionname'].isin(session_names)]
    else:
        filtered_geodf = geodf.iloc[0:number_of_sessions]
        
    return filtered_geodf

def splitting(supporting_session_traj_wkt, buffer, reference_segment_wkt):
    supporting_session_traj = wkt.loads(supporting_session_traj_wkt)
    reference_segment = wkt.loads(reference_segment_wkt)
    
    poly = reference_segment.buffer(buffer)
    split_lines = split(supporting_session_traj, poly)
    return poly, split_lines

def split_and_classify2(combined_polygon, trajectory):

    split_lines = split(trajectory.geometry, combined_polygon)
    inside_lines = []
    outside_lines = []
    for s_line in split_lines:
        if combined_polygon.contains(s_line):
            inside_lines.append(s_line)
        else:
            outside_lines.append(s_line)
    return inside_lines, outside_lines


def split_and_classify(supporting_session_traj, map_segment, buffer):
    if map_segment is not None:
        session_name, count, bearing, reference_segment = map_segment
        poly, split_lines = splitting(supporting_session_traj.wkt, buffer, reference_segment.wkt)
        inside_lines = []
        outside_lines = []
        for s_line in split_lines:
            if poly.contains(s_line):
                inside_lines.append(s_line)
            else:
                outside_lines.append(s_line)
        return inside_lines, outside_lines
    else:
        return [], []

def split_linestring_into_segments(linestring):
    # Initialize an empty list to hold the line segments
    segments = []
    
    # Get the points of the linestring
    points = list(linestring.coords)

    # Iterate over pairs of points (each pair forms a line segment)
    for i in range(len(points) - 1):
        segment = LineString([points[i], points[i+1]])
        segments.append(segment)

    return segments

def draw_line(line, color='black', label=None):
    x, y = line.xy
    plt.plot(x, y, color=color)
    
    # Add label if provided
    if label is not None:
        mid_point_index = len(x) // 2
        plt.text(x[mid_point_index], y[mid_point_index], label, fontsize=12)

def draw_polygon(polygon, color='black', fill_color='', label=None):
    x, y = polygon.exterior.xy
    #plt.fill(x, y, fill_color, alpha=0.3)  # fill the polygon with color
    plt.plot(x, y, color=color)  # draw the polygon outline
    
    # Add label if provided
    if label is not None:
        centroid = polygon.centroid
        plt.text(centroid.x, centroid.y, label, fontsize=12)

def calculate_bearing(point1, point2):
    lon1 = math.radians(point1[0])
    lon2 = math.radians(point2[0])
    lat1 = math.radians(point1[1])
    lat2 = math.radians(point2[1])
     
    dLon = lon2 - lon1
    dLat = lat2 - lat1
    #bearing = math.atan2(dLon, dLat)
    y = math.sin(dLon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dLon)
    bearing = (math.degrees(math.atan2(y, x)) + 360) % 360
    return bearing

def calculate_bearing_for_segments(segments):
    bearings = []
    for segment in segments:
        coords = list(segment.coords)
        bearing = calculate_bearing(coords[0], coords[1])
        bearings.append(bearing)
    return bearings

def simplify_linestring_by_bearing(session_name, linestring, bearing_difference_threshold=10):
    # Split the LineString into segments
    segments = split_linestring_into_segments(linestring)

    # Calculate the bearing for each segment
    bearings = calculate_bearing_for_segments(segments)

    # Initialize list to hold the line sections
    line_sections = []

    # Iterate over the segments and split the line based on bearing difference
    current_section = [segments[0]]
    bearings_sum = bearings[0]
    bearing_count = 1
    for i in range(1, len(segments)):
        # Calculate the difference in bearing between the current segment and the previous one
        bearing_difference = abs(bearings[i] - (float(bearings_sum)/float(bearing_count)))

        # If the bearing difference is greater than the threshold, split the line
        if bearing_difference > bearing_difference_threshold:
            # Add the current section to the list of line sections
            line_sections.append((session_name, bearing_count, round(bearings_sum / bearing_count,1), LineString([point for segment in current_section for point in segment.coords])))
            # Start a new section with the current segment
            current_section = [segments[i]]
            bearings_sum = bearings[i]
            bearing_count = 1
        else:
            # Otherwise, add the current segment to the current section
            current_section.append(segments[i])
            bearings_sum += bearings[i]
            bearing_count += 1
            
    
    # Add the last section to the list of line sections
    line_sections.append((session_name, bearing_count, round(bearings_sum / bearing_count,1), LineString([point for segment in current_section for point in segment.coords])))

    return line_sections

def split_list_of_tuples(data, chunk_size):
    chunks = []
    if not data:
        return []

    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        chunks.append(chunk)
    return chunks

def join_linestrings(linestrings):

    multiline = linemerge(linestrings)
    if multiline.geom_type == 'LineString':
        return multiline
    else:
        return list(multiline)

def _merge_trajectories(map_segments, polygon_buffer, row, inside_lines_global, outside_lines_global): 
    
    merged_linestrings = join_linestrings([linestring for (_,_,_,linestring) in map_segments])
    
    disjoint_count = 0
    for index, map_segment in enumerate(merged_linestrings): 
        inside_lines, outside_lines = split_and_classify(row.geometry, map_segment, polygon_buffer) 
        if inside_lines == []:
            disjoint_count+=1
        inside_lines_global.append(inside_lines) if len(inside_lines) > 0 else None 
        outside_lines_global.append(outside_lines) if len(outside_lines) > 0 else None 

    # return inside_lines_global, outside_lines_global
    return [], []

def find_outside_lines_depricated(map_segments, polygon_buffer, row):
    with Manager() as manager:
        inside_lines_global = manager.list()
        outside_lines_global = manager.list()

        cpu = 1 #multiprocessing.cpu_count()
        print(int(len(map_segments)/cpu))
        processes = split_list_of_tuples(map_segments, int(len(map_segments)/cpu))

        processes = [multiprocessing.Process(target=_merge_trajectories, args=(process, polygon_buffer, row, inside_lines_global, outside_lines_global)) for process in processes]

        for process in processes:
            process.start()

        for process in processes:
            process.join()

        # Convert back to normal list
        inside_lines_global = list(inside_lines_global)
        outside_lines_global = list(outside_lines_global)

        return inside_lines_global, outside_lines_global

def construct_bbox_and_grid(gdf, cell_size=20):
    # Create a GeoDataFrame from the input data
    bbox = unary_union(gdf.geometry).bounds  
    grid = []
    # Loop through the extents of bbox, creating a grid
    x_min, y_min, x_max, y_max = bbox
    ids = []
    for x_id, x in enumerate(range(int(x_min), int(x_max), cell_size)):
        for y_id, y in enumerate(range(int(y_min), int(y_max), cell_size)):
            cell = box(x, y, x+cell_size, y+cell_size)
            intersecting_sessions = gdf[gdf.geometry.apply(lambda geom: cell.intersects(geom))]
            if intersecting_sessions.shape[0]>0:  # if there is at least one intersecting session
                grid.append((f"{x_id}_{y_id}", 
                             '|'.join(intersecting_sessions.session_name.tolist()),
                             '|'.join([str(item) for item in intersecting_sessions.bearing.tolist()]),
                             cell))
    
    # Create DataFrame
    df = pd.DataFrame(grid, columns=['id', 'sessions', 'bearings', 'geometry'])            
    return df

def visualize_grid_and_linestrings(grid, linestrings):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Draw each cell in the grid
    for cell in grid:
        x,y = cell.exterior.xy
        ax.plot(x, y, color='#6699cc', alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)
    
    # Draw each linestring
    for linestring in linestrings:
        x, y = linestring.xy
        ax.plot(x, y, color='#FF0000', alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)
    
    ax.set_aspect('equal', 'box')
    plt.show()

def visualize_grid_gdf(grid_gdf):
    # Create a new column for the number of intersecting sessions
    grid_gdf['num_sessions'] = grid_gdf['sessions'].apply(lambda x: len(x.split('|')))

    fig, ax = plt.subplots()

    # Plot the grid cells, coloring by the number of intersecting sessions
    grid_gdf.plot(column='num_sessions', 
                  legend=True, 
                  ax=ax, 
                  cmap='viridis', 
                  linewidth=0.8)

    plt.show()

def visualize_lines_with_buffer(map_segments):
    for index, (session_name, count, bearing, segment) in enumerate(map_segments):
        polygon = segment.buffer(POLYGON_BUFFER)
        draw_polygon(polygon, 'orange', 'yellow')  # , f'{index}_{count}_{bearing}' orange outline and yellow fill for the polygon
        draw_line(segment, 'red')
                
    plt.show()

def save_map_segments(map_segments, session_name):
    map_segments_df = pd.DataFrame(map_segments, columns=['session_name', 'count', 'bearing', 'geometry'])
    map_segments_df.to_csv(os.path.join('map_segments', f'{session_name}.csv'), index=False)

def add_geometries(session_name, polygon_increment):
    # read geometries from csv file if exists session_name.csv
    path = os.path.join('csv', f'{session_name}_lane_marking_dashed.csv')
    if os.path.exists(path):
        lane_marking_dashed_df = pd.read_csv(path)
    
    lane_marking_dashed_df['geometry'] = lane_marking_dashed_df['geometry'].apply(convert_wgs84_to_utm)
    
    # Create a GeoDataFrame from the pandas DataFrame
    geodf = gpd.GeoDataFrame(lane_marking_dashed_df, geometry='geometry')
  
    # Create a Polygon object from the increment
    clip_polygon = Polygon(polygon_increment)
  
    # Clip the GeoDataFrame with the Polygon
    clipped_geodf = gpd.clip(geodf, clip_polygon)
    
    return clipped_geodf

def join_polygons(polygons):

    polygon_objects = [Polygon(p) for p in polygons]
    combined_polygon = cascaded_union(polygon_objects)
    
    return combined_polygon

def select_geometries_based_on_latest_sessions():
    
    total_geodf = gpd.GeoDataFrame()
    external_lines = []
   
    combined_polygon = None
    for row in result_geodf.itertuples():
        # cache
        start_time = datetime.now()
        print(f'session_name={row.sessionname}')
        print(f'map_segments={len(external_lines)}')
        if external_lines == []:
            external_lines = [row.geometry]
        else:
            # find outside lines based on previous map_segments and new session trajectory
            _, external_lines = split_and_classify2(combined_polygon, row)

        # create outside_segments based on outside lines
        polygons = []
        print(len(external_lines))                    
        for line in external_lines:
            #print(f'outside_lines={line[0]}')
            external_segments = simplify_linestring_by_bearing(row.sessionname, line, bearing_difference_threshold=15)
            polygons_new = [segment.buffer(POLYGON_BUFFER) for _, _, _, segment in external_segments]
            tmp_gdf = add_geometries(row.sessionname, join_polygons(polygons_new))
            total_geodf = pd.concat([total_geodf, tmp_gdf])
            
            polygons += polygons_new
        
        combined_polygon = join_polygons(polygons)
        end_time = datetime.now()        
        print(f'Execution time: {end_time - start_time}')

    total_geodf.reset_index(drop=True, inplace=True)
    total_geodf.to_csv('geom_gdf.csv', index=False)


if __name__ == "__main__":
    
    # Example usage
    # measure time of execution
    start_time = datetime.now()
    csv_file_path = 'geometry_selector/utrecht_session_traj.csv'
    session_names = ['EL6W028_2022_05_21__07_54_51', 'EL6W028_2022_05_21__08_54_51', 'WY6796L_2020_04_13__13_49_43']
    result_geodf = read_and_create_geodf(csv_file_path, None, 6)
    #print(result_geodf)
    
    select_geometries_based_on_latest_sessions()
    end_time = datetime.now()
    print(f'Total execution time: {end_time - start_time}')
        
    #polygon = segment.buffer(buffer)
    # for index, (session_name, count, bearing, segment) in enumerate(map_segments):
    #     polygon = segment.buffer(POLYGON_BUFFER)
    #     draw_polygon(polygon, 'orange', 'yellow')  # , f'{index}_{count}_{bearing}' orange outline and yellow fill for the polygon
    #     draw_line(segment, 'red', f'{session_name[-8:]}')
        
    # plt.show()
    
    # Construct a grid of cells
    # bbox, grid = construct_bbox_and_grid([segment for _, _, _, segment in map_segments], cell_size=40)
    # visualize_grid_and_linestrings(grid, [segment for _, _, _, segment in map_segments])
    # map_segments_df['geometry'] = map_segments_df['geometry'].apply(wkt.loads)
    # map_segments_geodf = gpd.GeoDataFrame(map_segments_df, geometry='geometry')
    
    # if os.path.exists('grid.csv'):
    #     grid_df = pd.read_csv('grid.csv')
    # else:
    #     grid_df = construct_bbox_and_grid(map_segments_geodf, cell_size=80)
    #     grid_df.to_csv('grid.csv', index=False)
    
    
    # visualize_grid_gdf(grid_df)
    