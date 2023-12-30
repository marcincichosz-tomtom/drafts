from shapely.geometry import LineString, Polygon
import geopandas as gpd
from geopy.distance import geodesic

def split_line_into_segments(line, segment_length_meters):
    """
    Splits a LineString into segments of a specified length.

    Parameters:
    - line: Shapely LineString object representing the trajectory.
    - segment_length_meters: Length of each segment in meters.

    Returns:
    A list of LineString objects representing the segments.
    """
    total_length = line.length
    num_segments = int(total_length / segment_length_meters)

    segments = []
    for i in range(num_segments):
        start_dist = i * segment_length_meters
        end_dist = (i + 1) * segment_length_meters
        start_point = line.interpolate(start_dist)
        end_point = line.interpolate(end_dist)
        segment = LineString([start_point, end_point])
        segments.append(segment)

    return segments

def create_polygon_from_line(line):
    """
    Creates a polygon from a LineString by buffering around it.

    Parameters:
    - line: Shapely LineString object.

    Returns:
    A Polygon object representing the buffered region around the input LineString.
    """
    polygon = line.buffer(0.0001)  # Adjust the buffer distance as needed
    return polygon

def main():
    # Example LineString representing a trajectory
    trajectory = LineString([(0, 0), (1, 1), (2, 0), (3, 1), (4, 0)])

    # Split the trajectory into segments of 500 meters
    segment_length_meters = 500
    line_segments = split_line_into_segments(trajectory, segment_length_meters)

    # Create polygons from line segments
    polygons = [create_polygon_from_line(segment) for segment in line_segments]

    # Display the original trajectory and created polygons
    gdf = gpd.GeoDataFrame(geometry=[trajectory] + polygons)
    gdf.plot()

if __name__ == "__main__":
    main()
