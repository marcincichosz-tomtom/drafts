import rasterio
from pyproj import Proj, transform

def create_pgw_from_geotiff(geotiff_path, pgw_path):
    """
    Create a PGW (World File) from a GeoTIFF file with WGS84 CRS.

    Parameters:
    - geotiff_path (str): Path to the input GeoTIFF file.
    - pgw_path (str): Path to the output PGW file.
    """
    with rasterio.open(geotiff_path) as dataset:
        # Extract information from the GeoTIFF
        transform = dataset.transform
        crs = dataset.crs

        # Check if the CRS is WGS84
        # if crs.to_epsg() != 4326:
        #     raise ValueError("The GeoTIFF must have a WGS84 CRS (EPSG:4326).")

        # Extract pixel size and coordinates from the GeoTIFF transform
        pixel_size_x = transform.a
        pixel_size_y = transform.e
        upper_left_x = transform.c
        upper_left_y = transform.f

    # Create the PGW file content
    pgw_content = f"{pixel_size_x}\n0.0\n0.0\n{-pixel_size_y}\n{upper_left_x}\n{upper_left_y}"
    
    input_crs = Proj(crs)
    output_crs = Proj(init='epsg:4326')  # WGS84
    lon, lat = transform(input_crs, output_crs, upper_left_x, upper_left_y)

    # Print the resulting geographical coordinates
    print("Upper-left corner coordinates:")
    print("Longitude:", lon)
    print("Latitude:", lat)    
    # Write the PGW content to the specified file
    with open(pgw_path, 'w') as pgw_file:
        pgw_file.write(pgw_content)

if __name__ == "__main__":
    # Replace 'path/to/your/input.tif' and 'path/to/your/output.pgw' with the actual paths
    input_geotiff_path = '/home/marcin/repos/drafts/tiff/utrecht/tile_0.tif'
    output_pgw_path = '/home/marcin/repos/drafts/tiff/utrecht/tile_01.pgw'

    create_pgw_from_geotiff(input_geotiff_path, output_pgw_path)
