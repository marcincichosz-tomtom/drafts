import rasterio
from rasterio.transform import from_origin

# The metadata dictionary you provided
metadata = {
    'driver': 'GTiff',
    'dtype': 'float32',
    'nodata': None,
    'width': 2000,
    'height': 2000,
    'count': 1,
    'crs': rasterio.crs.CRS.from_wkt('PROJCS["unknown",GEOGCS["unknown",DATUM["Unknown based on WGS84 ellipsoid",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",52.1037574429941],PARAMETER["central_meridian",5.08257671725096],PARAMETER["scale_factor",1],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH]]'),
    'transform': rasterio.transform.Affine(0.08, 0.0, 622.3981163733292, 0.0, -0.08, -2240.209401622186)
}

# Extract pixel size and coordinates from the GeoTIFF metadata
pixel_size_x = metadata['transform'][0]
pixel_size_y = -metadata['transform'][4]  # Negative because the transform has a negative scale for the y-axis
upper_left_x = metadata['transform'][2]
upper_left_y = metadata['transform'][5]

# Create the PGW file content
pgw_content = f"{pixel_size_x}\n0.0\n0.0\n{pixel_size_y}\n{upper_left_x}\n{upper_left_y}"

# Specify the path to the output PGW file
pgw_path = '/home/marcin/repos/drafts/tiff/utrecht/tile_0.pgw'

# Write the PGW content to the specified file
with open(pgw_path, 'w') as pgw_file:
    pgw_file.write(pgw_content)
