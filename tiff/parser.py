import rasterio

# Specify the path to your GeoTIFF file
geotiff_path = '/home/marcin/repos/drafts/tiff/utrecht/tile_0.tif'

# Open the GeoTIFF file
with rasterio.open(geotiff_path) as dataset:
    # Get basic information about the GeoTIFF
    print("Driver:", dataset.driver)
    print("Width:", dataset.width)
    print("Height:", dataset.height)
    print("Number of Bands:", dataset.count)

    # Access metadata
    print("Metadata:", dataset.meta)

    # Access geospatial information
    print("CRS (Coordinate Reference System):", dataset.crs)
    print("Transform (Affine transform parameters):", dataset.transform)

    # Read raster data as a NumPy array
    raster_data = dataset.read(1)  # Change the band index as needed

# Now you have the raster data in the variable 'raster_data' as a NumPy array
# You can perform various operations or visualization with the raster data.
