import rasterio
import numpy as np
from rasterio.warp import calculate_default_transform, reproject, Resampling
from PIL import Image

def geotiff_to_pgw_png(input_geotiff, output_pgw, output_png):

    # Read the GeoTIFF file
    with rasterio.open(input_geotiff) as src:
        # Convert the transform to WGS84
        transform = src.transform
        crs = src.crs
        new_crs = rasterio.crs.CRS.from_string("EPSG:4326")  # WGS84

        transform, width, height = calculate_default_transform(crs, new_crs, src.width, src.height, *src.bounds)
        out_image = np.empty((src.count, height, width))

        reproject(
            source=rasterio.band(src, range(1, src.count + 1)),
            destination=out_image,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=new_crs,
            resampling=Resampling.nearest
        )

        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": height,
            "width": width,
            "transform": transform,
            "crs": new_crs
        })

        # Make sure the image array is in the format (height, width, bands)
        if out_image.shape[0] == 1:
            out_image = out_image.reshape((out_image.shape[1], out_image.shape[2]))
        else:
            out_image = np.transpose(out_image, (1, 2, 0))

        # Write the .pgw file
        gdal_coeffs = transform.to_gdal()
        world_file_coeffs = [gdal_coeffs[1], gdal_coeffs[2], gdal_coeffs[4], gdal_coeffs[5], gdal_coeffs[0], gdal_coeffs[3]]
        with open(output_pgw, 'w') as pgw:
            pgw.write('\n'.join([str(i) for i in world_file_coeffs]))

        # Create the .png file
        img = Image.fromarray(out_image.astype('uint8'))
        img.save(output_png)

# Example usage

geotiff_to_pgw_png('/home/marcin/repos/drafts/tiff/utrecht/tile_1.tif', '/home/marcin/repos/drafts/tiff/utrecht/output.pgw', '/home/marcin/repos/drafts/tiff/utrecht/output.png')
