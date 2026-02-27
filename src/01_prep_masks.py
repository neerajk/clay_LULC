import os
from osgeo import gdal
from tqdm import tqdm

# Enable GDAL to throw Python exceptions instead of silent C++ errors
gdal.UseExceptions()

def prepare_masks():
    # --- DEFINE YOUR PATHS HERE ---
    uk_vec = "../raw_data/uk_shape_files/uttarakhand_Boundary.shp"  # Path to your Uttarakhand shapefile (.shp)
    
    # Mapping years to their respective source .tif files
    years_map = {
        1985: "../raw_data/uk_decadal_LULC/Decadal_LULC_India_1336/data/LULC_1985.tif", # Path to LULC_1985.tif
        #raw_data/uk_decadal_LULC/Decadal_LULC_India_1336/data/LULC_1985.tif
        1995: "../raw_data/uk_decadal_LULC/Decadal_LULC_India_1336/data/LULC_1995.tif", # Path to LULC_1995.tif
        2005: "../raw_data/uk_decadal_LULC/Decadal_LULC_INDIA_1336/data/LULC_2005.tif"  # Path to LULC_2005.tif
    }
    # ------------------------------

    # Ensure output directory exists
    output_dir = "../data/processed/lulc_masks"
    os.makedirs(output_dir, exist_ok=True)
    
    for yr, src_raster in tqdm(years_map.items(), desc="[Task 1/2] Clipping UK LULC Labels"):
        out_raster = os.path.join(output_dir, f"uk_{yr}_30m.tif")
        
        # Validation: Ensure the source path isn't empty and actually exists
        if not src_raster or not os.path.exists(src_raster):
            print(f"\n[SKIP] Source for {yr} not found or path empty: '{src_raster}'")
            continue

        try:
            # Warp: Clip to boundary, reproject to UTM 44N, and force 30m resolution
            gdal.Warp(out_raster, src_raster, options=gdal.WarpOptions(
                cutlineDSName=uk_vec,
                cropToCutline=True,
                dstSRS='EPSG:32644', 
                xRes=30, yRes=30,     
                dstNodata=0,
                multithread=True      
            ))
        except Exception as e:
            print(f"\n[ERROR] Failed processing year {yr}: {e}")

if __name__ == "__main__":
    prepare_masks()