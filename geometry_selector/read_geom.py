import os
import pyspark
from shapely import wkb, wkt

# Definiuj ścieżkę do folderu root
root_dir = 'lbem-sessions-utrecht-city-block_infer'
spark = pyspark.sql.SparkSession.builder.appName("SparkSQL").getOrCreate()
spark.conf.set("spark.sql.parquet.enableVectorizedReader", "false")
# Przejrzyj wszystkie foldery i podfoldery od zadanego folderu root
dirpath, sessions, filenames = os.walk(root_dir).__next__()
for session_name in sessions:
    for lane_type in ['lane_marking_dashed', 'lane_marking_solid']:
        # Utwórz pełną ścieżkę do podfolderu
        subfolder_path = os.path.join(dirpath, session_name, lane_type)
        print(subfolder_path)
        # Wczytaj dane w formacie spark delta table
        delta_table = spark.read.load(subfolder_path)
        pandas_df = delta_table.toPandas()
        # Wyeksportuj dane do pliku csv
        pandas_df['geometry'] = pandas_df['geometry'].apply(lambda x: wkt.dumps(wkb.loads(bytes(x))) if x is not None else None)
        pandas_df.to_csv(os.path.join('csv',f'{session_name}_{lane_type}.csv'), index=False)
