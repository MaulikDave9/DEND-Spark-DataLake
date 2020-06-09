import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, row_number
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format
from pyspark.sql.window import Window

# Read AWS credentials
config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID']=     config['AWS']['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY']= config['AWS']['AWS_SECRET_ACCESS_KEY']

"""
Creating a Apache spark session on AWS to process the input data.
Output: Spark session.
"""
def create_spark_session():
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .config("spark.hadoop.fs.s3a.impl","org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3a.awsAccessKeyId", os.environ['AWS_ACCESS_KEY_ID']) \
        .config("spark.hadoop.fs.s3a.awsSecretAccessKey", os.environ['AWS_SECRET_ACCESS_KEY']) \
        .getOrCreate()
    
    sc = spark.sparkContext
    sc._jsc.hadoopConfiguration().set("fs.s3a.access.key", config['AWS']['AWS_ACCESS_KEY_ID'])
    sc._jsc.hadoopConfiguration().set("fs.s3a.secret.key", config['AWS']['AWS_SECRET_ACCESS_KEY'])
    
    return spark


"""
Processing Song data.
Input: spark session, input data to extract song_data, output path to write parquet files
Output: Write songs and artist table as parquet file to given output path
"""
def process_song_data(spark, input_data, output_data):

    # get filepath to song data file
    song_data = input_data
    #song_data  = 'data/song-data/*/*/*/*.json'  # local
    print("processing song data from: ", song_data)
    
    # read song data file and create view for staging songs
    df_song = spark.read.json(input_data)
    df_song.createOrReplaceTempView("staging_songs_table")

    # extract columns to create songs table: song_id, title, artist_id, year, duration
    songs_table = spark.sql("""SELECT song_id, title, artist_id, year, duration FROM staging_songs_table ORDER BY song_id""").dropDuplicates()
    
    # write songs table to parquet files partitioned by year and artist
    songsParquetPath = "{}{}".format('data/output-data/','songs.parquet')
    songs_table.write.mode('overwrite').partitionBy("year","artist_id").parquet(songsParquetPath)

    # extract columns to create artists table: artist_id, name, location, lattitude, longitude
    artists_table = spark.sql(""" SELECT artist_id, artist_name AS name, artist_location AS location, 
                                         artist_latitude AS lattitude, artist_longitude AS longitude 
                                  FROM staging_songs_table 
                                  ORDER BY artist_id """).dropDuplicates() 
    
    # write artists table to parquet files
    artistsParquetPath = "{}{}".format('data/output-data/','artists.parquet')
    artists_table.write.mode('overwrite').parquet(artistsParquetPath)

"""
Processing log data.
Input: spark session, input data to extract log_data, output path to write parquet files
Output: Write users, time, songplays table parquet file to given output path
This function also uses the songs, artists parquet output from earlier to join with log_table to create songplays table.
"""
def process_log_data(spark, input_data, output_data):
    
    # get filepath to log data file
    #log_data =  'data/log-data/*.json' # local
    log_data = input_data
    print("processing log data from: ", log_data)
    
    # read log data file
    df_log = spark.read.json(log_data)
    
    # filter by actions for song plays
    df_log.createOrReplaceTempView("staging_log_data")
    log_table = spark.sql("""SELECT * FROM staging_log_data WHERE page='NextSong'""")

    # extract columns for users table: user_id, first_name, last_name, gender, level  
    users_table = spark.sql(""" SELECT userId AS user_id, firstName AS first_name, lastName AS last_name, gender, level 
                                FROM staging_log_data 
                                ORDER BY user_id """).dropDuplicates() 
    
    # write users table to parquet files
    usersParquetPath = "{}{}".format('data/output-data/','users.parquet')
    users_table.write.mode('overwrite').parquet(usersParquetPath)

    # Refernce: https://knowledge.udacity.com/questions/67777
    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda x: datetime.fromtimestamp(x/1000.0).strftime('%Y-%m-%d %H:%M:%S'))
    log_table = log_table.withColumn('timestamp', get_timestamp('ts'))
 
    # create datetime column from original timestamp column
    get_datetime = udf(lambda x: datetime.fromtimestamp(x/1000.0).strftime('%Y-%m-%d %H:%M:%S'))
    log_table = log_table.withColumn('datetime', get_datetime('ts'))

    # extract columns: start_time, hour, day, week, month, year, weekday
    log_table.createOrReplaceTempView("staging_time")
    time_table = spark.sql("""SELECT DISTINCT datetime AS start_time, hour(timestamp) AS hour, day(timestamp) AS day, 
                                              weekofyear(timestamp) AS week, month(timestamp) AS month, year(timestamp) AS year,
                                              dayofweek(timestamp) AS weekday
                              FROM staging_time
                              ORDER BY start_time
                           """).dropDuplicates()
 
    # write time table to parquet files partitioned by year and month
    timesParquetPath = "{}{}".format('data/output-data/','times.parquet')
    time_table.write.mode('overwrite').partitionBy("year", "month").parquet(timesParquetPath)  

    # References: https://knowledge.udacity.com/questions/150979
    #            https://stackoverflow.com/questions/40508489/spark-merge-2-dataframes-by-adding-row-index-number-on-both-dataframes
    #            https://github.com/jaycode

    # read in song data to use for songplays table
    songs_table = spark.read.parquet('data/output-data/songs.parquet')
    artist_table = spark.read.parquet('data/output-data/artists.parquet')

    # extract columns from joined song and log datasets to create songplays table 
    #songplays: songplay_id, start_time, user_id, level, song_id, artist_id, session_id, location, user_agent
    w = Window().orderBy('song_id')
    df_joined = log_table.join(songs_table.alias("st"),   log_table.song   == col('st.title')) \
                         .join(artist_table.alias("at"),  log_table.artist == col('at.name')) \
                         .select(
                             col('timestamp').alias('start_time'),
                             col('userId').alias('user_id'),
                             'level',
                             'st.song_id',
                             'at.artist_id',
                             col('sessionId').alias('session_id'),
                             'at.location',
                             col('userAgent').alias('user_agent'),
                             year(col('timestamp')).alias('year'), 
                             month(col('timestamp')).alias('month')) \
                         .withColumn('songplay_id', row_number().over(w))

    df_joined.createOrReplaceTempView("songplays_staging")
    songplays_table = spark.sql("""SELECT * FROM songplays_staging""").dropDuplicates()

    # write songplays table to parquet files partitioned by year and month
    # Reference: https://stackoverflow.com/questions/50962934/partition-column-is-moved-to-end-of-row-when-saving-a-file-to-parquet
    # Per above reference parition field year and month are not written into the parquet file, only as folder names:year, month
    #songplays_table
    songplaysParquetPath = "{}{}".format('data/output-data/','songplays.parquet')
    songplays_table.write.mode('overwrite').partitionBy("year","month").parquet(songplaysParquetPath)
    
"""
Creates Spark session, processes song and log data.
Creates songs, artists, users, time, songplays table by extracting input from song and log raw files,
transforming as necessary and loading in parquet files.
"""
def main():
    spark = create_spark_session()

    song_data   = config['DATAPATH']['SONG_DATA']
    log_data    = config['DATAPATH']['LOG_DATA'] 
    output_data = config['DATAPATH']['OUTPUT_DATA']
    
    process_song_data(spark, song_data, output_data)    
    process_log_data(spark,  log_data, output_data)

if __name__ == "__main__":
    main()