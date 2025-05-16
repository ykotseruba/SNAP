
csvstack `find data_v5 -name "data.csv"` > data_v5.csv
ssconvert data_v5.csv annotations/sensor_bias_data_v5.xlsx
#rm -rf data_v5.csv