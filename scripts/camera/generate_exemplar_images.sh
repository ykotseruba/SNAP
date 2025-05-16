
mkdir annotations/images_data_v5
mkdir annotations/masks_data_v5

for img_dir in data_v5/*;
do
  img_name="${img_dir/data_v5/}"_10_auto.jpg
  cp $img_dir/10_auto.jpg annotations/images_data_v5/${img_name}
done
