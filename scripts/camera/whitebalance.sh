# for wb in {0..6};
# do
# 	gphoto2 --set-config whitebalance=${wb} --capture-image-and-download --filename wb_1000lux_${wb}.%C
# done

for wba in {-9..9}; 
do 
	for wbb in {-9..9}; 
	do	
		filename=10lx_6/wb_10lux_6_${wba}_${wbb}.%C
		gphoto2 --set-config whitebalance=6 --set-config whitebalanceadjusta=$wba --set-config whitebalanceadjustb=$wbb --capture-image-and-download --filename $filename
	done
done