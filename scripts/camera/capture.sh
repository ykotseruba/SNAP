#!/bin/bash
# ./capture.sh [class] [lighting]
#set -x

if [[ $# -ne 2 ]]; then
	echo "Invalid number of arguments. 2 required (class, lux - )."
	echo "Usage: ./capture.sh [class] [lighting]"
	echo "Lighting options (lux): 10, 1000"
	exit 1
fi

##### CHANGE THIS BEFORE RUNNING THE SCRIPT #####
SAVE_DIR=/path/to/where/the/images/will/be/saved

# Full range of camera settings on Canon EOS T7
## UPDATE IF USING A DIFFERENT CAMERA
declare -a shutter=( 1/4000 1/2000 1/1000 1/500 1/250 1/125 1/60 1/30 1/15 1/8 1/4 0.5 1 2 4 8 15 30 )
declare -a iso=( 100 200 400 800 1600 3200 6400 )
declare -a fnumber=( 22 16 11 8 5.6 )


dt=$(date '+%d-%m-%Y_%H-%M-%S')

obj_id=$( head counter )
class_name=$1
lux_val=$2

# s_0, i_0, f_0 are the "optimal" settings
# these correspond to EV_offset=0
# we set these values by hand for our setup
# based on the auto setting of Canon
if [[ $lux_val == 1000 ]]
then
	wb=6
	wba=2
	wbb=9
	s_0=0
	i_0=6
	f_0=4
elif [[ $lux_val == 10 ]]
then
	wb=6
	wba=2
	wbb=9
	s_0=7
	i_0=5
	f_0=4
else
	echo "Second argument must be 10 or 1000"
	exit -1
fi

################################################

echo EV: $EV


EV_range=8

# array of 1s for keeping track of valid EV settings
valid_ev=()
for value in $(seq 0 $((EV_range*2+1))); do valid_ev+=(1); done


# imageformat
# Choice: 0 Large Fine JPEG 6000x4000
# Choice: 1 Large Normal JPEG
# Choice: 2 Medium Fine JPEG
# Choice: 3 Medium Normal JPEG
# Choice: 4 Small Fine JPEG
# Choice: 5 Small Normal JPEG
# Choice: 6 Smaller JPEG 1920x1280
# Choice: 7 Tiny JPEG
# Choice: 8 RAW + Large Fine JPEG
# Choice: 9 RAW

# /main/imgsettings/whitebalance
# Label: WhiteBalance
# Readonly: 0
# Type: RADIO
# Current: Auto
# Choice: 0 Auto
# Choice: 1 AWB White
# Choice: 2 Daylight
# Choice: 3 Shadow
# Choice: 4 Cloudy
# Choice: 5 Tungsten
# Choice: 6 Fluorescent
# Choice: 7 Flash
# Choice: 8 Manual

echo gphoto2 --auto-detect
echo gphoto2 --set-config /main/imgsettings/imageformat=6 
echo gphoto2 --set-config capturetarget=0 # 0-internal RAM, 1-memory card

# create directory for saving the images
mkdir -p $SAVE_DIR/${obj_id}_${class_name}/${lux_val}

# create a csv file for saving image metadata
DATA_CSV=$SAVE_DIR/${obj_id}_${class_name}/${lux_val}/data.csv
rm -rf $DATA_CSV
touch $DATA_CSV
# write header
echo Path,Object_name,Object_id,Category,EV_offset,Lux,Shutter_speed,ISO,F-Number >> $DATA_CSV


read -p "Set the camera to NO FLASH mode and press any key"

fname_base=$SAVE_DIR/${obj_id}_${class_name}/${lux_val}_auto
auto_fname=${fname_base}.%C

gphoto2 -q --capture-image-and-download --filename ${auto_fname} 

sleep 2

read -r fname auto_shutter auto_iso auto_fnumber <<< `exiftool -csv -ExposureTime -ISO -FNumber "$fname_base".jpg | tail -n +2 | tr "," " "`
echo $auto_shutter, $auto_iso $auto_fnumber

#echo Path,Object_name,Object_id,Category,EV_offset,Lux,Shutter_speed,ISO,F-Number >> $DATA_CSV
echo ${obj_id}_${class_name}/${lux_val}_auto.jpg,${obj_id}_${class_name},${obj_id},${class_name},auto,${lux_val},$auto_shutter,$auto_iso,$auto_fnumber >> $DATA_CSV

read -p "Set the camera to MANUAL MODE and press any key"

gphoto2 --set-config whitebalance=$wb --set-config whitebalanceadjusta=$wba --set-config whitebalanceadjustb=$wbb




# setup progress bar
total_steps=$((${#shutter[@]}*${#iso[@]}*${#fnumber[@]}))
echo $total_steps
progress=0

echo WhiteBalance $wb Adjust A $wba Adjust B $wbb
echo Initial: $s_0, $i_0, $f_0
echo Settings: ${shutter[s_0]}, ${iso[i_0]}, ${fnumber[f_0]}

full_count=0

echo -n "[--------------------] 0% "

# iterate over all possible settings of shutter speed, iso and f-number
for (( s=0; s < ${#shutter[@]}; s++ )) 
do
	cur_shutter=${shutter[s]}
	
	for (( i=0; i < ${#iso[@]}; i++ ))  
	do 
		cur_iso=${iso[i]}
		
		# gphoto2 --set-config iso=${cur_iso}
		for (( f=0; f < ${#fnumber[@]}; f++ )) 
		do
			#echo -e ' \t\t ' F-Number $cur_fnumber
			cur_fnumber=${fnumber[f]}

			# Compute EV offset
			# sum of differences between current and initial settings for iso, shutter and fnumber
			ev_off=$(((s - s_0) + (i - i_0) + (f - f_0)))

			#echo ShutterSpeed=$cur_shutter ISO=$cur_iso FNumber=$cur_fnumber EV_offset=$ev_off
			
			# the next line checks whether the ev_offset is within range
			# the last part of the condition checks if this is a valid_ev
			# if a picture has been taken for that EV bin before and was severely over- or underexposed
			# we skip this EV bin
			if [ ${ev_off} -le $EV_range ] && [ ${ev_off} -ge -$EV_range ] && [ ${valid_ev[$((ev_off+EV_range))]} -eq 1 ]
			then
				FLIP=$(($RANDOM%2))
				#FLIP=1
				if [ $FLIP -eq 1 ]; then

					# file name template
					fname_base=$SAVE_DIR/${obj_id}_${class_name}/${lux_val}/$ev_off/${s}_${i}_${f}
					fname=${fname_base}.%C

					# check if this image exists already
					if [ -f "${fname_base}.jpg" ]; then
						echo $fname exists ... skipping
						continue
					fi
					#--set-config whitebalance=6 
					# take a picture and save it to the file name
					gphoto2 --set-config aperture=${cur_fnumber} --set-config shutterspeed=${cur_shutter} --set-config iso=${cur_iso} --capture-image-and-download --filename ${fname} >> $SAVE_DIR/${obj_id}_${class_name}/${lux_val}_log.txt 2>&1
					full_count=$((full_count+1))

					# # Check if image is over or underexposed
					img_avg=`python3 compute_img_average.py "${fname%.*}".jpg` # Computes the average grayscale value of the image
					
					if [ "$img_avg" -gt 253 ] || [ "$img_avg" -lt 3 ]; then
						# if the image is very under- or over-exposed
						# we mark the corresponding EV bin as invalid
						# delete the image and reduce the image count

						# if ev offset is positive, then all next ev indices will
						# be overexposed too, so we set the valid_ev to 0
						if [ $ev_off -gt 0 ]; then
							for (( j=$((ev_off+EV_range)); j < ${#valid_ev[@]}; j++ )) 
							do
								valid_ev[$j]=0
							done
						# similarly for the negative ev offsets, all ev indices
						# smaller than this are also marked as invalid
						elif [ $ev_off -lt 0 ]; then
							for (( j=$((ev_off+EV_range)); j > 0; j-- )) 
							do
								valid_ev[$j]=0
							done
						fi														
						rm -rf "${fname%.*}".*
						full_count=$((full_count-1))
					else
					  # only record metadata for valid images
					  echo ${obj_id}_${class_name}/${lux_val}/$ev_off/${s}_${i}_${f}.jpg,${obj_id}_${class_name},${obj_id},${class_name},${ev_off},${lux_val},$cur_shutter,$cur_iso,$cur_fnumber >> $DATA_CSV
					fi
				#else
				#	echo "Skipped - Coin Flip"				
				fi	
			#else
			#	echo Skipped $ev_off...
			fi

			############### PROGRESS BAR ##############
		    # Calculate the number of '#' to display
		    let filled_slots=progress*20/total_steps

		    # Create the progress bar string
		    bar=""
		    for ((j=0; j<$filled_slots; j++)); do
		        bar="${bar}#"
		    done

		    # Create the remaining bar string
		    for ((j=filled_slots; j<20; j++)); do
		        bar="${bar}-"
		    done

		    # Calculate percentage
		    let percentage=progress*100/total_steps

		    # Print the progress bar
		    echo -ne "\r[${bar}] ${percentage}% "

		    # Update progress
		    let progress++	
		    ##########################################		

		done
	done
done

# delete empty directories
find $SAVE_DIR/${obj_id}_${class_name}/ -type d -empty -delete

# show the total number of images taken
echo Total image captured $full_count

# check the battery level
# Note: this is unreliable as it is incremented in 25% intervals
# so the camera may be almost out of juice but still show 25% level
gphoto2 --get-config batterylevel

# there is a counter file in this directory, increment by 1
# after every successful capture
echo INCREMENT counter IF RECORDING FINISHED SUCCESFULLY
