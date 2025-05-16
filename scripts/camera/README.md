# Installation

### yoctopuce v3 light sensor

Download library for Command line API from https://www.yoctopuce.com/EN/libraries.php
Other installation instructions are at https://www.yoctopuce.com/EN/products/yocto-light-v3/doc/LIGHTMK3.usermanual.html

On Linux, add path to binaries to PATH variable and add udev rules as described in troubleshooting https://www.yoctopuce.com/EN/products/yocto-light-v3/doc/LIGHTMK3.usermanual.html#CHAP25SEC3

Note: if the original instructions do not work, renaming the rules file to 99-yoctopuce.rules might help.

### gphoto2

Download and install libgphoto2 (http://www.gphoto.org/)

Follow instructions at http://www.gphoto.org/doc/manual/

Build libgphoto2 with usb support
https://github.com/gphoto/libgphoto2


```
sudo apt-get install libexif-dev libjpeg-dev libpopt-dev libusb-dev
sudo apt-get install autopoint
autoconf (or autoreconf -f -i if failed before)
```

In libgphoto2 directory run:

```
./configure
make
sudo make install

```

# Collecting images

## Set environment variables
 
In the `capture.sh` script, set SAVE_DIR to the directory where you want images to be saved. If using a camera different from ours, update camera setting lists. Update white balance settings, and s_0, i_0, f_0 which will be used to compute EV offsets.

Follow the following procedure to capture one set of images for 1 illumination condition.

1. Place the object(s) on the table, put the camera on the tripod, and adjust focus. 

Note: When the camera is first connected to the laptop, it may mount it as a hard drive. **Unmount** the camera before capture.

2. 5. Adjust the lights to the appropriate illumination level. For SNAP, we used 10 and 1000 lux. 

Connect the light sensor to the laptop and adjust its location on the table so that it faces the LED lights to get an accurate reading. Turn off any other lights in the room and close the curtains.

To ensure consistent illumination we did the following:

- Put marks on the dials controlling the LED lights for each lighting condition. 

- After adjusting the light, we checked the light level by running the `lux.sh` from the terminal, which outputs the current illumination level in lux. Note: light sensor output fluctuates, so you will need to run the script multiple times and make sure that the range is accurate. For example, for 1000 lux, we usually got somewhere between 990-1020 lux.

5. Once all settings are done, from sensor_bias directory directory run `capture.sh` script and follow the instructions on screen. The script will ask to first turn the camera to auto mode, will take one shot. Then you will be asked to switch it to manual and the rest will be handled automatically.

```
camera/capture.sh <object_class> <lighting>
```

<object_class> is a string denoting the class label (e.g. book) and lighting options are: 10 or 1000, e.g. `camera/capture.sh book 10`

Add `time` before the command to measure time it takes to run.  To monitor the progress remotely, run the script in `tmux` and connect to the sandbox laptop from a different machine.

The script will then iterate through all camera parameters and save images on the laptop as a JPEG. One set of images for the given illumination conditions will take approx. 40-45 minutes.

**Note for Mac OS**: on Mac OS Ventura 13.4 and above the capture script must be executed with `sudo` because of the `ptpcamerad` process that cannot be terminated and constantly takes over the camera.

**Note on the camera battery.** Monitor the status of the battery. One full charge is usually enough to take 3 sets of images with Canon, but it's better not to push it too much. If the battery runs out in the middle of the set, the whole set will need to be recaptured since the camera position will shift when you change the battery. We kept a charger with an extra battery on hand and swapped the batteries after every 2 sets.

# Generating/updating dataset statistics

`capture.sh` automatically generates a csv file that contains the following columns

Path,Object_name,Object_id,Category,EV_offset,Lux,Shutter_speed,ISO,F-Number

- Path - path to the image
- Object_name - object_id and category, e.g. 21_book
- Object_id - object id, e.g. 21
- Category - object class, e.g. book
- EV_offset - offset indicating exposure level
- Lux - lighting level
- Shutter_speed, ISO, F-Number - values for parameters

Note: all image file are coded as shutter_speed_ISO_FNumber.JPEG. To make them shorter, we use the indices of the parameter values rather than values. 

For example, for Canon the following shutter speed settings are available [1/4000 1/2000 1/1000 1/500 1/250 1/125 1/60 1/30 1/15 1/8 1/4 0.5 1 2 4 8 15 30]

So if the image is named `0_1_4.jpg', this means it was taken with 0th shutter speed setting which is 1/4000, 1st setting of the FNumber and 4th setting of the ISO. 

Whenever new images are added, run the following command to update the excel file with the list of all images and corresponding camera settings.
```
sh camera/generate_data_spreadsheet.sh
```