# OpenposeTools

A tool for drawing coloured lines on to a video file according to the openpose format.

### Install
After cloning and setting up an environment call:
```bash
pip install -r requirements
```
### Usage
```bash
usage: python Visualizer.py [-h] [-oi --out_images OUT_IMAGES] [-coco]
                     [-t --temp tempfolder] [--maxframes maxframes]
                     videofile json outfile

Draw the lines given with the json data on to the given video.

positional arguments:
  videofile             the video file to write on
  json                  the folder of json files with the data for each frame
                        (might be zipped)
  outfile               the output video file

optional arguments:
  -h, --help            show this help message and exit
  -oi --out_images OUT_IMAGES
                        the output folder for the images
  -coco                 add if the COCO openpose format is used instead of body_25
  -t --temp tempfolder  folder for saving the temp files
  --maxframes maxframes
                        maximal number of frames before splitting the video
                        sequence - default to 100

```
*For now a continuation of this project is found at [RedHenLab/OpenposeTools](https://github.com/RedHenLab/OpenposeTools)*
