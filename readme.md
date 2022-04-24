# To-Do

1. Unwanted file purge
2. 

<br><br>

# Setup

1. Run - `git clone `
2. Run - `python setup.py` only once.
3. Run - `GAN ` to activate shared environment afterwards whenever required.
4. Run - `SuperGAN_DATA` for switching to data directory whenever required.

<br><br>

# How to

## Download a Video

Format - <br>
`yt-dlp $source_url --merge-output-format mp4 -o $output_path`

> Example - <br>
> `yt-dlp 'https://www.youtube.com/watch?v=fCF8I_X1qKI' #@param {type:"string"} --merge-output-format mp4 -o ./data/video_Downloads/elon_musk.mp4`

<br>

## Trim a Video

Format - <br>
`ffmpeg -y -i $input_path -ss $source_start -to $source_end -r $fps $output_path`

> Example - <br>
> `ffmpeg -y -i ./data/video_downloads/elon_musk.mp4 -ss '00:01:48' -to '00:03:33' -r 30 ./data/input/elon_musk.mp4`

<br>

## Crop a Video

Format - <br>
`ffmpeg -i $input_path -filter:v 'crop=w=$width:h=$height:x=$top_left_x:y=$top_left_y' -r $fps $output_path`

> Example - <br>
> `ffmpeg -i ./data/input/elon_musk.mp4 -filter:v 'crop=w=620:h=530:x=650:y=0' -r 30 ./data/input/elon_musk_croped.mp4`

<br>

## How to Stack models

### 1. Run FSGAN

1.

<br>

### 2. Extract Frames

1. 

<br>

### 3. Run GFPGAN

1.

<br>

### 4. Encode Video

1. 
