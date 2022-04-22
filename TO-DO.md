# TO-DO

1. Unwanted file purge
2. 


# How to

## Download a video
`yt-dlp $source_url --merge-output-format mp4 -o $output_path`

`yt-dlp 'https://www.youtube.com/watch?v=fCF8I_X1qKI' #@param {type:"string"} --merge-output-format mp4 -o ./data/video_Downloads/elon_musk.mp4`

<br>

## Trim a video
`ffmpeg -y -i $input_path -ss $source_start -to $source_end -r $fps $output_path`

`ffmpeg -y -i ./data/video_downloads/elon_musk.mp4 -ss '00:01:48' -to '00:03:33' -r 30 ./data/input/elon_musk.mp4`

## To Crop

`ffmpeg -i $input_path -filter:v 'crop=w=$width:h=$height:x=$top_left_x:y=$top_left_y' -r $fps $output_path`

`ffmpeg -i ./data/input/elon_musk.mp4 -filter:v 'crop=w=620:h=530:x=650:y=0' -r 30 ./data/input/elon_musk_croped.mp4`