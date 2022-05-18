<p align="center">
  <h1>SuperGAN</h1>

[![LICENSE](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/TencentARC/GFPGAN/blob/master/LICENSE)  

</p>

---

## :wrench: Dependencies and Installation

### Dependencies
- [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Python = 3.8
- [PyTorch = 1.10](https://pytorch.org/)
- NVIDIA GPU (atleast 8 GB) + [CUDA](https://developer.nvidia.com/cuda-downloads) = 11.3
- Linux


### Installation



1. Clone repo

    ```bash
    git clone https://github.com/95anantsingh/NYU-SuperGAN.git
    cd NYU-SuperGAN
    ```

1. Create conda environment

    ```bash
    conda env create -f environment.yml
    conda activate GAN
    ```

1. Download weights

    ```bash
    wget -i SuperGAN/weight_urls --directory-prefix SuperGAN/data/weights
    ```
    


## :zap: Quick Inference



### Inference


```bash
cd SuperGAN
python faceswap.py
```

Make necessary changes in faceswap.py for different input videos.


## :file_folder: Project Structure

1. `SuperGAN` contains all the project files
2. `SuperGAN/data` contains input output videos and pretrained weights
3. `SuperGAN/faceswap.py` main inference file


## :blue_book: Documentaion

Project report can be found at [docs/SuperGAN_Report.pdf](https://github.com/95anantsingh/NYU-SuperGAN/tree/main/docs/project_report.pdf)
<br>
This Project was part of graduate level Deep Learning course at New York University

## :scroll: License

SuperGAN is released under Apache License Version 2.0.


## :e-mail: Contact

If you have any question, please email `anant.singh@nyu.edu`


<!-- # To-Do

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

1.  -->