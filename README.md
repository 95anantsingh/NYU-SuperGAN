<p align="center">
  <h1>SuperGAN</h1>

[![LICENSE](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/TencentARC/GFPGAN/blob/master/LICENSE)  

</p>

---

## :wrench: Dependencies and Installation

- [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Python = 3.8
- [PyTorch = 1.10](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- Linux

### Installation

We now provide a *clean* version of GFPGAN, which does not require customized CUDA extensions. <br>
If you want to use the original model in our paper, please see [PaperModel.md](PaperModel.md) for installation.

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

<!-- 1. Download weights

    ```bash
    conda env create -f environment.yml

    conda activate GAN
    ``` -->


## :zap: Quick Inference



### Inference

Make necessary changes in faceswap.py before running.
```bash
cd SuperGAN
python faceswap.py
```

If you want to use the original model in our paper, please see [PaperModel.md](PaperModel.md) for installation and inference.



## :scroll: License and Acknowledgement

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