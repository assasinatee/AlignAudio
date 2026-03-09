# AlignAudio: Dual-Alignment for Noise-Robust Speech-to-Audio Generation
[![githubio](https://img.shields.io/badge/GitHub.io-Audio_Samples-blue?logo=Github&style=flat-square)](https://assasinatee.github.io/AlignAudio/)

**AlignAudio** is a noise-robust end-to-end speech-to-audio (STA) generation framework designed to directly map speech to environmental audio, offering lower latency than cascaded ASR-TTA systems. The framework addresses noise challenges through dual alignment: (1) aligning clean and noisy speech representations to preserve semantic cues; (2) enforcing consistency in flow-matching generation to ensure temporal coherence.

Experiments show that AlignAudio performs comparably to baselines on clean speech and significantly outperforms them under noisy conditions. At 0dB SNR, AlignAudio improves FD by 27.1% and CLAP by 31.9%.

### Table of Contents
 - [Environment and Data Preparation](#Preparation)
 - [Stage1: Representation-level Alignment](#Rep)
 - [Stage2: Generation-level Alignment](#Gen)

***

<a id="Preparation"></a>

### :scissors: Environment and Data Preparation
#### Environment Setup ####
### Dependency Installation

First, please install dependencies required for training and inference.

```shell
conda create -n alignaudio python=3.10                
```

Then install python dependencies:

```shell
conda activate alignaudio
pip install -r requirements.txt
```

#### Data Preparation #### 
Generating corresponding speech from captions in Audiocaps
```shell
git clone https://github.com/assasinatee/AlignAudio
python /src/data_preparation/speecht5_tts/speecht5_infer.py \
    --speaker CLB \
    --split_files "/data/audiocaps/test/timelabel_text.json" \
    --save_dir "./output_tts"
```
Adding noise from MUSAN, RIRS_NOISES and DEMAND.(DEMAND can replace musan’s place to add out-of-domain environmental noise)

```shell
python /src/data_preparation/speecht5_tts/add_noise.py \
    --clean_dir /path/to/clean \
    --noisy_dir /path/to/noisy_results \
    --musan_dir /data/musan \
    --rirs_dir /data/rirs \
    --snr_db 5 \
    --drr_db 5
```

***

<a id="Rep"></a>

### :bulb: Stage1: Representation-level Alignment
Aligning clean and noisy speech representations to preserve semantic cues
```shell
cd src/noise_distillation
python scripts/train.sh
```

***

<a id="Gen"></a>

### :seedling: Stage2: Generation-level Alignment
Enforcing consistency in flow-matching generation to ensure temporal coherence.
 ```shell
cd src/noise_ft
python scripts/train.sh
 ```

