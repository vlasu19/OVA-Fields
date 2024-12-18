<p align="center">

  <h1 align="center">OVA-Fields: Open Vocabulary Affordance Fields for Scene Understanding</h1>
  <p align="center">
    <a href="https://github.com/vlasu19"><strong>Heng Su</strong></a>
    ·
    <a href="https://orcid.org/0000-0002-0434-202X"><strong>Mengying Xie</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=5GVcOTEAAAAJ&hl=en"><strong>Nieqing Cao</strong></a>
    ·
    <a href="https://yding25.github.io/"><strong>Ding Yan</strong></a>
    <br>
    <a href="https://www.researchgate.net/profile/Fuqiang-Gu"><strong>Fuqiang Gu</strong></a>
    ·
    <a href="https://github.com/ssspeg"><strong>Beichen Shao</strong></a>
    ·
    <a href="http://www.cs.cqu.edu.cn/info/1274/7704.htm"><strong>Chao Chen*</strong></a>


  </p>

  <h2 align="center">Submited to CVPR 2025</h2>
  <h3 align="center"><a href="">Paper</a> | <a href="">Video</a> | <a href="">Project Page</a></h3>
  <div align="center"></div>

</p>

<p align="center">
  <a href="">
    <img src="https://github.com/vlasu19/OVA-Fields/blob/master/resources/introduction.png" width="100%">
  </a>
</p>
<p align="center">
<strong>OVA-Fields</strong> enables robots to detect and interact with functional parts in real 3D scenes by mapping natural language commands to precise affordance locations.
</p>

<br>

<!-- TABLE OF CONTENTS -->

<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#installation">Installation</a>
    </li>
    <li>
      <a href="#data-preparation">Data Preparation</a>
    </li>
    <li>
      <a href="#checkpoints">Checkpoints</a>
    </li>
    <li>
      <a href="#run">Run</a>
    </li>
    <li>
      <a href="#acknowledgement">Acknowledgement</a>
    </li>
    <li>
      <a href="#citation">Citation</a>
    </li>
  </ol>
</details>


## News :triangular_flag_on_post:

- [2024/12/12] Code is released.


## Installation

```
conda create -n ova python=3.10
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

cd OVA-Fields
pip install -r requirements.txt

cd gridencoder
conda install nvidia/label/cuda-12.1.0::cuda-nvcc
you can use command: `which nvcc` to see where the nvcc installed.
~which nvcc: /home/user/anaconda3/envs/ova/bin/nvcc
export CUDA_HOME=/home/user/anaconda3/envs/ova
python setup.py install
```

## Data Preparation

Our data collection is conducted using the **Record3D App** on iPhone/iPad Pro equipped with a LiDAR module. This app efficiently captures RGB-D video frames while recording associated data, including camera intrinsics, extrinsics, and pose information. The collected data can be exported in **.r3d** format, which our code can directly read and process.

You can use your own device to collect data for custom scenes. During the data collection process, please keep the camera stable and ensure the entire object is captured. This will help improve the data quality and enhance the model's performance.

you can find more infomation at: [Record3D](https://github.com/marek-simonik/record3d)

Additionally, we provide the dataset used in our paper, which can be downloaded at [Google Drive](https://drive.google.com/file/d/1_3bCNzlL-WXtHrt86otewXANmHl_R88H/view?usp=drive_linkOnce), you can run the **demo.ipynb** file to quickly test the performance of our model.

## Checkpoints

Our pretrained model can be downloaded from [Google Drive](https://drive.google.com/file/d/1c7vfFWWDBZEn9XYfaSk7pmghoLD5K7nW/view?usp=drive_link).


## Run

After setting up the required environment and obtaining the necessary data, you can train our model by following these steps:

1. Modify the `DATA_PATH` in **build_model.py** to point to your data directory.

2. Modify the `SAVE_DIRECTORY` in **train.py** to specify the path where you want to save the model.

3. Run the following command to build the model:

   ```bash
   python build_model.py
   ```

4. Run the following command to start training the model:

   ```bash
   python train.py
   ```

You can also easily run our code and obtain relevant visualizations by executing **demo.ipynb**.

## Acknowledgement

We would like to extend our gratitude to [CLIP-Fields](https://github.com/notmahi/clip-fields ), whose code has greatly supported our work.

## Citation

If you find our code or paper useful, please cite

```bibtex

```
