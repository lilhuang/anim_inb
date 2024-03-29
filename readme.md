DEPENDENCIES:

We ran our code with cuda 11.3.1/cudnn 8.2.1/gcc 11.2.0 in Python 3.9.5. For package dependencies, first run:

```pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113```

Then run:

```pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu113_pyt1110/download.html```

Then run:

```pip install git+https://github.com/S-aiueo32/lpips-pytorch.git```

Then finally run:

```pip install -r requirements.txt```



TO TRAIN:

To run out-of-the-box, make sure your image data is in the following format:

```
[Image data root directory]
├── test
│   ├── triplet1
│   │         ├── frame_0.png
│   │         ├── frame_1.png
│   │         ├── frame_2.png
│   ├── triplet2
│   │         ├── frame_0.png
│   │         ├── frame_1.png
│   │         ├── frame_2.png
│   ...       ...
└── train
    ├── triplet1
    │         ├── frame_0.png
    │         ├── frame_1.png
    │         ├── frame_2.png
    ├── triplet2
    │         ├── frame_0.png
    │         ├── frame_1.png
    │         ├── frame_2.png
    ...       ...
```

Then, make sure the flow data is in the following format:

```
[Flow data root directory]
├── test
│   ├── triplet1.npz
│   ├── triplet2.npz
    ...
└── train
    ├── triplet1.npz
    ├── triplet2.npz
    ...
```

The name of each .npz file should be the same as the names of each triplet subdirectory. In addition, each .npz file should have 6 arrays stored in them with the names "02", "20", "01", "10", "12", "21", where the array called "02" indicates the flow from frame_0 to frame_2 in the corresponding triplet directory, the array called "01" indicates the flow from frame_0 to frame_1, and so on.


Then, go to the configs folder. There is an example config file called "config_example.py". In that file, change the paths in lines 1 and 2 to correspond to the image and flow root directories, respectively. In addition, change the "name" on line 40 to whatever you want your output directories to be called.


Finally, to run the training code, go to the main directory. We ran our code with cuda 11.3.1/cudnn 8.2.1 in Python 3.9.5. To train, call:

```python train_anime_sequence_2_stream.py configs/config_example.py```

Note that the argument is simply the path to the config file you just edited; feel free to write other config files and change that argument as needed.







