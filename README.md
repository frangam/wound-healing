# Predicting Wound Progress Framework (PWPF)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8130984.svg)](https://doi.org/10.5281/zenodo.8130984)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) <img src="https://img.shields.io/github/release/frangam/wound-healing.svg"/> [![GitHub all releases](https://img.shields.io/github/downloads/frangam/wound-healing/total)](https://github.com/frangam/wound-healing/releases/download/1.0/wound-healing-v1.0.zip)

Software for Predicting the Dynamic Evolution of Tumor Migration with Artificial Intelligence.

[[`Paper`]Coming Soon] [[`Dataset`](https://doi.org/10.5281/zenodo.8131123)] [[`BibTeX`](#citing-pwpf)]

## Model architectures
<p float="left">
  <p>Quantifying Wound Progress:</p>
  <img src="doc/architecture.png?raw=true" width="47.25%" />
  <p>Predicting Next Frame:</p>
  <img src="doc/next-frame-arch.png?raw=true" width="31.5%" /> 
</p>

## Download last release
Support us downloading our last release

- Click on this button to download [![GitHub all releases](https://img.shields.io/github/downloads/frangam/wound-healing/total)](https://github.com/frangam/wound-healing/releases/download/1.0/wound-healing-v1.0.zip)
- We also appreciate your support downloading it on ZENODO site: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8130984.svg)](https://doi.org/10.5281/zenodo.8130984)


## Citation
Please, cite our work:

- [Garcia-Moreno, F.M.](https://frangam.com/); Ruiz-Espigares, J., Gutierrez-Naranjo, M.A. "Predicting Wound Progress Framework (PWPF)", 2023.



## Installation

The code requires `python>=3.9`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

Install wound-healing-PWPF:

```
pip install git+https://github.com/frangam/wound-healing.git
```

or clone the repository locally and install with

```
git clone git@github.com:frangam/wound-healing.git
cd segment-anything; pip install -e .
```

## <a name="Dataset"></a>Dataset
You can download our MCF-7 Dataset at:  [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8131123.svg)](https://doi.org/10.5281/zenodo.8131123)

## <a name="Models"></a>Model Checkpoints

Click the links below to download the checkpoint for the corresponding model type.

- [Next-Frame model.](https://github.com/frangam/wound-healing/blob/master/models/next-frame-prediction.h5)
- [Quantifying Wound Progress model.](https://github.com/frangam/wound-healing/blob/master/models/quantify-wound-progress.h5)


These models can be instantiated by running:

#### For predicting next frame:

```
from tensorflow.keras.models import load_model
from woundhealing.model import ssim, mse, psnr

next_frame_model = load_model("<path/to/checkpoint>", custom_objects={'ssim': ssim, 'mse': mse, 'psnr': psnr)

#you need to provide your frames in gray scale (1, frames, height, width, 1)
new_prediction = next_frame_model.predict(np.expand_dims(frames, axis=0))
```

#### For quantifying wound progress:

```
from tensorflow.keras.models import load_model

quantifying_model = load_model("<path/to/checkpoint>")

#you need to load your X_test and images_tests
predictions = quantifying_model.predict([X_test, images_test])
```

## Wound Segmentation
We use a combination of [Meta's Segment Anything model](https://github.com/facebookresearch/segment-anything) and [MedSAM model]()


## Citing PWPF

If you use PWPF in your research, please use the following BibTeX entry.

```
@misc{Garcia-Moreno-PWPF,
  title={Predicting Wound Progress Framework (PWPF)},
  author={Garcia-Moreno, Francisco Manuel and Ruiz-Espigares, Jesus and Gutierrez-Naranjo, Miguel Angel},
  year={2023},
  doi={10.5281/zenodo.8130984},
  url={https://github.com/frangam/wound-healing},
  note = {version 1.0}
}
```

If you use our MCF-7 Dataset:

```
@misc{Ruiz-EspigaresMCF7,
  title={Evolution of CSCs and non CSCs MCF-7 Migration in Wound Healing Assay},
  author={Ruiz-Espigares, Jesus and Garcia-Moreno, Francisco Manuel},
  year={2023},
  doi={10.5281/zenodo.8131123},
  url={https://doi.org/10.5281/zenodo.8131123},
  note = {version 1.0}
}
```

