## SSIM

Including the original frames in every prediction

```
./ssim.py --base results/next-image/synthetic/model_9/include_originals/
```

only predictions

```
./ssim.py --base results/next-image/synthetic/model_9/no_include_originals_only_predictions/
```

** For assessing real images:

-- including the original frames in every prediction

```
./ssim.py --base results/next-image/real/model_9/include_originals/
```

-- only predictions

```
./ssim.py --base results/next-image/real/model_9/no_include_originals_only_predictions/
```


## Install pytorch-fid

Install from [pip](https://pypi.org/project/pytorch-fid/):

```
pip install pytorch-fid
```

## Usage
To compute the FID score between two datasets, where images of each dataset are contained in an individual folder:

```sh
python -m pytorch_fid results/next-image/synthetic/original/ results/next-image/synthetic/prediction/ --device cuda:0
```

### Using different layers for feature maps

In difference to the official implementation, you can choose to use a different feature layer of the Inception network instead of the default `pool3` layer.
As the lower layer features still have spatial extent, the features are first global average pooled to a vector before estimating mean and covariance.

This might be useful if the datasets you want to compare have less than the otherwise required 2048 images.
Note that this changes the magnitude of the FID score and you can not compare them against scores calculated on another dimensionality.
The resulting scores might also no longer correlate with visual quality.

You can select the dimensionality of features to use with the flag `--dims N`, where N is the dimensionality of features.
The choices are:
- 64:   first max pooling features
- 192:  second max pooling features
- 768:  pre-aux classifier features
- 2048: final average pooling features (this is the default)

```sh
python -m pytorch_fid results/next-image/synthetic/original/ results/next-image/synthetic/prediction/ --device cuda:0 --dims 64
```

```sh
python -m pytorch_fid ~/proyectos/TGEN-timeseries-generation/results/evaluation_synthetic_quality/loso/exp-classes-all-classes/data/fold_0/real_train/ ~/proyectos/TGEN-timeseries-generation/results/evaluation_synthetic_quality/loso/exp-classes-all-classes/data/fold_0/train/  --device cuda:0


python -m pytorch_fid ~/proyectos/TGEN-timeseries-generation/results/evaluation_synthetic_quality/loto/real-train-data/data/fold_0/  ~/proyectos/TGEN-timeseries-generation/results/evaluation_synthetic_quality/loto/epochs-10000/data/fold_0/ --device cuda:1 
````

## Generating a compatible `.npz` archive from a dataset
A frequent use case will be to compare multiple models against an original dataset.
To save training multiple times on the original dataset, there is also the ability to generate a compatible `.npz` archive from a dataset. This is done using any combination of the previously mentioned arguments with the addition of the `--save-stats` flag. For example:
```sh
python -m pytorch_fid --save-stats path/to/dataset path/to/outputfile
```