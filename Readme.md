# 3DFeat-Net: Weakly Supervised Local 3D Features for Point Cloud Registration -- TensorFlow 2 Implementation

![pic-network](docs/pic-network.png)

## About

This work is based on our ECCV2018 paper. 3DFeat-Net is an approach for learning features for point cloud geometric registration under weak-supervision, where the supervision is given in terms of whether 2 point clouds have very high overlap or low (or no) overlap. For details, please read our paper which can be found on [arXiv](https://arxiv.org/abs/1807.09413).

Bibtex:

```
@inproceedings{yew2018-3dfeatnet,
    title={3DFeat-Net: Weakly Supervised Local 3D Features for Point Cloud Registration},
    author={Yew, Zi Jian and Lee, Gim Hee},
    booktitle={ECCV},
    year={2018}
}
```

## Environment

This version of the code is developed and tested on the following environment:

* Python ~~3.5~~ <b>3.7.11</b>
* Tensorflow ~~1.4~~ <b>2.5.0</b> (with Cuda ~~8.0~~ <b>11.1</b>)
* Numpy ~~1.13.3~~ <b>1.19.5</b>
* Scikit-learn ~~0.19.1~~ <b>0.24.2</b>

We also use MATLAB scripts for evaluation and processing of data.

To install, create a new `conda` environment in **Python 3.7**, and install the `requirements_tf2.txt` using **pip**. Attempting to install via `conda` will not work, for some reason.

## Network

The network model is in `models/feat3dnet.py`.

### Prerequisites

Before using the model, you first need to compile the customized tf_ops in the folder `tf_ops` (we use the customized grouping and sampling ops from [PointNet++](https://github.com/charlesq34/pointnet2)).

`cd` to `tf_ops` and run `bash tf_ops_compile.sh`. If `nvcc` is not added to path, so do first.
~~Check and execute `tf_xxx_compile.sh` under each subfolder. Update the python and nvcc file if necessary. The scripts has been updated for TF1.4, so if you're using TF version < 1.4, refer to the original script provided with PointNet++ for compilation.~~

### Training

#### Preparation of data

1. Follow instructions [here](scripts_data_processing/Readme.md#training-data) to download and prepare the training data
2. Also download the test data for descriptor matching (i.e. the 30,000 cluster pairs) by following the instructions [here](scripts_data_processing/Readme.md#test-data). We monitor the false alarm rate at 95% recall, as the training loss is not very informative (The provided script evaluates on all of the test data which can be slow; you can change this behavior by modifying VAL_PROPORTION in train.py)
3. Both the training and test sets should be placed in the same folder. The provided scripts assume they're placed in `../data/oxford`, which should contain two subfolders: `clusters` and `train`.

#### Training

Training is divided into 2 stages, where the first stage only trains the descriptor subnetwork without rotation and attention. For convenience, we provide a training script which runs both parts. Simply execute`./train.sh` (you can configure the top few lines to select the GPU, etc).

Training takes around 1-1.5 days to saturate. During training, progress can be monitored by running `tensorboard --logdir=./ckpt` from the root folder, and the false alarm rate will be shown in the fp_rate graph.

### Inference on Example data

1. Run `inference_example.sh` which will load the pretrained model in the folder `ckpt` and generate the keypoints and descriptors for the example data in `example_data`. A sample checkpoint can be downloaded from [here](https://drive.google.com/open?id=1JYZvFmMO3hgLN4ao3MqcMVgHYZtHhPOv). The output will be stored in `example_data/results`.
2. Run the MATLAB script `scripts/computeAndVisualizeMatches.m` which will match the features, estimate the relative transformation (with RANSAC) between the point clouds and display the results.

It should be straightforward to run on your own data, just make sure the data is in the expected format (see `scripts_data_processing/Readme.md`). Note however the following:

* z-axis should be pointing vertically upwards
* The network considers up to 64 points per cluster. For dense point clouds, it will pick the points randomly (as long the flag `--randomize_points` is set which will randomize the input point ordering). This means that the performance may differ slightly with each run.

## Datasets
Refer to [scripts_data_processing/Readme.md](scripts_data_processing/Readme.md).

## Conversion to ONNX
To convert the trained model for inference into ONNX format, first wait for training to complete. Subsequently, run the `inference_example.sh` script, which will call sample data on the model so that a `SavedModel` can be built. Subsequently, it will save the TensorFlow `SavedModel` to a directory (by defaut, `./ckpt/infer_model`).

Subsequently, call the `tf2onnx.convert` submodule (should be installed as part of `requirements.txt`):
```bash
python -m tf2onnx.convert 
--saved-model ./inference_savedmodel/ --output onnx_modles/model_infer.onnx \
--load_op_libraries ./tf_ops/grouping/tf_grouping_so.so,./tf_ops/sampling/tf_sampling_so.so \
--rename-inputs pointcloud --rename-outputs keypoints,features,attention \
--custom-ops QueryBallPoint,GroupPoint,FarthestPointSample,GatherPoint,KnnPoint
```
This will save an `onnx` model named `model_infer.onnx` in the 3DFeatNet base directory. The model can then be verified visually calling `netron model_infer.onnx`.

## Setting up the TensorRT development environment
TensorRT comes with a prebuilt Docker container with TensorRT 8.x in Ubuntu 18.04 with CUDA 11.3. We modified the Dockerfile to include the resources we have been using.

To setup the Docker container, first install Docker. Clone the [TensorRT](https://github.com/NVIDIA/TensorRT) and [onnx-tensorrt](https://github.com/onnx/onnx-tensorrt) repos to a convenient location. We suggest the same parent directory as the one 3DFeatNet is in, and the sample build script below works if it is.
```bash
ARG_IMAGENAME="tensorrt-ubuntu18.04-cuda11.3_3dfn"  # change this to your liking
3DFN_LOC = $(pwd)

sudo docker build -f ${3DFN_LOC}/docker/ubuntu-18.04_modded.Dockerfile \
--build-arg CUDA_VERSION=11.3 --build-arg uid=$(id -u) \
--build-arg gid=$(id -g) --tag=$ARG_IMAGENAME
```
To run the Docker image, run
```bash
sudo docker run -it -v "${3DFN_LOC}/..":"/workspace" \  # This ensures that the TensorRT and onnx-trt build folders can be found.
-p 80 -p 6006:6008 \                                    # binds ports for HTTP and TensorBoard.
--gpus all ${ARG_IMAGENAME}:latest
```
When inside the Docker image, run
```bash
bash 3DFeatNet/docker/startup.sh
```
This installs the TensorRT and onnx-trt inside the Docker image. You can also edit this script to your liking.

## Running inference in ONNX

## Converting ONNX to TensorRT

## Running inference in TensorRT

# Anything below this line is experimental
*****
## Conversion to ONNX and TensorRT
To convert the trained model for inference into ONNX format, first wait for training to complete. Subsequently, run the `inference_example.sh` script, which will call sample data on the model so that a `SavedModel` can be built. Subsequently, it will save the TensorFlow `SavedModel` to a directory (by defaut, `./ckpt/infer_model`).

Subsequently, call ```python -m tf2onnx.convert --saved-model ./ckpt/infer_model --output model_infer.onnx --load_op_libraries ./tf_ops/grouping/tf_grouping_so.so,./tf_ops/sampling/tf_sampling_so.so```. The last flag is important, as it registers the custom ops in ONNX.
The model can then be verified visually calling `netron model_infer.onnx`.

Attempted to use the command 
```bash
python -m tf2onnx.convert --saved-model inference_savedmodel/ --output model_train.onnx --load_op_libraries ./tf_ops/grouping/tf_grouping_so.so,./tf_ops/sampling/tf_sampling_so.so --verbose --rename-inputs pointcloud --rename-outputs sliced_pointcloud,features,attention
```
However, the custom ops are not correctly registered in ONNX, meaning that the ONNX model is unlikely to work out. This has to be resolved before further conversion occurs.

For TensorRT conversion, the docker container was installed. Subsequently, call `sudo ./docker/launch.sh --tag tensorrt-ubuntu18.04-cuda11.3 --gpus all` from `/media/intern/SSD1/TensorRT` to start the Docker container. 

**TODO:** 
1. A symlink to the 3DFeatNet Tf2 repo needs to be created beforehand to allow the Docker Container to access the created ONNX model before things work!
2. The `nvidia-docker-toolkit` needs to be setup on the workstation.