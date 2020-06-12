## Induction-Networks

[![issues-open](https://img.shields.io/github/issues/ShaneTian/Induction-Networks?color=success)](https://github.com/ShaneTian/Induction-Networks/issues) [![issues-closed](https://img.shields.io/github/issues-closed/ShaneTian/Induction-Networks?color=critical)](https://github.com/ShaneTian/Induction-Networks/issues?q=is%3Aissue+is%3Aclosed) [![license](https://img.shields.io/github/license/ShaneTian/Induction-Networks)](https://github.com/ShaneTian/Induction-Networks/blob/master/LICENSE)

Unofficial code for [Induction Networks](https://www.aclweb.org/anthology/D19-1403/) by PaddlePaddle.

## DataSet

## Usage
### Requirements
You can use `pip install -r requirements.txt` to install the following dependent packages:

- ![python-version](https://img.shields.io/badge/python-v3.7.4-blue)
- ![numpy-version](https://img.shields.io/badge/numpy-v1.17.2-blue)
- ![paddlepaddle-gpu-version](https://img.shields.io/badge/paddlepaddle--gpu-v1.7.0-blue)
- ![visualdl-version](https://img.shields.io/badge/visualdl-v1.3.0-blue)

### Training

Training script is `./run.sh`:
```bash
CUDA_VISIBLE_DEVICES=0 python3 -u train.py \
    --train_data_path ./data/ARSC/ARSC_train.json \
    --val_data_path ./data/ARSC/ \
    --test_data_path ./data/ARSC/ \
    -N 2 \
    -K 5 \
    -Q 5 \
    --train_episodes 10000 \
    --val_steps 200 \
    --max_length 512 \
    --hidden_size 128 \
    --att_dim 64 \
    --induction_iters 3 \
    --relation_size 100 \
    -B 4 \
    --lr 1e-3 \
    --use_cuda \
    --emb_path ./embedding/glove.6B.300d/ \
    --logdir ./log > ./log/run.log 2>&1 &
```

You can use `python3 train.py -h` to see all available parameters.

### Test

In fact, if the `--test_data_path` is given in the training, the test task will be always performed after training.

### Inference

Inference code is `inference.py`. You only need to modify the way data is read in this code.

### Visualization

```bash
visualdl --logdir ./log/visualdl_log/ -m ./log/infer_model/
```

**Note:** In order to visualize metrics and graph correctly, you must use `visualdl==1.3.0` and `protobuf==3.6.1`.


## Maintainers

[@ShaneTian](https://github.com/ShaneTian).

## License

[Apache License 2.0](LICENSE) © ShaneTian