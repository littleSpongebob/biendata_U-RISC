### biendata U-RISC 神经元识别大赛简单/复杂双赛道第三方案
[比赛界面](https://www.biendata.com/competition/urisc/)
[详细思路介绍](https://blog.csdn.net/qq_21407487/article/details/104385629)
* 运行环境为CUDA9.0,PYTHON=3.6
*  按照requirement.txt安装依赖，pip install -r requirements.txt，建议不要这么干，因为偷懒没有新建新的python环境，导致导出来的包非常多，可以先安装下面两个包，尝试运行，看缺失那些在进行安装。

*  安装apex
```
    $ git clone https://github.com/NVIDIA/apex
    $ cd apex
    $ pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
    详见https://github.com/NVIDIA/apex

```
* 安装segmentation_models.pytorch
```
    $ pip install git+https://github.com/qubvel/segmentation_models.pytorch

```
* 在./data/complex/ori_imgs下放训练集图片，./data/complex/ori_masks放训练的label

* 在根目录下执行train.sh脚本，里面有相关的裁切数据，执行训练，预测的脚本命令
* 会在当前目录的log下保存训练的ckpt文件，结构如下
```
log
├── complex_1
│   ├── ckpt
│   ├── simple_1.log
│   ├── epoch_10
│   ├── ...
│   └── epoch_95
├── complex_2
│   ├── ckpt
│   ├── simple_2.log
│   ├── epoch_10
│   ├── ...
│   └── epoch_95
```
* test_model_complex.py脚本可以进行预测，结果会生成在根目录的test_result_complex目录下