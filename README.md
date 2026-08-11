## Visualizing the Smart Environment in AR: An Approach Based on Visual Geometry Matching
### 依赖
按照以下步骤创建环境
```shell
git clone https://github.com/overlappredator/OverlapPredator.git;
conda create --name Insight;
conda activate Insight;
cd Insight; pip install -r requirements.txt
cd cpp_wrappers; sh compile_wrappers.sh; cd ..
```
### 数据集

测试数据集按以下方式排列：

- `data`
    - `1`
        - `dataset_test`
        - `gt`
          - `gt.info`
          - `gt.log`
        - `pkl`
          - `test_info.pkl`
    - `2`

### 训练
将data目录下的dataset_train和dataset_val复制到data_ThreeDMatch目录下，然后执行
```shell
bash train
```
修改权重在configs/train/indoor.yaml文件中的pretrain位置
### 测试
先执行
```shell
bash clear_last_test_data
```
将data目录下的dataset_test复制到data_ThreeDMatch目录下，将gt下的两个文件复制到configs/benchmarks_nodes/nodes/nodes_600目录下，将pkl中的文件复制到node_configs目录下，然后执行
```shell
bash eve
```
修改权重在configs/test/indoor.yaml文件中的pretrain位置，结果文件在snapshot/indoor/est_traj目录下
