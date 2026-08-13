## Visualizing the Smart Environment in AR: An Approach Based on Visual Geometry Matching
### Dependencies
Create the environment by following the steps below:
```shell
git clone https://github.com/overlappredator/OverlapPredator.git;
conda create --name Insight;
conda activate Insight;
cd Insight; pip install -r requirements.txt
cd cpp_wrappers; sh compile_wrappers.sh; cd ..
```
### Training
```shell
bash train
```
To change the weights, modify the pretrain field in the configs/train/indoor.yaml file.
### Testing
First run:
```shell
bash clear_last_test_data
```
Then run:
```shell
bash eve
```
To change the weights, modify the pretrain field in the configs/test/indoor.yaml file. The result files are saved under the snapshot/indoor/est_traj directory.
