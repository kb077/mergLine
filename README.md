#### 环境配置
Ubuntu python3.9 
cd 0_requirements
conda actiavte env
pip install -r requirements.txt

#### 构建
需要将数据解压到目录1_dataset
解压后的数据在1-9-23，1-9-8 等按日期编号的文件夹下面
cd 1_dataset
1_partiion.py 根据中心点剪切牙模预备体，注意数据未做增强，对手动定位中心点效果有影响
2_boundary.py , 3_labeler.py 自动给预备体打便签
前处理的数据会保存在 data/

4_dataset_trian.py 构建训练集
5_dataset_test.py  构建测试集
构建的数据集会保存在 marginSeg/predict/data

#### 训练 
cd 1_dataset/marginSeg/predict
文件夹models_pointmlp_biou_alpha包含已经训练的模型
重新训练要修改文件training.py

#### Demo
复制1_dataset/marginSeg/predict/models_pointmlp_biou_alpha到3_demo/models_pointmlp_biou_alpha
cd 3_demo
python main.py 注意后处理算法还不稳定




