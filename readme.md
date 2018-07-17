## Pspnet 代码

* [paper解读](https://github.com/jiye-ML/Semantic_Segmentation_Review.git)
* [参考github](https://github.com/hellochick/PSPNet-tensorflow)


## 遇到的问题

### sess.run init和预训练模型加载的时机

* 应该先完成初始化工作，然后在加载模型，否则，最后训练的时候模型使用的参数就会是随机初始化的，而不是预训练过得。


### argparse.ArgumentParser 的bool类型 action='store_true和default=True
* 都是模型初始化值，但是某些情况下， action形式会失效


### 因为batch_norm的参数is_training设置了False
* 导致模型训练的时候出现了严重的问题。loss无穷大。
* 这应该和



### 实验

1. 实验一：
    * 使用PSP101训练网络
    * 初始学习率 1e-3，每15000衰减0.1，每一步训练时间1s， batch=2,图片大小480;
    * 结果： 对于细节信息训练不好，其它大的目标基本出现，
    * 使用mom优化方法
    * 有可能过拟合了
2. 实验二：
    * 使用PSP101训练网络
    * 初始学习率 1e-3，每15000衰减0.1，每一步训练时间1.3s， batch=2,图片大小480;
    * 使用adam优化方法
    * 结果： 直接蹦了。原因不明。
4. 实验四：
    * 使用PSP50训练网络
    * 初始学习率 1e-4，每15000衰减0.1，每一步训练时间0.7s， batch=2,图片大小480;
    * 使用mom优化方法



-------------------------

3. 实验三：
    * 改用将label下采样到特征图大小的方式，评判
    
5. 
    * 加入GCN全局卷积模块试试。Large Kernel Matters
    
