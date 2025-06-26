# ECG-文本多模态对比学习
#记录我的一个科研项目以下为描述信息

多模态ECG-文本对比学习，主要识别的是心电图（ECG）中的病理类别
Frozen Language Model Helps ECG Zero-Shot Learning
https://zhuanlan.zhihu.com/p/660512113
1 对比学习
层次化对比学习：对ECG片段（如P波、QRS波）与文本中的局部描述（如“PR间期延长”）进行细粒度对齐。
跨模态注意力机制：使用Transformer交叉注意力层显式建模ECG信号与文本词元的交互。
2 数据增强
使用扩散模型（如ECG-DDPM）生成多样化的ECG信号，增强小样本类别。
3 时序特征
使用1D-CNN提取局部特征，LSTM/Transformer捕获长期依赖。
4 语言模型微调
冻结的语言模型（如BERT）可能无法适配医疗术语。轻量级适配器：在冻结的BERT上添加可训练适配器层（如LoRA），低成本调整文本嵌入空间。扩展临床术语词表，避免通用语言模型对专业词汇的次优编码。
PTB-XL
PTB-XL Test Set
MIT-BIH Test Set

#具体任务：
1.理解论文的技术思路
2.复现论文代码仓库
3.改进论文代码以获得更高准确率的结果
