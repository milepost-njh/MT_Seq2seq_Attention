# MT_Seq2seq_Attention

NLP - 机器翻译 西班牙语翻译为英语


该项目是基于英语对西班牙语数据的NLP机器翻译问题。使用Seq2seq + Attention算法实现。

提供了英语对西班牙语数据，使用tensorflow2.2 + keras 实现NLP机器翻译问题。

一，各文件介绍

requirement.txt 需安装的环境软件

data_spa_en   英语对应西班牙语数据

seq2seq_attention.py 模型运行文件

二，项目运行方式

python seq2seq_attention.py

三，总结
项目结果：

本项目采用了30000多条双语对应数据，使用NLP-seq2seq+attention模型训练
模型达到效果较好。

不足之处:
1，由于seq2seq+attention模型本质上还是RNN及其变种模型LSTM,GRU等为基础构成的，缺乏并行计算的能力，执行效率不高。
2，由于seq2seq+attention模型中encoder和decoder自身并没有携带attention，若句子较长，信息损失较多。可以使用Transformer及其变种Bert模型达成。
