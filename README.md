## Anon-NLP-Toolkit



------

PS:目前还是半成品哈，基本搞完了会删掉这行

#### 说明：

:small_airplane: **目标:** 封装一套各个任务领域预训练语言模型微调用的极简启动工具包，启动流程只需要一个Yaml配置文件

:robot: **结构:** 将微调进行流程分解，使用继承多态的方式达到代码的最大复用。

:rocket: **流程:** 任何一套微调流程都由五个部分组成，极简且清晰（模型配置，模型结构，分词器，数据预处理，训练执行器）



#### 进度：

**已完成：**

| 任务         | 预训练模型                                                   | 结构          | 数据集示例 | Yaml示例 |
| ------------ | ------------------------------------------------------------ | ------------- | ---------- | -------- |
| 命名实体识别 | [bert-base-chinese](https://huggingface.co/ckiplab/bert-base-chinese) | bertBiLSTMCrf |            |          |
| 命名实体识别 | [bert-base-chinese](https://huggingface.co/ckiplab/bert-base-chinese) | bertCrf       |            |          |
| 命名实体识别 | [bert-base-chinese](https://huggingface.co/ckiplab/bert-base-chinese) | bertSoftmax   |            |          |
| 命名实体识别 | [bert-base-chinese](https://huggingface.co/ckiplab/bert-base-chinese) | bertSpan      |            |          |
| 分类         | [bert-base-chinese](https://huggingface.co/ckiplab/bert-base-chinese) | bertSoftmax   |            |          |

**计划中：**





