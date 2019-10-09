# CCF_Negative_Financial_Information_and_Subject_Judgment
#### a simple and effective baseline!
### 适用于将该题目分为两个子任务（负面分类，主体判定）的同学使用。
~~1.本代码不包含负面分类的任务代码，可参考各类 伯特，罗伯特，阿尔伯特 的 分类baseline代码。~~
1.由于有同学需要，就把自己的放上来了，不过是Mxnet，直接使用还需配置相应环境，详情见（http://zh.gluon.ai/chapter_prerequisite/install.html#%E4%BD%BF%E7%94%A8GPU%E7%89%88%E7%9A%84MXNet），这是gluon的文档，内容丰富，甚至完全能够作为深度学习入门教程。如果不想安装，代码也可以作为参考，和pytorch大概也只是一些函数名字不一样。  
2.只需把第一个任务的结果（id,negative)文件加入代码中即可得到最终的结果（id,negative,key_entity)  
3.代码只用了两个方式过滤实体，“取最大字符串”和“NIKE实体过滤”  
  a.取最大字符串方法意思是：如果存在一个实体的字符串可以包含其他实体的字符串，那么就取这一个实体就行。如：小资易贷，小资易贷有限公司，就只取：小资易贷有限公司  
  b.NICK实体过滤的NIKE全称"Not In Key Entity" but in entity。统计每个实体在Entity中出现又在KeyEntity中出现的次数，以及对应没有出现的次数。
设定一个比例，在Entity中出现又在KeyEntity中出现的次数的比例低于多少的实体为"NIKE"实体，直接过滤。  

#### 主体判定在该方式下可以达到线上0.558左右。如果第一个任务负面分类分数F1_s达到0.96 x 0.4 + 0.558 = 0.9415,如果你的F1_s达到0.98 x 0.4 + 0.558 = 0.9495！
#### 代码中func_on_row()里面你还可以继续添加你觉得可以过滤的方法进一步提升效果。
#### 最后，觉得有用，star！感谢。
