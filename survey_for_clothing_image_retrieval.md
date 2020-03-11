### 关于服装图像检索的综述 A Survey for Clothing Image Retrieval
#### 整理者：艾宏峰 Alvin Ai  
**** 
[Paper](http://openaccess.thecvf.com/content_ICCVW_2019/papers/CVFAD/Kinli_Fashion_Image_Retrieval_with_Capsule_Networks_ICCVW_2019_paper.pdf)
2019 **Fashion Image Retrieval with Capsule Networks**  
**关键词**：[SC/RCCapsNet](PAPER_IMG/SC_RCCapsNet.jpg)  
**描述**：这篇论文其实是在Sabour和Hinton新提出的神经网络Capsule Networks基础上改进用于执行服装检索任务。该模型单用图片不用属性和landmark信息就能比FashionNet表现更好，因为Capsule Network可以潜在学习目标的pose configuration（而CNN神经网络模型会丢失目标的层次空间信息），Capsule Network包含两个主要模块：特征抽取块和胶囊层，为了得到更强的特征，作者设计了两个不同的特征抽取块：Stacked Convolutions（SC）和Residual Connection(RC)。之后还跟着两个全连接胶囊层（Primary Capsule和Class Capsule）。  
**心得**：虽然和其它S0TA模型相比，SC/RCCapsNet少了大概2倍的参数量，但是由于Capsule Networks中的dynamic routing算法需要更长的训练时间，因此出于时间考虑，像难样本策略，进阶版目标函数，网络集成和关注机制等其它手段都没加进去。
**** 
[Paper](http://openaccess.thecvf.com/content_CVPRW_2019/papers/FFSS-USAD/Park_Study_on_Fashion_Image_Retrieval_Methods_for_Efficient_Fashion_Visual_CVPRW_2019_paper.pdf) 
2019 **Study on Fashion Image Retrieval Methods for Efficient Fashion Visual Search**  
**关键词**：[DenseNet121+OS+IS](PAPER_IMG/DenseNet121+OS+IS.jpg)  
**描述**：作者提出了个服装检索框架DenseNet121+OS+IS，首先训练集有anchor image, postive和negative image构成，框架backbone是采用了ResNet, DenseNet等CNN结构以提出特征，然后分开两条loss path（即基于目标类别的分类损失OC loss和基于实例的相似损失IS），在为了缓解OC和IS损失在特征嵌入空间上的不同，作者使用了一个feature relaxation module（它由一个卷积层，relu，dropout和最终卷积层构成）去调整在OC损失计算前的特征分布。  
**心得**： 作者对比了多种CNN结构，发现DenseNet121比ResNet50, SEResNet50好。而且如果把OS和IS结合使用，模型在consumer-to-shop检索任务中获得较大的提升，DenseNet121+OS+IS比FashionNet表现更好。
**** 
[Paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Ge_DeepFashion2_A_Versatile_Benchmark_for_Detection_Pose_Estimation_Segmentation_and_CVPR_2019_paper.pdf) [Dataset](https://github.com/switchablenorms/DeepFashion2)
2019 **Deepfashion2: A versatile benchmark for detection, pose estimation, segmentation and re-identification of clothing images**  
**关键词**：DeepFashion2, [Match R-CNN](PAPER_IMG/DeepFashion2.jpg)   
**描述**：DeepFashion2数据集大小是DeepFashion的3.5倍。每张服装图片上的服装单品都标注有：scale，occlusion，zooming，viewpoint，bounding box，dense landmark和per-pixel mask。作者提出了基于Mask R-CNN改进的一个baseline模型 - Match R-CNN。该模型有三部分组成：第一是Feature Network（FN），FN包含一个ResNet-FPN backbone，RPN和RoiAlign，主要是用于抽取pyramid feature map。第二是Perception Network（PN）, PN包含三个分支，分别是landmark estimation, clothes detection和mask prediction。第三部分为Match Network（MN），MN包含一个特征提取器和相似度学习网络，其任务是解决服装检索问题。整个模型包含五个损失函数：cls，box，pose，mask和pair。  
**心得**：  
- 人工剔除了大遮挡，小尺度和低分辨率的图像。  
- 在模型采用多尺度训练，短边800，长边不超过1333。在consumer-to-shop服装检索上使用1x schedule训练。衣服检测和分割采用2x，但使用1x也行，不过学习率要方法一倍。  
- 模型表现“中等尺度，轻微遮挡，不放大，正面图”的数据上表现最好。但只要存在“小或大尺度，严重遮挡和放大，纯商品图，侧面或背面图”任何一种条件，都会减低模型表现，其中，侧/背面是因为丢失正面一些关键特征而引发检索错误，而纯商品图表现不好是因为存在deformation问题。  
- 在consumer-to-shop服装检索上，box+pose+class的特征搭配能取到最好的检测效果。说明了landmark位置在多个场景下鲁棒性很好。
**** 
[Paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Ge_Deep_Metric_Learning_ECCV_2018_paper.pdf) 
2018 **Deep metric learning with hierarchical triplet loss**  
**关键词**：[HTL](PAPER_IMG/HTL.jpg)  
**描述**：因为随机采样triplet样本，会因为信息冗余和minibatch无法获取到全数据集的分布情况而可能导致拟合慢且结果出现次优问题。为了解决这个问题，作者提出了hierarchical triplet loss（HTL），它鼓励minibatch抽取的训练样本具有相似的视觉外表但却是不同的语义内容（即不同类别），这使得模型学习到更微妙的辨别特征。首先使用基于传统triplet loss进行预训练的模型去构建一个hierarchical class tree。然后根据层次类别数进行抽样（anchor-neighbor  sampling），之后使用dynamic violate margin去替换原triplet loss里的triplet loss，进行损失计算，。dynamic violate margin是将数据分布的全局语境信息考虑进去，所以HTL会更好些。表现比FashionNet好很多。  
**心得**：该论文其实就是提供了一种新采样方法（A-H sampling）和与之配套的损失函数HTL。唯一要注意的就是在第一个epoch时，得先使用标准triplet loss训练模型，得到层次树后，再使用HTL进行训练和树的更新。虽然A-H抽样的目的是跟hard negative mininig相似，但是结果前者比后者提升更高。  
**** 
[Paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Wonsik_Kim_Attention-based_Ensemble_for_ECCV_2018_paper.pdf) 
2018 **Attention-based ensemble for deep metric learning**  
**关键词**：[ABE-M](PAPER_IMG/ABE-M.jpg)  
**描述**：该论文对Deep Metric Learning提出一种基于关注机制的集成手段 - ABE-M。不同于传统M-heads集成，ABE-M以GoogLeNet为网络基础，使用部分网络结构抽取特征输入到M个不同的关注模块（本质是inception和conv），得到M个关注masks，让他们与抽取特征图相乘得到M个关注特征图，之后在经过全局特征嵌入函数，然后计算损失（本质是contrastive loss）。为了避免不同的学习器关注特征不同区域，加入divergence loss作为一种正则化手段，增加输入图像嵌入后的点之间的距离。表现比FashionNet好很多。  
**心得**：  
- ABE-M使用软关注机制，从而使得反向传播是基于全梯度计算方式，而硬关注需要policy gradient estimation。  
- divergence loss的lambda系数是1时，表现最好。ABE-M没有它不行，不然所有学习器得到的都是相似的嵌入。而对于M-heads集成来说，divergence loss不重要。  
- ABE-M比M-heads集成参数量少，表现好。但ABE-M由于额外的关注模块计算，所以需要更高的flops。
**** 
[Paper](https://homepages.dcc.ufmg.br/~nivio/papers/dan@ijcnn18.pdf) 
2018 **Effective fashion retrieval based on semantic compositional networks**  
**关键词**：[Comp-Net](PAPER_IMG/Comp-Net.jpg)  
**描述**：该篇论文主要是利用图片和其标注(occasion，style和season)来检索配套服装。作者提出的semantic compositional network（即Comp-Net）是先用一个composition learning网络学习到语义空间里的特征，然后进行相似度排名。在composition learning网络中有两个比较重要的点：conditional normalization和cost-sensitive minimization，前者是作者将标注的co-occurrence information加入到softmax的概率计算中，而后者是加大了对错误套装匹配的损失。  
**心得**：conditional normalization（将图像的文本信息考虑到softmax中）有参考价值。
****
[Paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Opitz_BIER_-_Boosting_ICCV_2017_paper.pdf) 
2017 **BIER-boosting independent embeddings robustly**  
**关键词**: [BIER](PAPER_IMG/BIER.jpg)  
**描述**：由于嵌入向量有高相关性，因此作者提出一种在embedding集成手段BIER，在不同的learners（即final linear layers）下，将boosting思路引入到梯度更新上。例如学习器1先计算损失，然后对数据集充分配权重（错分类样本收到更大权重，正确分类样本赋小更小权重），接着学习器2在更新权重下的数据集上计算损失，依次类推到最后一个学习器计算完损失，最后在统一所有学习器进行梯度更新。这种boosting手段保证了弱学习器的多样性，避免但用一个嵌入向量层引起的过拟合问题，而且还能使得模型关注样本不同地方。同时作者为了各学习器刚开始就独立开来，提出新的权重初始化手段，实验表现很好，且没有引入多余的参数。  
**心得**：baseline的嵌入特征长度是512，作者建议是将其分为不重复的3-4 groups（即学习器），太多学习器会减弱学习器的表征能力，太少学习器，则减少特征的多样性和独立性。  
****
[Paper](https://arxiv.org/pdf/1710.11446.pdf) 
2017 **Clothing retrieval with visual attention model**  
**关键词**: [Visual Attention Model](PAPER_IMG/VAM.jpg)，Impdrop  
**描述**：该论文将关注机制引入到服装检索中，使得消除冗余背景带来的影响。整个模型结构分为两个分支：全局分支和关注分支。在关注分支上，图片先进入到一个全卷积网络FCN得到关注图，之后与用VGG/GoogLeNet的低层进行特征抽取得到特征图进行融合，融合手段为Impdrop（即在关注图上与伯努利序列结合后，再与特征图进行乘积操作）。之后融合的特征关注图通过高层网络，与得到的全局特征进行合并得到最终的特征向量。使用损失函数是triplet loss。模型表现比FashionNet高很多。  
**心得**：  
- 如果直接将关注图和输入图像直接合并，会出现extra fake edge。而将关注图与低层特征图结合可以避免这种影响。  
- 考虑人和服装共现关系，作者的FCN是在clothes parsing和person segmentation上预训练过的。  
- 如果单使用product方法融合特征图和低层全局特征会引起过拟合。
****
[Paper](https://www.researchgate.net/profile/Sanyi_Zhang/publication/321785076_Watch_Fashion_Shows_to_Tell_Clothing_Attributes/links/5cdf0e3c92851c4eabaa3e07/Watch-Fashion-Shows-to-Tell-Clothing-Attributes.pdf) 
2017 **Watch fashion shows to tell clothing attributes**  
**关键词**: [Unsupervised triplet network](PAPER_IMG/Unsupervised_triplet_network.jpg)  
**描述**：该论文提出了一种半监督方法去预测服装属性，首先是用无标注视频去预训练CNN模型，然后用带属性标注的图像数据集去微调训练模型。详细来说，先用基于VGG-16预训练的fast R-CNN模型检测出视频中的服装，然后将某视频第一个选取的帧数下的服装作为base frame，同一视频下其它帧数下的服装作为positive frame，其它视频上服装为negative frame，三个图像分别输入到一个共享权重的CNN模型中，计算triplet loss，当模型训练好后，再用属性标注的图片进行正式训练。  
**心得**：  
- video-context信息有助于提高模型表现。该论文一般是均匀间隔25帧抽取视频切片，这是因为周围帧数变化较少，没必要每帧切片都要。另外对于短视频，例如7/8s，只抽视频开头切片，在长一点的视频，开头常是背景，作者选择跳过开头进行切片。  
- 在进行proposal generation时，作者使用了三种颜色类型（HSV，Lab和rgl），同时分别修改它们阈值为50，100，150。  
- 在输入video frame前，作者先将它们resize为256x256x3，然后用227x227分别取四个角和中央的patch，并后续使用水平翻转使得一个frame能抽取到10个patch。 - 在无监督triplet network训练时，帧之间的相似度是以余弦距离衡量的。  
- 在Triplet Selection上，作者先是随机选择进行训练，然后等到10epoch训练收敛差不多了，之后在每个迷你批次里，选择前4的最难训练的negative samples进行训练。
- 如果在triplet network和服装预测模型中使用的CNN模型是在IImageNet上预训练过，结果会有大的提升。
****
[Paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Cheng_Video2Shop_Exact_Matching_CVPR_2017_paper.pdf) 
2017 **Video2shop: Exact matching clothes in videos to online shopping images**  
**关键词**: Video2Shop，[AsymNet](PAPER_IMG/AsymNet.jpg)  
**描述**：该论文应该是基于阿里本身的工作需求提出的，目的是将视频中出现的服装匹配到网络商店内的服装。作者提出了一个网络叫AsymNet，首先使用Faster-RCNN和Kernelized Correlation Filters（KCF）分别对图像和视频中的服装进行检测和追踪，然后使用图像特征网络（IFN）和视频特征网络（VFN）抽取特征，之后图片和视频每个frame下的特征被输入到similarity network node（SNN）去计算它们之间的相似度，然后通过一个由Fusion node（FN）构成双层树结构计算框架，最后得到全局相似度。  
**心得**：  
- AsymNet落地性很强，但训练较繁琐，比如AsymNet要给14个类别单独fine-tuning训练得到14个模型，进而预测结果。
- 在数据预处理上，重复的服装帧数被移除。训练集和测试集划分是4:1。另外，Faster-RCNN有提前在网络购物图片上预训练过。  
- 模型的缺点是在服装检索上，有些衣服的款式是相同的，但颜色不同，模型会把它们认做不匹配。
****
[Paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Yuan_Hard-Aware_Deeply_Cascaded_ICCV_2017_paper.pdf) 
2017 **Hard-aware deeply cascaded embedding**  
**关键词**：[HDC](PAPER_IMG/HDC.jpg)  
**描述**：对于复杂模型，大多数样本属于简单样本，容易收敛但也易过拟合。而对于简单模型，大多数样本被认作难样本，难识别。因此该论文提出HDC集成模型，该模型由K个由简单到复杂的模型组成，首先样本先通过简单模型，根据损失函数，判断样本中是否有该简单模型认做的难样本，有的话，则传递给下一个较复杂的模型计算loss，直到模型遍历完或者模型不再认定该样本为难样本。  
**心得**：一种不同等级的hard samples mining集成思路。
****
[Paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Liu_DeepFashion_Powering_Robust_CVPR_2016_paper.pdf) [Dataset](mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html) [Code](https://github.com/open-mmlab/mmfashion)
2016 **DeepFashion: Powering Robust Clothes Recognition and Retrieval with Rich Annotations**  
**关键词**：DeepFashion, [FashionNet](PAPER_IMG/DeepFashion.jpg)   
**描述**：作者开放关于对衣服的数据集DeepFashion，数据集包含了800,000张图片，且标注了大量的attributes，clothing landmarksh和不同场景下的图片对。t同时还提出了一个多任务通用的模型FashionNet。FashionNet的backbone是VGG-16，在VGG最后一个卷积层被替换成三个分支，它们分别为global apperance branch（用于抽取整个服装商品的全局特征），local appearance branch（在估计的clothing landmarks上池化抽取局部特征），最后一个pose branch用于预测landmark location和landmark visibility（即它们是否被遮挡））。有针对类别，属性，triplet和landmark的损失函数。  
**心得**：  
- 在数据清理上，作者讲图片输入到AlexNet中对比它们fc-responses来检测near- and exact-duplicate图片。  
- landmark location可以有效地处理deformation和pose variation的问题。而如果用人体姿态中的关节去替换使用衣服landmark，FashionNet的表现会下降6-9个百分点。    
- 衣领的检测率比其他衣服部位高，因为衣领随人的脖子，变化单一，而其他衣服部位因人体不同姿态而变化多样。  
- 在衣服检索过程中，衣服尺度的变化严重影响检索表现，其次是衣服背面图。    

