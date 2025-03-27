# VAR 阅读笔记

- 背景

- 原理

- 架构

  - Encoding+VQ：

    - 先进行插值，将特征图 `f` 插值到当前分辨率 `(h_k,w_k)` 

    - 对插值后的 `f` 进行量化

    - 将量化后的 `f` 在 `Codebook` 中查找对应值，得到量化后的特征图 `z`

    - 将 `z` 插值回原始分辨率 `(h_K,w_K)` 

    - 对特征图 `f` 进行残差更新

      - $f=f-\phi_k(z_k)$ ，其中 $\phi_k$ 是
      - 只减去当前尺度的特征，保留之前其他尺度的特征，一方面能够保证模型在下一个更高分辨率的feature map生成过程中能够专注于当前模型没有捕捉到的细节，防止重复提取特征，提高效率，更专注于没有捕捉到的内容，另一方面保留之前其他尺度的特征能够有效保留全局信息，防止丢失低分辨率的信息，低分辨率一般是存储全局结构信息，这样模型就能保留全局结构

      【低分辨率是存储全局结构信息（轮廓，大致形状），中分辨率是存储中等尺度的细节（物体的部分结构），高分辨率是存储图像细节信息（纹理、边缘）】

  - Decoding：

    - 从队列中取出索引矩阵 $r_k$ （$k=0,1,...,K$）
    - 使用 $r_k$ 在 `Codebook` 中查找，得到量化矩阵 $z_k$ 
    - 将 $z_k$ 插值回原始分辨率 $(h_K,w_K)$ 
    - 将 $z_k$ 通过 $\phi_k$ 卷积后加上 $f$ ，$f$ 记录了从低分辨率到高分辨率的的图像信息

  【为什么队列存放的是 $r_k$ 而非 $z_k$ ？——见 `Zotero` 中论文笔记】

- 参数设置

  - 基准学习率 `base_lr`：1e-4，是基于批量大小 `batch_size`：256 来设置的，学习过程中的具体学习率计算公式如下：
    $$
    lr = base\_lr\times\frac{batch\_size}{base\_batch\_size}
    $$

  - batch_size：768~1024

  - epochs：200~350，视模型大小而定

  - optimizer：AdamW，$\beta_1=0.9,\beta_2=0.95,decay=0.05$

  -  

- 实验结果

  - 图像生成效果

    测试数据集：$ImageNet\space 256\times 256$ 与 $512\times 512$ 的条件生成任务 

    模型：`VAR` 模型，深度是 `16` `20` `24` `30` 

    

  - 幂律缩放规律 `Power-law Scaling Laws` 

    数据集：$ImageNet\space training\space set$ ，每轮 `1.28M` 图片

    模型大小：18M~2B，不同大小的模型对应不同的训练轮数，epochs范围是200~350

    - $L-N\space or\space Err-N$ ：参数量 $N$ 的 $scaling \space laws$ 。 数据集使用  $ImageNet$ ，`50000` 张图片，都测试了最后一个尺度的 $L_{last}\space or \space Err_{last}$ 与 $N$ 的关系和所有尺度上的平均 $L_{avg} \space or \space Err_{avg} $ 与 $N$ 的关系

    - $L-C_{min}\space or\space Err-C_{min}$ ：作者通过绘制帕累托边界 $Pareto \space Frontier$ 来确定达到特定性能（特定 $L$ 或 $Err$ ）所需要的 $C_{min}$ ，后续的幂律关系是 $L$ $Err$ 与 $C_{min}$ 的关系，$C$ 的单位是1 $PFlops$ 。$L-C_{min}\space or\space Err-C_{min}$ 的幂律关系的成立是通过 $Pareto\space Frontier$ 与幂律曲线的拟合得出的

      

  - 零样本任务泛化 `Zero-shot Task Generalization` 

    - 任务类型：（1）内绘 `in-painting` （2）外绘 `out-painting` （3）类条件图像编辑 `class-conditional image editing` 

    模型：`VAR-d30`

    

  - 消融实验 `Ablation Study`

    

- 优势