# iKUN 代码库深入解析

 *图1：iKUN 模块与现有多目标跟踪器结合，实现“语言描述 -> 跟踪”的即插即用管道。左侧为人提供的文字描述，传统跟踪器提取视觉轨迹，iKUN 根据文本线索识别并输出匹配描述的目标轨迹。*

## 项目概述

**iKUN (insertable Knowledge Unification Network)** 是一个用于**指称多目标跟踪**（Referring Multi-Object Tracking, RMOT）的深度学习模块。简单来说，给定一段描述性文本（例如“左侧正在转弯的红色汽车”），iKUN 能够在视频中跟踪符合该描述的目标。iKUN 的特色在于它是一个**可插入的模块**：可以无缝集成到现有的多目标跟踪器后，而无需从头重新训练整个跟踪器。它通过预训练的 CLIP 模型提供跨模态特征表示，并引入**知识统一模块 (KUM)** 将文本线索融合到视觉特征中，从而**自适应地提取与描述相关的目标特征**。此外，iKUN 提出了**神经卡尔曼滤波 (NKF)** 来根据运动状态动态调整跟踪过程中的噪声（这一点在代码中未详细体现，可能集成在跟踪器中），以及一种**相似度校准**方法，在测试时利用伪频率信息校正模型对罕见描述的置信度评分。

**主要组成部分**：iKUN 项目代码主要包括以下模块：

- **配置 (`opts.py`)**：定义训练/测试所需的超参数和路径配置。
- **数据加载 (`dataloader.py` & `utils.py`)**：准备训练和测试所需的数据，包括Refer-KITTI数据集的解析、图像预处理和批采样策略等。
- **模型定义 (`model.py`)**：构建 iKUN 的神经网络结构，包含 CLIP 文本/图像编码器，以及不同模式的知识统一模块 (KUM) 实现（如 **级联注意力**、**互相关**、**文本优先调制**三种变体）。
- **损失函数 (`loss.py`)**：定义训练使用的相似度二分类损失（带可选**Focal Loss**调制）。
- **训练过程 (`train.py`)**：训练循环，将数据送入模型、计算损失、反向传播更新参数，并定期评估模型性能。
- **测试与评估 (`test.py` & `similarity_calibration.py`)**：使用训练好的模型对新视频进行文字指称跟踪，包括对跟踪结果打分、可选的置信度校准，以及生成最终输出以评估模型表现。

下面我们将按照代码的执行流程，逐个模块进行详细解析，并穿插关键代码段和注释，帮助理解 iKUN 的实现细节。

## 配置与参数 (`opts.py`)

配置模块使用 `argparse` 定义了运行所需的各种参数。`opts.py` 中构造了一个 `opts` 类，在其初始化函数中添加了各类参数选项，并通过 `parse()` 方法解析命令行参数。主要参数包括：

- **基本设置**：如使用的GPU设备 (`--gpus`，默认`'0,1'`)、随机种子 (`--seed`，默认`1000`)、实验名称 (`--exp_name`，默认`'iKUN'`)、文件保存根路径 (`--save_root`)，以及结果保存后缀等。代码示例：

  ```python
  self.parser.add_argument('--gpus', type=str, default='0,1')
  self.parser.add_argument('--seed', type=int, default=1000)
  self.parser.add_argument('--exp_name', type=str, default='iKUN')
  self.parser.add_argument('--save_root', type=str, default='autodl-fs/iKUN_files')
  ```

  这些参数决定了程序的基本环境，如使用哪些GPU以及结果输出的位置。解析后代码会将 `opt.save_dir` 设为 `save_root/exp_name` 供后续使用。

- **数据与预处理**：如输入图像尺寸列表 `--img_hw`（默认 `[(224,224), (448,448), (672,672)]`，分别可能对应小/中/大尺度图像输入）、图像归一化均值和方差 (`--norm_mean`, `--norm_std`)、数据加载线程数 `--num_workers` 等。还有用于采样视频帧的参数：`--sample_frame_len`（每次采样的连续帧数，默认8）、`--sample_frame_num`（从连续帧中再选取的子帧数量，默认2）、`--sample_frame_stride`（滑动窗口步长，默认4）等。这些参数用于**决定如何从视频序列中采样帧片段**。

- **模型参数**：包括 `--clip_model` 指定使用的 CLIP 模型类型（默认 `'RN50'`，即 ResNet-50，另一可选为 `'ViT-B-32'` 等）、`--feature_dim` 模态统一的特征维度（默认1024）、`--truncation` 文本截断长度（默认10，用于截取部分文本特征）、`--tg_epoch` 文本指导特征融合开始的epoch（默认0，即训练开始就启用文本指导，但可以设为大于0来先训练视觉部分）、`--kum_mode` 知识统一模块模式（默认None，不启用文本引导；可选 `'cascade attention'`、`'cross correlation'`、`'text-first modulation'`）。例如：

  ```python
  self.parser.add_argument('--clip_model', type=str, default='RN50')
  self.parser.add_argument('--kum_mode', type=str, default=None)
  ```

  这些参数控制使用哪种预训练 CLIP 模型和 KUM融合策略。

- **训练超参数**：如批大小 `--train_bs`（默认8）、初始学习率 `--base_lr`（1e-5）及余弦退火最终学习率 `--cosine_end_lr`（0）、AdamW优化的权重衰减 `--weight_decay`（1e-5）、学习率预热设置（`--warmup_epoch` 预热epoch数, `--warmup_start_lr` 预热初始lr）、训练最大 epoch 数 `--max_epoch`（默认100）等。此外还有日志打印频率 `--train_print_freq`、验证评估频率 `--eval_frequency`、模型保存频率 `--save_frequency` 等，用于控制训练输出。

- **损失函数超参数**：`--loss_rho`、`--loss_gamma`、`--loss_reduction` 分别对应损失函数中的 $\rho$（控制正负样本损失权重）、$\gamma$（Focal Loss中的难易调节因子，默认2）、以及损失的reduction方式（默认`'sum'`求和）。

- **恢复训练**：`--resume_path` 若指定则从对应检查点恢复模型参数。

- **测试参数**：包括测试批大小 `--test_bs`（默认1，每次测试一个tracklet）、要加载的模型权重文件名 `--test_ckpt`（默认`iKUN.pth`）以及是否启用相似度校准 `--similarity_calibration`（布尔开关，默认False）。

完成参数定义后，`opts.parse()` 会将命令行或默认值解析为一个 `opt` 对象，并进一步设置：

```python
opt.save_dir = join(opt.save_root, opt.exp_name)
opt.data_root = join(opt.save_root, 'Refer-KITTI')
opt.track_root = join(opt.save_root, 'NeuralSORT')
```

即根据提供的 `save_root`，推导出数据和跟踪结果所在目录（假设数据和NeuralSORT跟踪结果按README要求放置）。最后还通过

```python
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus
```

设置环境变量来限定可用的GPU设备。配置模块在整个项目中提供了统一的 `opt` 变量，供其他模块导入使用（例如 `from opts import opt`），从而让各部分共享这些超参数配置。

总结来说，`opts.py` 为训练和测试提供了**灵活的命令行参数接口**，涵盖了从数据处理到模型结构再到训练细节的各项配置，为后续模块运行做好准备。

## 数据加载与预处理 (`dataloader.py` & `utils.py`)

iKUN 的数据加载模块负责将原始数据（视频帧、跟踪结果、文本描述）组织成模型可训练/测试的输入批次。它包含两个核心数据集类：**RMOT_Dataset**（用于训练，有真实标注）和 **Track_Dataset**（用于测试，有跟踪器输出）。同时，`utils.py` 定义了一些辅助数据结构、转换和全局常量。

### Refer-KITTI 数据及辅助信息

Refer-KITTI 数据集提供了视频帧的标注，包括每个帧中各目标的边界框 (`bbox`)和对应的文字描述标签。`utils.py` 文件中预定义了一些与数据集相关的全局变量：

- `VIDEOS`：划分了训练、验证、测试的视频编号列表。例如 `VIDEOS['train'] = ['0001', '0002', ...]`，`VIDEOS['test'] = ['0005','0011','0013']` 等。这些编号对应KITTI序列。
- `RESOLUTION`：字典，提供每个视频帧图像的原始高宽分辨率，用于将归一化坐标还原为像素坐标。例如 `'0001': (375, 1242)`。
- 文本相关辅助：`WORDS_MAPPING` 定义了一些同义词或复数到单数形式的映射（如 `'cars'->'car'`, `'people'->'pedestrian'` 等)；`WORDS` 列出了一些需要过滤的停用词（`'dropped'`列表）以及可能的描述类别（如`'color'`, `'direction'`, `'status'`等各自的候选词）。
- `EXPRESSIONS`：字典，列出了各数据划分中可能出现的描述短语列表。例如 `'train'`键对应训练集中出现的所有描述短语组合。**注意**：在代码的 track 数据集中会用到 `EXPRESSIONS[video]`，表示某视频相关的所有描述（实际实现中，`EXPRESSIONS` 可能在读取数据时动态扩展，每个视频对应其包含的描述集合）。
- `ID2EXP`：将每条可能的描述短语映射到一个ID的字典 ([utils.py](file://file-8rluj8uhpg816kgadnzbbx%23:~:text=id2exp = ,7: 'left car in black/))等（训练时将选定描述转换为ID）。这在RMOT_Dataset中用于记录目标描述的索引，方便计算指标。
- 函数 `expression_conversion(expression)`：执行文本标准化，将描述中的连字符`-`替换为空格，并处理特定短语（如 `"light color"` -> `"light-color"`），然后替换同义词为统一用词 ([utils.py](file://file-8rluj8uhpg816kgadnzbbx%23:~:text=def expression_conversion(expression): ,word}/))。这一函数用于**规范化文本描述**，确保风格一致，便于匹配和嵌入。
- 类 `SquarePad`：图像变换类，将输入PIL图像四周填充黑边成正方形（边长等于长宽最大值），以便后续缩放不会改变原始图像比例 ([utils.py](file://file-8rluj8uhpg816kgadnzbbx%23:~:text=,pad(image, padding, 0, 'constant/))。在transform中先调用SquarePad保证框外区域填充，再进行Resize/Crop。
- 函数 `tokenize(text)`：封装了`clip.tokenize`，将输入字符串列表转换为CLIP模型需要的token张量 ([utils.py](file://file-8rluj8uhpg816kgadnzbbx%23:~:text=def tokenize(text): token = clip,return token/))。模型输入需要将文字转为token序列（长度77，上下文标准长度），这个函数直接调用了CLIP提供的tokenizer。

### 训练数据集：RMOT_Dataset

**RMOT_Dataset** 类继承自 `torch.utils.data.Dataset`，用于提供**带标注的训练样本**。其目标是：对每个视频的每个目标轨迹，采样若干片段，并为其中的某一帧选择一条描述性短语作为监督标签（正例或负例），从而构建**二分类问题**（描述是否匹配当前目标）。

- **初始化**：`RMOT_Dataset(mode, opt, only_car=False)` ([dataloader.py](file://xn--file-wbafkg6fnayniaswptm9mm%23:~:text=def __init__,expression-xt13it15ieculp7lguo9a601etr5b/))
   构造函数接受 `mode`（'train' 或 'test'）和配置 `opt`。`only_car` 可以用于过滤仅包含 "car" 类目标的描述（默认为 False）。初始化时会：

  - 调用 `get_transform(mode, opt, idx)` 生成图像变换管道 `self.transform`，这里 `idx` 取值0、1、2 分别对应 `opt.img_hw` 列表中的三种尺寸 ([dataloader.py](file://file-wbafkg6fnayniaswptm9mm%23:~:text=def get_transform,norm_std/))。具体来说：
    - **训练模式** (`mode=='train'`)：transform0 包括 `SquarePad` + `RandomResizedCrop(opt.img_hw[idx], ratio=opt.random_crop_ratio)` 随机裁剪到指定尺寸（例如224） + 转Tensor + 按`norm_mean`/`norm_std`归一化 ([dataloader.py](file://file-wbafkg6fnayniaswptm9mm%23:~:text=def get_transform,norm_std/))。随机裁剪用于数据增强。
    - **测试模式**：transform0 用 `SquarePad` + `Resize(opt.img_hw[idx])` 固定缩放到目标尺寸 + Tensor + 归一化 ([dataloader.py](file://file-wbafkg6fnayniaswptm9mm%23:~:text=,idx]), t.totensor(), t.normalize(opt.norm_mean, opt.norm_std),/))。
    - transform2 则对应 `opt.img_hw[2]`（最大尺寸，如672），用于处理全局大图。同样根据训练/测试决定是RandomResizedCrop还是Resize ([dataloader.py](file://file-wbafkg6fnayniaswptm9mm%23:~:text=return t.compose(,idx]), t.totensor(), t.normalize(opt.norm_mean, opt.norm_std/))。transform1（中等尺寸448）在本项目中可能未显式用到，但也预生成。
    - 另外还有 `'unnorm'` 模式 ([dataloader.py](file://file-wbafkg6fnayniaswptm9mm%23:~:text=elif mode == 'unnorm': mean,/))，用于将标准化图像还原回原始像素值（测试保存结果用）。
  - 设置 `self.exp_key = 'expression_new'` 作为标注中使用的文本描述字段名（表示预处理后的描述）。
  - 调用 `self._parse_data()` 解析数据，结果保存在 `self.data` 字典中，并生成 `self.data_keys` 列表记录所有样本的键。`self.exp2id` 则利用 `ID2EXP` 把每个描述短语映射到其索引，便于后续将文字标签转成数值。

- **数据解析**：`_parse_data(self)` ([dataloader.py](file://file-wbafkg6fnayniaswptm9mm%23:~:text=def _parse_data(self): labels = json,listdir(join(expression_dir, video/)) ([dataloader.py](file://file-wbafkg6fnayniaswptm9mm%23:~:text=h, w = resolution,only_car)/))
   该函数读取标注文件并组织训练样本数据。关键步骤：

  1. 加载整个 Refer-KITTI 数据的标签JSON（`Refer-KITTI_labels.json`） ([dataloader.py](file://file-wbafkg6fnayniaswptm9mm%23:~:text=def _parse_data(self): labels = json,listdir(join(expression_dir, video/))。这个文件应包含所有视频、目标、帧的标注信息（每帧包含bbox和对应描述短语列表等）。
  2. 准备一个二级嵌套的默认字典 `data = multi_dim_dict(2, list)` 用于存储 **{对象 -> 帧序列数据}**，以及一个 `target_expressions = defaultdict(list)` 来存储每个视频对应的**目标短语全集**。
  3. **收集视频级描述全集**：遍历 `VIDEOS[self.mode]` 中每个视频，读取其 `expression/` 目录下的所有描述文件（每个文件名即一种描述，内容包含该描述在哪些帧出现等）。将每个描述文件名去掉扩展名得到描述短语 `expression`，用 `expression_conversion` 得到标准化的 `expression_new` ([dataloader.py](file://file-wbafkg6fnayniaswptm9mm%23:~:text=for video in videos[self.mode]: ,video/))。将每个规范化描述加入 `target_expressions[video]` 列表。这样得到该视频可能出现的所有描述短语集合。
  4. **遍历目标轨迹**：对于每个视频，再遍历其标签JSON中的每个目标 `obj_id` 及其帧标注 `obj_label` ([dataloader.py](file://file-wbafkg6fnayniaswptm9mm%23:~:text=for obj_id, obj_label in labels,only_car) ): num += 1/))：
     - 计算该目标轨迹中有标注类别的帧数 `num`（`value['category']`非空的帧）。如果设置了`only_car`则仅统计类别为 `'car'` 的帧。
     - 跳过帧数过少的目标：如果 `num <= sample_frame_len` 或总帧数 `len(obj_label) <= sample_frame_len`，则该目标轨迹长度太短，不足以采样一个片段（少于连续8帧），忽略之 ([dataloader.py](file://file-wbafkg6fnayniaswptm9mm%23:~:text=if len(value['category']) ,self.opt.sample_frame_len/))。
     - 否则，初始化一个键 `obj_key = "{video}_{obj_id}"` 表示唯一目标，并准备临时字典 `curr_data` 用来收集此目标的帧序列数据。然后遍历该目标的每个帧 `frame_id` 及标注 `frame_label` ([dataloader.py](file://file-wbafkg6fnayniaswptm9mm%23:~:text=for frame_id, frame_label in obj_label,video], self.exp_key, self.only_car/))（注意代码保证了帧ID单调递增 ([dataloader.py](file://file-wbafkg6fnayniaswptm9mm%23:~:text=,video], self.exp_key, self.only_car/))）：
       - **筛选目标描述**：调用 `filter_target_expressions(frame_label, target_expressions[video], self.exp_key, self.only_car)` ([dataloader.py](file://file-wbafkg6fnayniaswptm9mm%23:~:text=,video], self.exp_key, self.only_car/))，获取 `tgt_exps`（候选描述）和对应 `tgt_labels` 列表。这个函数会根据给定帧的标注和视频全局描述列表，返回当前帧可能的**目标描述短语列表**以及每个短语是否适用于该帧的标签（1表示该描述在此帧真实存在，0表示不存在）。实现上：对于视频的每个候选短语，若 `only_car` 为 True 则要求短语中包含"car"词；然后如果该短语存在于此帧的GT描述集合 `GT_EXPRESSIONS` 就标记为1，否则为0 ([dataloader.py](file://file-wbafkg6fnayniaswptm9mm%23:~:text=gt_expressions = gt,append(0) return out_exps, out_labels/))。这样 `tgt_exps` 覆盖了视频中所有潜在描述，并标记出当前帧哪些描述是真的。
       - 如果没有任何 `tgt_exps`（即没有描述包含"car"或其他原因），跳过该帧 ([dataloader.py](file://file-wbafkg6fnayniaswptm9mm%23:~:text=tgt_exps, tgt_labels = filter_target_expressions,tgt_exps) == 0: continue/))。
       - **获取GT描述**：从帧标注中取出真正的描述列表 `exps = frame_label['expression_new']`，并用 `filter_gt_expressions(exps, None)` ([dataloader.py](file://file-wbafkg6fnayniaswptm9mm%23:~:text=,save curr_data['expression'].append(exps/))过滤得到最终**真实描述列表**（第二个参数None表示不过滤，只是返回原列表或做一些映射处理）。如果该帧没有任何真实描述（`len(exps)==0`），也跳过 ([dataloader.py](file://file-wbafkg6fnayniaswptm9mm%23:~:text=,exps) == 0: continue/))。
       - **获取边界框**：取出当前帧bbox坐标 `(x,y,w,h) = frame_label['bbox']` ([dataloader.py](file://file-wbafkg6fnayniaswptm9mm%23:~:text=,h/))。结合上面的 `RESOLUTION[video] = (H, W)`，将归一化坐标换算成像素坐标：左上角 `(x*W, y*H)`，右下角 `((x+w)*W, (y+h)*H)`，并记录帧ID。
       - **保存帧数据**：将上述信息添加到 `curr_data` 列表字典中：
         - `curr_data['expression']`：真实描述列表（可能多条）。
         - `curr_data['target_expression']`：候选描述短语列表（全局的）
         - `curr_data['target_labels']`：候选描述对应的0/1标签列表
         - `curr_data['bbox']`：该帧的边界框 [frame_id, x1, y1, x2, y2]
            每处理完一帧就 append 到各自列表末尾 ([dataloader.py](file://file-wbafkg6fnayniaswptm9mm%23:~:text=curr_data,obj_key] = curr_data.copy() return data/))。
     - 遍历完此目标的所有帧后，如果收集的帧数超过 `sample_frame_len`，则将该目标的数据 `curr_data` 存入 `data[obj_key]` ([dataloader.py](file://file-wbafkg6fnayniaswptm9mm%23:~:text=curr_data['bbox'].append([frame_id, x ,obj_key] = curr_data.copy() return data/))。这表示该目标有足够长的轨迹可供训练采样。

  `_parse_data` 最终返回整理好的 `data` 字典。处理完所有视频，将有若干键（如 `"0001_3"` 表示视频0001中目标ID=3的轨迹）可用于训练。每个键对应的值 `curr_data` 包含该轨迹的一系列帧的信息（bbox列表、每帧的真实表达、候选表达及标签）。

- **获取数据长度**：`__len__(self)` 返回 `len(self.data_keys)` ([dataloader.py](file://file-wbafkg6fnayniaswptm9mm%23:~:text=def __len__(self): return len(self/))。即数据集中**轨迹片段样本**的数量（每个目标轨迹算一个基础样本，还会在getitem中动态采样片段）。

- **索引取样**：`__getitem__(self, index)` ([dataloader.py](file://file-wbafkg6fnayniaswptm9mm%23:~:text=def __getitem__,data_key/)) ([dataloader.py](file://file-wbafkg6fnayniaswptm9mm%23:~:text=if self.mode == 'train': ,1/))这是核心，将一个轨迹数据进一步加工成模型可直接输入的张量：

  1. **选取轨迹**：根据索引取对应的 `data_key` 和 `data = self.data[data_key]`，并解析出视频号 `video` ([dataloader.py](file://file-wbafkg6fnayniaswptm9mm%23:~:text=def __getitem__,data_key/))。这里 `data` 包含整个轨迹所有帧的列表。

  2. **随机采样帧片段**：取该轨迹总帧数 `data_len = len(data['bbox'])` 和预设的连续采样长度 `sample_len = opt.sample_frame_len`（如8）以及采样帧数 `sample_num = opt.sample_frame_num`（如2）。初始化空列表 `sampled_indices` 用于存储选中的帧索引。然后分情况：

     - **训练模式**：随机截取一个长度为sample_len的连续子序列 ([dataloader.py](file://file-wbafkg6fnayniaswptm9mm%23:~:text=if self.mode == 'train': ,1/))。实现上：

       - `start_idx = random.randint(0, data_len - sample_len)` 随机选择片段起点索引；

       - `stop_idx = start_idx + sample_len - 1`；

       - 计算步长 `step = sample_len // sample_num`（例如8帧选2帧则 step=4）。然后在每个步长区间内随机取一帧索引：

         ```python
         for idx in range(start_idx, stop_idx+1, step):
             sampled_indices.append(random.randint(idx, idx + step - 1))
         ```

         这样如果sample_len=8、step=4，将在`[start_idx, start_idx+3]`内随机取一帧，在`[start_idx+4, start_idx+7]`内随机取一帧，共2帧，模拟从片段的前后部分各随机采样。

     - **测试模式**：为了可重复性，按固定策略采样 ([dataloader.py](file://file-wbafkg6fnayniaswptm9mm%23:~:text=elif self.mode == 'test': ,append(idx + step %2F%2F 2/))：

       - `start_idx = index % (data_len - sample_len)` 这样如果遍历数据时重复使用同一轨迹，可以不同起点地滑窗采样（不过通常测试每轨迹只用一次，这里的取模保证索引大于轨迹数时循环滑动）。

       - `stop_idx = start_idx + sample_len - 1`；同样 `step = sample_len // sample_num`；然后在每个区间取中点帧：

         ```python
         for idx in range(start_idx, stop_idx+1, step):
             sampled_indices.append(idx + step // 2)
         ```

         例如8帧选2帧，则取`start_idx+2`和`start_idx+6`（各区间的中间位置）。

     - 经过以上，两种模式下 `sampled_indices` 都得到长度为 `sample_num`（2）的帧索引列表。

  3. **加载帧图像**：根据采样到的帧索引列表，读取对应的图像文件 ([dataloader.py](file://file-wbafkg6fnayniaswptm9mm%23:~:text=,for idx in sampled_indices/))：

     ```python
     images = [
         Image.open(join(opt.data_root,
                         f'KITTI/training/image_02/{video}/{frame_id:06d}.png'))
         for idx in sampled_indices
         for frame_id in [ data['bbox'][idx][0] ]
     ]
     ]  
     ```

     每个 `data['bbox'][idx]` 的第0位存储了帧ID（帧号），用它拼接成文件路径读取图片。结果 `images` 列表包含该片段中采样的两帧图像 (PIL格式)。

  4. **汇总描述文本**：初始化 `expressions = []`，遍历 `sampled_indices`，将每个索引对应帧的所有真实描述 `data['expression'][idx]` 扩展加入 `expressions` 列表 ([dataloader.py](file://file-wbafkg6fnayniaswptm9mm%23:~:text=,idx]) expressions = sorted(list(set(expressions/))。然后取集合去重并排序：

     ```python
     expressions = sorted(list(set(expressions)))
     ```

     这样 `expressions` 保存了该片段里出现过的所有GT描述短语（字符串列表）。随后代码将其用逗号拼接成单一字符串返回（见最后返回部分）。

  5. **裁剪目标图像**：对每帧图像，按照其bbox裁剪出目标区域的小图：

     ```python
     cropped_images = self._crop_image(images, sampled_indices, data, 'small')
     ```

     其中 `_crop_image` 函数根据模式`'small'`对每对 (image, idx) 裁剪 ([dataloader.py](file://file-wbafkg6fnayniaswptm9mm%23:~:text=def _crop_image,elif mode == 'big/))：

     - `'small'` 模式下直接使用每帧自己的 bbox：`image.crop(data['bbox'][idx][1:])` 获取目标区域（注意bbox存储格式为[frame, x1, y1, x2, y2]），然后施加 transform[0]（将尺寸缩放到224并归一化）。所有裁剪后的Tensor通过 `torch.stack(..., dim=0)` 堆叠成形状 `[T, C, H, W]`，这里 T = sample_num 帧数。例如 (2, 3, 224, 224)。
     - （另有 `'big'` 模式用于裁剪包含片段中所有帧目标的**大视野区域**，在RMOT_Dataset中未用到，略过）。

  6. **处理全局图像**：同时将每帧完整图像按较大尺寸transform2处理：

     ```python
     global_images = torch.stack([ self.transform[2](img) for img in images ], dim=0)
     ```

     这样得到形状 `[2, 3, H2, W2]` （例如2帧，每帧3通道，672x672）的大图张量。`global_images` 保留了目标所在帧的全局场景，用于模型融合局部与全局特征。

  7. **选取目标描述**：这是关键一步，为当前样本选择一个“目标描述”及其标签：

     - 如果训练模式：随机选择此片段中的一个帧索引 `idx` 作为“目标帧” ([dataloader.py](file://file-wbafkg6fnayniaswptm9mm%23:~:text=if self,1 sampled_target_idx = choice/))；测试模式则固定使用片段中心帧（sampled_indices列表中间那个） ([dataloader.py](file://file-wbafkg6fnayniaswptm9mm%23:~:text=idx = choice(sampled_indices, size=1),idx/))。

     - 提取该帧对应的候选描述列表和标签列表：

       ```python
       target_expressions = data['target_expression'][idx]
       target_labels = data['target_labels'][idx]
       ```

       这里 `target_expressions` 列出可能描述当前帧的短语全集，`target_labels` 列出每个短语是真(1)/假(0)标签。对于训练，往往只有一两个短语为真，其余为假；对于测试，我们会用到全部。

     - **训练**：因为训练每次只针对一个描述进行二分类，所以进一步**随机抽取1条**描述作为本次样本的“查询描述” ([dataloader.py](file://file-wbafkg6fnayniaswptm9mm%23:~:text=if self,i] for i in sampled_target_idx/))：

       ```python
       sampled_target_idx = choice(range(len(target_expressions)), size=1)
       sampled_target_exp = [ target_expressions[i] for i in sampled_target_idx ]
       sampled_target_label = [ target_labels[i] for i in sampled_target_idx ]
       exp_id = self.exp2id[sampled_target_exp[0]]
       ```

       这样选出一个描述短语及对应标签（1或0）。`exp_id`记录该描述的索引ID。由于 `opt.sample_expression_num==1`（仅采一个描述），这里取一个无需去重。

     - **测试**：不抽样，`sampled_target_exp = target_expressions`（保留该帧的所有候选描述），`sampled_target_label = target_labels`（对应的一组0/1标签），并将 `exp_id = -1` 作为占位符 ([dataloader.py](file://file-wbafkg6fnayniaswptm9mm%23:~:text=exp_id = self.exp2id,1/))。即测试时每个样本会包含**多个描述的判断**。

     经过这一步：

     - **训练样本**含一个目标描述及其标签，表示“这个轨迹片段中选定帧是否具有某描述属性”，模型需要输出相应的高/低相似度。
     - **测试样本**则携带一个帧的多条描述及标签，用于后续计算各描述的匹配分数。

  8. **整理输出**：函数最终返回一个字典 ([dataloader.py](file://file-wbafkg6fnayniaswptm9mm%23:~:text=sampled_target_label = torch,join(sampled_target_exp), target_labels=sampled_target_label/))：

     - `'cropped_images'`: 裁剪的小图张量 (shape [T,3,224,224])
     - `'global_images'`: 全局图张量 (shape [T,3,H2,W2], 如H2=W2=672)
     - `'expressions'`: 片段内真实描述的合集字符串（逗号分隔）
     - `'target_expressions'`: 选定的目标描述字符串（训练时就是一条，测试时可能是多个用逗号连接）
     - `'target_labels'`: 目标描述对应的标签张量（训练shape [1]，测试shape [N_descriptions]）
     - `'expression_id'`: 描述ID（训练为相应索引，测试为 -1）
     - 还有 `'start_idx'`, `'stop_idx'`, `'data_key'` 等信息用于参考。

  如此，每次 `__getitem__` 从一个完整轨迹中采样出一个**包含若干帧的小片段**及其相关文字描述，用于训练模型理解“视觉片段 vs. 文本描述”的关系。

**小结**：RMOT_Dataset 将**视频序列监督信息转化为训练对**。模型输入由两部分图像组成：`cropped_images` 提供了目标的局部视图，`global_images` 提供了场景的全局视野；同时输入的文本(`target_expressions`)描述了目标的属性或状态，并带有标签(`target_labels`)指示描述是否真实。训练时每个样本是一对图像片段和一句描述（真假作为监督），测试时样本可包含多描述以评估每个描述的匹配程度。

### 测试数据集：Track_Dataset

**Track_Dataset** 类同样继承自 `Dataset`，用于**测试阶段**结合跟踪器输出进行推断。它利用**跟踪器（如 NeuralSORT）提供的目标轨迹**，为每个轨迹片段搭配所有可能的描述查询，生成需要模型判别的组合。

- **初始化**：`Track_Dataset(mode, opt)` ([dataloader.py](file://file-wbafkg6fnayniaswptm9mm%23:~:text=class track_dataset,self.data = self._parse_data/))
   仅需 opt 和模式（'test'或'val'）。内部：

  - 调用 `get_transform` 设置 `self.transform`（0,1,2三种，与RMOT类似）。
  - 调用 `self._parse_data()` 解析跟踪结果数据，存入 `self.data` 列表。

- **解析跟踪数据**：`_parse_data(self)` ([dataloader.py](file://file-wbafkg6fnayniaswptm9mm%23:~:text=for video in videos[self.mode]: ,empty((0, 10)) max_obj_id = 0/)) ([dataloader.py](file://file-wbafkg6fnayniaswptm9mm%23:~:text=tracks_2 = np,for obj_id in ids/))
   这一函数读取跟踪器的输出文本（predict.txt）来构建测试样本：

  1. 设定采样长度和步长：`sample_length = opt.sample_frame_len`，`sample_stride = opt.sample_frame_stride`（如8和4）。

  2. 初始化空列表 `DATA` 用于存放所有样本条目。

  3. 遍历每个测试/验证视频 `for video in VIDEOS[self.mode]`:

     - **读取跟踪结果**：假设跟踪器输出在 `opt.track_root` 下，对每个视频有 `car/predict.txt` 和 `pedestrian/predict.txt` 两类文件。代码通过 `np.loadtxt` 分别读入车辆和行人轨迹 ([dataloader.py](file://file-wbafkg6fnayniaswptm9mm%23:~:text=tracks_1 = np,track_root, video, 'pedestrian/))。格式每行可能为 `[frame, obj_id, x, y, w, h, ...]`（KITTI MOT格式有10列，后面可能有置信度等字段）。

       - 若读到的数组维度不是2D（无轨迹）则设为空数组。将行人轨迹的 obj_id 偏移（`tracks_2[:,1] += max_obj_id`）避免ID冲突，然后合并车和人 ([dataloader.py](file://file-wbafkg6fnayniaswptm9mm%23:~:text=tracks_2 = np,concatenate((tracks, tracks_2), axis=0/))。
       - 对合并轨迹按 (obj_id, frame) 排序 ([dataloader.py](file://file-wbafkg6fnayniaswptm9mm%23:~:text=tracks = np,1/))确保每个目标帧序有序。

     - **分割连续轨迹段**：取出所有目标ID集合。对于每个 `obj_id` ([dataloader.py](file://file-wbafkg6fnayniaswptm9mm%23:~:text=ids = set(tracks,1e5/))：

       - 提取该ID的所有帧记录 `tracks_id = tracks[tracks[:,1] == obj_id]`，找到起止帧 `frame_min, frame_max` ([dataloader.py](file://file-wbafkg6fnayniaswptm9mm%23:~:text=tracks_id = tracks,0/))。

       - 检查轨迹连续性：遍历帧序列，找到中断的地方，将长轨迹拆分成若干连续子轨迹段 `frame_pairs` ([dataloader.py](file://xn--file-wbafkg6fnayniaswptm9mm%23:~:text=%23 ,sub,frame_idx previous_frame = frame_idx-th70m0vsbllrtkqu4do03nczqhb8ar361bbzpelr7obkna9j11abzzcftm/))。逻辑：如果下一帧不等于当前帧+1，就认为上一个段结束、新段开始。`frame_pairs` 收集每段的起止帧号。

       - **滑窗片段**：对于每个连续段 `(f_min, f_max)`，进一步按照 `sample_stride` 划窗提取长度为 `sample_length` 的子轨迹 ([dataloader.py](file://file-wbafkg6fnayniaswptm9mm%23:~:text=,range(f_start, f_stop + 1/))：

         - 初始化 `total_length` 计算总帧数校验。

         - 从 `f_min` 开始，每隔 `sample_stride` 帧取一个 `f_idx` 作为片段的结尾候选帧：

           ```python
           for f_idx in range(f_min, f_max+1, sample_stride):
               f_stop = min(f_max, f_idx + sample_length - 1)
               f_start = max(f_min, f_stop - sample_length + 1)
               ...
           ```

           这段代码对于每个f_idx确定一个片段的实际起止帧 `[f_start, f_stop]`，长度不超过sample_length。一般来说，它在轨迹上以步长滑动窗口，每段覆盖 `sample_length` 帧（最后一段可能不足也取到末尾）。

           - 从该目标的所有帧记录 `tracks_id` 中筛选出帧号在 `[f_start, f_stop]` 范围内的行，得到 `tracklets` ([dataloader.py](file://file-wbafkg6fnayniaswptm9mm%23:~:text=f_start = max(f_min, f_stop ,2:4] tracklets = tracklets.astype(int/))。`tracklets` 是一个形状 `(L, 6)` 的数组，包含该目标在这段中的每帧检测：[frame, id, x, y, x2, y2]。代码计算 `tracklets[:,4:6] += tracklets[:,2:4]` 将宽高转为x2,y2（右下角）坐标，然后转为int。并断言片段长度一致。

           - **配对描述**：对于每个描述 `expression` 在 `EXPRESSIONS[video]` 列表中的条目 ([dataloader.py](file://file-wbafkg6fnayniaswptm9mm%23:~:text=for expression in expressions,start_frame=f_start, stop_frame=f_stop, tracklets=tracklets, expression=expression,/))，都创建一个样本字典加入 `DATA`：

             ```python
             DATA.append({
                 'video': video,
                 'obj_id': int(obj_id),
                 'start_frame': f_start,
                 'stop_frame': f_stop,
                 'tracklets': tracklets,
                 'expression': expression,
             })
             ```

             即**每个轨迹子段**与**每条候选描述**组合，形成一个需要模型评分的实例。由于没有提前知道哪条描述是真正对应该目标，该设计相当于生成**所有可能的查询-轨迹对**，让模型去判断匹配程度。

           - 若一个子段已到轨迹尾（`f_stop == f_max`），则跳出内部循环，进入下一个轨迹段 ([dataloader.py](file://file-wbafkg6fnayniaswptm9mm%23:~:text=expression=expression, ,f_max: break/))。

         - 最后检查 `total_length == len(tracks_id)`确保所有帧都覆盖（完整性校验）。

     - 这样对每个目标的所有子轨迹都配上所有描述短语，存入DATA列表。

  `_parse_data` 返回 `DATA` 列表，每个元素是一个字典，表示**“某视频的某目标一段帧序 + 一个描述”**的组合。对于包含很多可能描述的序列，该列表会相当长（但提供了全面的查询组合）。

- **数据长度**：`__len__(self)` 返回 `len(self.data)` ([dataloader.py](file://file-wbafkg6fnayniaswptm9mm%23:~:text=def __len__(self): return len(self/))。

- **取样实现**：`__getitem__(self, index)` ([dataloader.py](file://file-wbafkg6fnayniaswptm9mm%23:~:text=def __getitem__,tracklets/)) ([dataloader.py](file://file-wbafkg6fnayniaswptm9mm%23:~:text=,for bbox in sampled_tracklets/))
   对于Track_Dataset，index直接对应DATA中的一个组合：

  1. 取出对应的 `data = self.data[index]` 字典，解析出 `video, obj_id, start_frame, stop_frame, tracklets, expression` ([dataloader.py](file://file-wbafkg6fnayniaswptm9mm%23:~:text=def __getitem__,tracklets/))。其中 `tracklets` 是前面得到的帧数组（每行 frame,id,x1,y1,x2,y2），`expression` 是一个描述短语字符串。

  2. **文本标准化**：对 `expression` 调用 `expression_conversion` 得到规范形式 `expression_converted` ([dataloader.py](file://file-wbafkg6fnayniaswptm9mm%23:~:text=,expression/))。存储原始和规范版本都有，便于一致性。

  3. **帧采样**：由于Track_Dataset每个样本**本身已是一个轨迹段**，这里不需要再随机选择连续帧段，但为了和训练时输入形式一致，也取一定数量的帧：

     - 直接用 `np.linspace(0, len(tracklets), sample_frame_num, endpoint=False, dtype=int)` 均匀采样索引 ([dataloader.py](file://file-wbafkg6fnayniaswptm9mm%23:~:text=,sampled_indices/))。例如若tracklets长度8、sample_frame_num=2，会得到索引[0,4]。
     - 用这些索引提取 `sampled_tracklets = tracklets[sampled_indices]`，相当于从该段中挑选两帧来代表。

  4. **加载图像**：类似RMOT_Dataset，对每个选中的帧，读取对应图像 ([dataloader.py](file://file-wbafkg6fnayniaswptm9mm%23:~:text=,for bbox in sampled_tracklets/))：

     ```python
     images = [
         Image.open(join(opt.data_root, f'KITTI/training/image_02/{video}/{frame:06d}.png'))
         for frame, *_ in sampled_tracklets
     ]
     ```

     这里直接从 tracklets 数组里拿出帧号 `bbox[0]` 来读图片。得到 `images` 列表长度为2。

  5. **裁剪与变换**：

     - `'cropped_images'`: 对每张image按照对应的tracklet bbox裁剪，并用transform0处理 ([dataloader.py](file://file-wbafkg6fnayniaswptm9mm%23:~:text=,dim=0/))：

       ```python
       cropped_images = torch.stack([
           self.transform[0](img.crop(bbox[2:6]))
           for img, bbox in zip(images, sampled_tracklets)
       ], dim=0)
       ```

       注意这里 `bbox[2:6]` 已经是[x1,y1,x2,y2]像素坐标，所以直接crop。结果是大小为2的张量列表（2帧目标小图）。

     - `'global_images'`: 将每张原图用transform2处理后堆叠 ([dataloader.py](file://file-wbafkg6fnayniaswptm9mm%23:~:text=,dim=0/))：

       ```python
       global_images = torch.stack([ self.transform[2](img) for img in images ], dim=0 )
       ```

       以获得对应的大图。

  6. **返回**：和RMOT_Dataset不同，这里每个样本对应一个描述，无需选择子描述。函数返回字典 ([dataloader.py](file://file-wbafkg6fnayniaswptm9mm%23:~:text=return dict,cropped_images=cropped_images, global_images=global_images, expression_raw=expression, expression_new=expression_converted,/))包含：

     - `'video', 'obj_id', 'start_frame', 'stop_frame'`: 标识这个轨迹段信息
     - `'cropped_images', 'global_images'`: 两个帧的图像张量
     - `'expression_raw'`: 原始描述短语字符串
     - `'expression_new'`: 规范化后的描述短语字符串

Track_Dataset 得到的数据样本基本格式与RMOT_Dataset相似（都有cropped_images、global_images、expression等），只是**每个样本固定对应单一的查询描述**，而RMOT_Dataset训练时实际用到的也是每样本单一描述（但测试模式下RMOT_Dataset会成批返回多描述以算准确率）。Track_Dataset 的作用是在**测试阶段**生成**“(轨迹段, 候选描述)”**对，让模型对每一对输出一个相似度评分，用于最终决定该描述是否属于该轨迹。

### 数据加载器和其他工具

`dataloader.py` 中还提供了简单的工厂函数：

```python
def get_dataloader(mode, opt, dataset='RMOT_Dataset', show=False):
    dataset = eval(dataset)(mode, opt, **kwargs)
    ...
    dataloader = DataLoader(dataset, batch_size=..., shuffle=..., num_workers=...)
    return dataloader
```

方便根据名称实例化数据集并封装为 PyTorch DataLoader。在训练中会用 `get_dataloader('train', opt, 'RMOT_Dataset')` 获取训练数据迭代器，测试中用 `'Track_Dataset'` 获取测试数据迭代器等。`show=True`时会调用数据集对象的 `show_information()` 打印数据集大小等信息 ([dataloader.py](file://file-wbafkg6fnayniaswptm9mm%23:~:text=def show_information(self): print( f'===> refer,/))。

总结本节：数据处理模块先通过解析标注/跟踪结果准备了**丰富的字典结构数据**，RMOT_Dataset 关注真实标注，用于监督训练模型区分正确/错误的描述，Track_Dataset 则基于跟踪器输出生成测试查询。随后，`__getitem__` 方法将这些数据转为**模型输入**（裁剪图像+全局图像+文本描述），确保训练和测试阶段数据形式一致，为模型的视觉-语言融合做准备。

## 模型定义 (`model.py`)

iKUN 的模型建立在 OpenAI 的 **CLIP (Contrastive Language-Image Pre-training)** 模型之上。CLIP 提供了预训练的图像编码器和文本编码器，可以将图像和文本投影到同一特征空间，使得描述对应图像时二者特征相似。iKUN 利用 CLIP 作为 backbone，结合**知识统一模块 (KUM)** 实现文本对视觉特征的引导。模型主要分为三部分：**特征提取（CLIP）**、**跨模态融合（KUM）**、**相似度计算**。

### CLIP 加载与特征投影

- **加载预训练 CLIP**：`model.py` 中定义了函数 `load_clip(model_path, input_resolution)` ([model.py](file://file-qsvqjbjmlsvhiym459fo7w%23:~:text=def load_clip,state_dict/)) ([model.py](file://file-qsvqjbjmlsvhiym459fo7w%23:~:text=model = myclip,vocab_size, transformer_width, transformer_heads, transformer_layers/))用于载入 CLIP 模型权重（TorchScript格式 `.pt` 文件）。根据权重判断模型类型（ViT或ResNet）并动态构造 CLIP 模型结构，然后使用 `convert_weights` 将参数转换为fp32，`model.load_state_dict` 加载权重 ([model.py](file://file-qsvqjbjmlsvhiym459fo7w%23:~:text=for key in [,key/))。通过传入 `input_resolution=224`，函数会调整模型预期的输入图像分辨率（对于ResNet50，CLIP默认image_resolution=224，对ViT-B/32也是224）。返回的模型被封装在一个自定义的 `MyCLIP` 类中 ([model.py](file://file-qsvqjbjmlsvhiym459fo7w%23:~:text=class myclip(clip): def __init__(self, ,args/))，此类继承自 CLIP 并添加了一些方法（如 `encode_text_2`）以便获取文本的局部特征表示。

- **Model 类初始化**：`class Model(nn.Module)` ([model.py](file://file-qsvqjbjmlsvhiym459fo7w%23:~:text=class model(nn,float/)) ([model.py](file://file-qsvqjbjmlsvhiym459fo7w%23:~:text=self.clip = load_clip( join(opt.save_root, f'clip%2F,get_text_fc(use_ln=false/))
   初始化步骤：

  1. 调用 `load_clip(join(opt.save_root, f'CLIP/{opt.clip_model}.pt'), input_resolution=224)` 加载指定的CLIP模型（如RN50.pt） ([model.py](file://file-qsvqjbjmlsvhiym459fo7w%23:~:text=self,text_dim = 1024/))。得到的 `self.clip` 包含图像编码器 `clip.visual` 和文本编码器 `clip.encode_text` 等。`self.clip = self.clip.float()` 保证使用float32精度。
  2. 定义特征维度：`self.img_dim = 2048`，`self.text_dim = 1024`。对于RN50，图像特征输出维度是2048（ResNet最后一层池化输出），文本特征维度是1024（CLIP文本embedding维度）。
  3. 定义全连接投影层：`self.img_fc` 和 `self.text_fc` ([model.py](file://file-qsvqjbjmlsvhiym459fo7w%23:~:text=self,freeze_text_encoder/))。这些层将原始CLIP特征变换到统一的特征空间维度（opt.feature_dim，例如1024）。代码通过 `get_img_fc(use_ln=False)` 构造img_fc，如果 `use_ln=True` 则会在线性层后加LayerNorm，这里默认False返回一个线性层`Linear(2048, 1024)` ([model.py](file://file-qsvqjbjmlsvhiym459fo7w%23:~:text=def get_img_fc,feature_dim/))。`get_text_fc` 则返回一个两层MLP：`Linear(1024,1024)+ReLU+Linear(1024,1024)`（若use_ln=True再加LayerNorm） ([model.py](file://file-qsvqjbjmlsvhiym459fo7w%23:~:text=def get_text_fc,12), ) else/))。实际上 text_fc 即使 use_ln=False 也有两层线性，作用可能是增加非线性变换能力。
  4. 调用 `self._freeze_text_encoder()` 冻结CLIP文本编码器的大部分参数 ([model.py](file://file-qsvqjbjmlsvhiym459fo7w%23:~:text=self,freeze_text_encoder/)) ([model.py](file://file-qsvqjbjmlsvhiym459fo7w%23:~:text=def _freeze_text_encoder(self): ,p.requires_grad = false/))。实现上，将 `self.clip.transformer`（Transformer编码器层）和 `self.clip.ln_final`（最终LayerNorm）以及 `self.clip.text_projection`（投影矩阵）这几部分参数 `requires_grad=False` ([model.py](file://file-qsvqjbjmlsvhiym459fo7w%23:~:text=def _freeze_text_encoder(self): ,p.requires_grad = false/))。**未冻结**的部分是词嵌入层 (`token_embedding`)和位置嵌入 (`positional_embedding`)——这意味着训练时可以对文本的embedding进行微调（相当于一种 prompt tuning），但文本编码器主体固定，保持CLIP预训练的语义。这样可防止在小数据集上过度调整语言模型，同时允许模型学习调整输入表示。

- **局部-全局特征融合**：iKUN 提出了**双流**架构，将每帧目标区域和全局区域分开编码再融合。Model初始化中：

  - `self.fusion_local_global = nn.MultiheadAttention(embed_dim=self.img_dim, num_heads=4, dropout=0.)` ([model.py](file://file-qsvqjbjmlsvhiym459fo7w%23:~:text=self,/))用于**局部 vs 全局特征融合**。这里embed_dim=2048，对应图像特征维度，4头注意力。后面将以目标区域特征作为 Query，全局区域特征作为 Key/Value，通过自注意力融合局部信息和全局上下文。
  - 定义**位置嵌入**：`self.pos_emb_local` 和 `self.pos_emb_global` ([model.py](file://file-qsvqjbjmlsvhiym459fo7w%23:~:text=local_reso = 7 ,randn(global_reso/))。它们是可学习参数向量，长度分别为 $7*7=49$（局部特征图大小）和 $21*21=441$（全局特征图大小）。这提供每个空间位置的固定偏置，使模型感知目标内部的位置结构和全局相对位置（因为后续将flatten特征图，注意力本身不区分空间顺序，所以加不同的位置embedding以编码空间信息）。初始化时用随机正态分布乘尺度 $1/\sqrt{49}$ 或 $1/\sqrt{441}$。

- **知识统一模块 (KUM) 设置**：根据配置的 `opt.kum_mode`，Model 初始化会配置不同的融合分支：

  - 如果 `kum_mode == 'cascade attention'`（级联注意） ([model.py](file://file-qsvqjbjmlsvhiym459fo7w%23:~:text=if self,1/))：

    - `self.fusion_visual_textual = nn.MultiheadAttention(embed_dim=self.img_dim, num_heads=4, dropout=0)`：用于**视觉特征 vs 文本特征的跨模态注意力**。embed_dim=2048，与视觉一致。该注意力模块将在融合了局部+全局后的视觉特征上，再以文本特征为 Key/Value 进行注意力，提取与文字相关的视觉信息。
    - `self.fusion_fc = nn.Linear(self.text_dim, self.img_dim)`：一个线性层，将文本特征从1024维映射到2048维，以便与视觉特征对齐用于注意力计算。
    - `self.fusion_ffn = FFN(self.img_dim, 0.1)`：FFN是定义的前馈网络类，内部是 `Linear(d_model, d_model) + Dropout + LayerNorm` 的结构，用于在注意力后做残差提升稳定性 ([model.py](file://file-qsvqjbjmlsvhiym459fo7w%23:~:text=class ffn(nn,layernorm(d_model/))。这里d_model=2048，dropout=0.1。
    - `self.fusion_drop = nn.Dropout(p=0.1)`：在融合过程中可能用于随机掩掉部分特征。

    级联注意模式的含义是：**先融合视觉双流，再用文本特征对融合结果做一次注意力调整**（级联两步）。因此需要上述针对文本的注意力模块和线性变换。

  - 如果 `kum_mode` 是 `'cross correlation'`（互相关）或 `'text-first modulation'`（文本优先调制） ([model.py](file://file-qsvqjbjmlsvhiym459fo7w%23:~:text=self,sequential/))：

    - 定义两个一维卷积：
       `self.fusion_conv1 = nn.Sequential(Conv1d(text_dim, img_dim, kernel_size=1, bias=False), BatchNorm1d(img_dim))`
       `self.fusion_conv2 = nn.Sequential(Conv1d(img_dim, img_dim, kernel_size=1, bias=False), BatchNorm1d(img_dim))`
       这两个卷积用于将文本特征（或融合后的特征）通过1x1卷积映射和调制通道。具体作用见稍后`cross_modal_fusion`实现。
    - 同时定义一个 `Dropout(p=0.1)` 存在于这两模式下融合的过程中。

    互相关和文本优先模式都通过卷积方式融合，不采用注意力机制。区别在于使用文本在融合时的位置不同（前者后融合，后者前融合），具体逻辑后续解释。

  - 如果 `kum_mode` 为 None 或未指定，则模型将不使用额外的文本融合模块，只依赖基础的局部-全局注意，不会初始化上述KUM组件。

在初始化结尾，还有一个 `_init_weights_function` ([model.py](file://file-qsvqjbjmlsvhiym459fo7w%23:~:text=def _init_weights_function,conv1d/))对线性、卷积、LayerNorm等层进行权重初始化（Xavier初始化等）来保证训练稳定。

### 模型前向传播与跨模态融合

Model类的 `forward(self, x, epoch=1e5)` 方法定义了模型前向计算逻辑 ([model.py](file://file-qsvqjbjmlsvhiym459fo7w%23:~:text=def forward,global_img'], textual_hidden, self.opt.kum_mode/)) ([model.py](file://file-qsvqjbjmlsvhiym459fo7w%23:~:text=modulation']: visual_feat = self.visual_local_global( x,global_img']) logits = f.cosine_similarity(visual_feat, textual_feat/))。它会根据是否启用文本指导（kum_mode以及当前epoch对比tg_epoch）选择不同分支。让我们分步解析 forward 过程：

1. **文本编码**：提取输入 `x` 字典中的 `'exp'`（文本）张量，通过 `self.textual_encoding()` 得到 `textual_hidden` 和 `textual_feat` ([model.py](file://file-qsvqjbjmlsvhiym459fo7w%23:~:text=def forward,global_img'], textual_hidden, self.opt.kum_mode/))。这里：

   - `x['exp']` 应为形状 [B, 77] 的文本 token张量（由前文 DataLoader 的 `tokenize(expression)` 产生，B为批大小）。

   - `textual_encoding(self, tokens)` ([model.py](file://file-qsvqjbjmlsvhiym459fo7w%23:~:text=def textual_encoding,1/))内部调用 `self.clip.encode_text_2(tokens, opt.truncation)` ([model.py](file://file-qsvqjbjmlsvhiym459fo7w%23:~:text=def textual_encoding,training/))。`encode_text_2` 是 MyCLIP 类自定义的方法 ([model.py](file://file-qsvqjbjmlsvhiym459fo7w%23:~:text=def encode_text_2,[batch_size, n_ctx, d_model/)) ([model.py](file://file-qsvqjbjmlsvhiym459fo7w%23:~:text=,1)] @ self.text_projection/))：

     ```python
     x = self.token_embedding(text) + self.positional_embedding   # 文本token+位置嵌入
     x = self.transformer(x.permute(1,0,2)).permute(1,0,2)         # 通过Transformer编码
     x = self.ln_final(x)
     hidden = x[:, :truncation] @ self.text_projection            # 取前 truncation 个token的隐层做投影
     x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection  # 取[EoT]位置的特征投影
     return hidden, x
     ```

     简言之：它让 CLIP 对文本编码后，返回两个结果：`hidden` 是**截断的隐藏序列**（每个文本保留前 `truncation` 个 token 的Transformer输出并投影），而 `x` 是CLIP标准输出的文本特征（取序列终止符[EOS]的输出并投影）。在我们的配置里 `truncation=10`，即保留每个句子前10个token的投影向量，组成文本的**局部特征**。

   - textual_encoding 将 `x_hidden, x = clip.encode_text_2(...)`，然后对最终特征 `x` 再通过 `self.text_fc` MLP 投影到统一维度 ([model.py](file://file-qsvqjbjmlsvhiym459fo7w%23:~:text=def textual_encoding,1/))。如果在训练模式，则返回 `(hidden, feat)`；如果在eval模式，则返回 `(hidden, normalize(feat))`（推理时会归一化文本特征以便计算余弦相似度）。

   - 因此 `textual_hidden` 形状 [B, truncation, text_proj_dim]（B×10×1024），`textual_feat` 形状 [B, feature_dim]（B×1024）。`textual_hidden`保留了一定的文本局部信息，`textual_feat`是整句的embedding。

2. **视觉特征提取与融合**：根据是否使用KUM：

   - 如果 **启用了文本指导** (`self.opt.kum_mode` 非空 且 当前 epoch ≥ `opt.tg_epoch`) ([model.py](file://file-qsvqjbjmlsvhiym459fo7w%23:~:text=textual_hidden, textual_feat = self.textual_encoding(x,first modulation/))：

     - **Cascade 模式**：调用

       ```python
       visual_feat = self.visual_local_global(x['local_img'], x['global_img'], textual_hidden, 'cascade attention')
       ```

       传入文本的隐藏序列特征 `textual_hidden` 以及模式。这里cascade模式将利用文本**序列特征**参与融合（因为要做cross-attention）。

     - **Cross-correlation 或 Text-first 模式**：调用

       ```python
       visual_feat = self.visual_local_global(x['local_img'], x['global_img'], textual_feat, self.opt.kum_mode)
       ```

       这两种模式只需要文本的全局embedding（textual_feat），因此传入embedding向量和对应模式字符串。

   - 如果 **未启用文本指导**（kum_mode为空或当前epoch小于阈值） ([model.py](file://file-qsvqjbjmlsvhiym459fo7w%23:~:text=x,global_img']) logits = f.cosine_similarity(visual_feat, textual_feat/))：

     - 调用 `visual_feat = self.visual_local_global(x['local_img'], x['global_img'])` 不传文本特征，即纯视觉融合。模型将在没有文本影响的情况下提取视觉统一特征。

   这样通过不同分支获取了 `visual_feat`：表示经过（可能有文本参与的）视觉特征融合后的**目标特征向量**。

   下面详解 `visual_local_global(local_img, global_img, text_feat=None, kum_mode=None)` 函数 ([model.py](file://file-qsvqjbjmlsvhiym459fo7w%23:~:text=def visual_local_global,c h w/)) ([model.py](file://file-qsvqjbjmlsvhiym459fo7w%23:~:text=,guided/))的实现（这是 KUM 融合的主体）：

   - **输入**：`local_img` shape [B, T, C, H, W] 和 `global_img` shape [B, T, C, H', W']。这里 B是batch，T是每个样本帧数（训练中T=2），C=3通道，H×W如224×224（裁剪图），H'×W'如672×672（全局图）。

   - **特征提取**：先reshape再送入 CLIP 图像编码器：

     ```python
     local_img = rearrange(local_img, 'b t c h w -> (b t) c h w')
     local_feat = self.clip.visual(local_img, with_pooling=False)  # 输出shape: [B*T, C_img, 7, 7]
     global_img = rearrange(global_img, 'B T C H W -> (B T) C H W')
     global_feat = self.clip.visual(global_img, with_pooling=False)  # 输出shape: [B*T, C_img, H1, W1]
     ```

     `clip.visual(..., with_pooling=False)` 返回最后一层卷积特征图，不做默认的平均池化。对于ResNet50，224分辨率输入输出7×7特征图，较大输入（如672）则输出21×21特征图（因为ResNet总stride=32，672/32=21）。代码中虽然CLIP加载时image_resolution=224，但实际上我们输入更大尺寸图片时，ResNet可以产生更大feature map，这也是设置pos_emb_global为441长度的原因。
      经过这步，我们得到：`local_feat` [B*T, 2048, 7,7]，`global_feat` [B*T, 2048, H1,W1]（H1=W1≈21）。

   - **展平特征图并加位置嵌入**：

     ```python
     local_feat = rearrange(local_feat, 'bt c h w -> bt c (h w)')   # 变为 [B*T, 2048, 49]
     global_feat = rearrange(global_feat, 'bt c H W -> bt c (H W)') # [B*T, 2048, 441]
     local_feat = local_feat + self.pos_emb_local  # 广播加，每个位置加对应pos偏置
     global_feat = global_feat + self.pos_emb_global
     local_feat = rearrange(local_feat, 'bt c l -> l bt c')   # 转置为 [49, B*T, 2048]
     global_feat = rearrange(global_feat, 'bt c L -> L bt c') # [441, B*T, 2048]
     ```

     现在 local_feat 和 global_feat 都被展平成二维序列：长度分别49和441，每个位置是2048维的通道特征。我们将它们形状调整为 `[序列长度, 批量, 通道]` 以便送入PyTorch的 MultiheadAttention（要求 [L, N, C] 格式）。

   - **文本引导融合**：根据模式 kum_mode 进行不同的跨模态融合：

     - **模式 `'text-first modulation'`**（文本优先调制） ([model.py](file://file-qsvqjbjmlsvhiym459fo7w%23:~:text=elif mode == 'text,[bt,c,hw/)) ([model.py](file://file-qsvqjbjmlsvhiym459fo7w%23:~:text=vis_feat = rearrange,return out_feat/)) ([model.py](file://file-qsvqjbjmlsvhiym459fo7w%23:~:text=if kum_mode == 'text,fusion_local_global( query=local_feat_2/))：
        该模式在**视觉局部/全局融合之前**，先用文本对每个视觉特征图施加调制：

       ```python
       local_feat_2 = self.cross_modal_fusion(local_feat, text_feat, B, T, 'text-first modulation')
       global_feat_2 = self.cross_modal_fusion(global_feat, text_feat, B, T, 'text-first modulation')
       fusion_feat = self.fusion_local_global(query=local_feat_2, key=global_feat_2, value=global_feat)[0]
       ```

       即：

       1. 对局部特征序列 local_feat 和文本embedding 调用 cross_modal_fusion 函数，mode设为'text-first modulation'，获得调制后的 `local_feat_2`；同样处理 global_feat 得到 `global_feat_2`。稍后详解 cross_modal_fusion 的实现。直观上，这一步**用文本embedding按通道缩放了视觉特征**。
       2. 然后，将调制后的局部特征作为 Query，调制后的全局特征作为 Key，全局原始特征作为 Value，喂入 `fusion_local_global` MultiheadAttention 进行融合 ([model.py](file://file-qsvqjbjmlsvhiym459fo7w%23:~:text=) fusion_feat = self,0] else/))。这里 query和key长度不同（49 vs 441），MultiheadAttention会对每个局部特征位置，在全局特征序列上做注意力聚合，输出 shape与query相同 `[49, B*T, 2048]`。结果保存在 fusion_feat 中。
           这个分支实现了：**文本先影响局部和全局表征，再融合局部-全局信息**，体现“文本优先”。

     - **其它模式** ('cascade attention' 或 'cross correlation' 或无文本)：

       - 如果 kum_mode 不是 'text-first modulation'，则跳过上述调制，直接进行**局部-全局注意力融合** ([model.py](file://file-qsvqjbjmlsvhiym459fo7w%23:~:text=else: %23 cross,0/))：

         ```python
         fusion_feat = self.fusion_local_global(query=local_feat, key=global_feat, value=global_feat)[0]
         ```

         这一步 outputs `fusion_feat` shape `[49, B*T, 2048]`。此时 fusion_feat 表示**融合了全局上下文的局部特征**：每个局部特征位置通过注意力汲取了全局特征的信息（类似Transformer的cross-attention，将环境信息注入目标特征）。

       - 然后添加残差：`fusion_feat = fusion_feat + local_feat` ([model.py](file://file-qsvqjbjmlsvhiym459fo7w%23:~:text=),bt c hw/))。保留原始局部特征，增强模型稳定性和表示完整性。

       - **cascade attention 模式** 或 **cross correlation 模式**：在这之后还需要将文本融合进来 ([model.py](file://file-qsvqjbjmlsvhiym459fo7w%23:~:text=%23 text,training/))：

         ```python
         if kum_mode in ('cascade attention', 'cross correlation'):
             fusion_feat = self.cross_modal_fusion(fusion_feat, text_feat, B, T, kum_mode)
         else:
             fusion_feat = rearrange(fusion_feat, 'L BT C -> BT C L')
         ```

         也就是：

         - Cascade attention：此时 fusion_feat 是 [49, B*T, 2048]，text_feat 对于 cascade 用的是文本序列 hidden（在 forward 中传入的 textual_hidden）。`cross_modal_fusion` 会对 fusion_feat 进行一次**以文本为Key的注意力**，从而**根据文本内容调制融合特征**。
         - Cross correlation：text_feat 用的是文本embedding向量，`cross_modal_fusion` 会将文本视作卷积核与 fusion_feat 做相关运算，实现**跨模态卷积融合**。
         - 这两种模式统一返回 `fusion_feat` 为 shape `[B*T, 2048, L]`（函数内会转换）。而如果没有文本引导（kum_mode=None）或者是text-first模式，则执行else分支简单reshape fusion_feat 为 [B*T, 2048, L] 以便下面池化。

       总结：

       - Cascade模式：**先融合局部全局，再用文本注意力调整**（级联两步）。
       - Cross-corr模式：**先融合局部全局，再用文本卷积调整**。
       - Text-first模式：**先文本调整局部/全局，再融合**。
       - None：**纯视觉融合**。

   - **时空池化**：`fusion_feat` 形状此时统一为 `[B*T, 2048, L]`（对于cascade/cross-corr L=49，text-first因未二次融合也是49，无文本则49）。接下来通过 `self.st_pooling(fusion_feat, bs=B)` ([model.py](file://file-qsvqjbjmlsvhiym459fo7w%23:~:text=fusion_feat = rearrange,1) return fusion_feat/))进行**时空池化**：

     ```python
     # st_pooling: 空间+时间池化，将多个帧的特征汇总为一个向量
     feat = F.adaptive_avg_pool1d(feat, 1).squeeze()        # 先对空间维度L平均池化 -> [B*T, 2048]
     feat = rearrange(feat, '(b t) c -> b c t', b=bs)       # 把时间维度恢复: [B, 2048, T]
     feat = F.adaptive_avg_pool1d(feat, 1).squeeze()        # 再对时间维度T平均池化 -> [B, 2048]
     feat = self.img_fc(feat)                              # 线性投影到 opt.feature_dim (1024)
     ```

     这个函数先对每个样本的每帧**空间特征**取平均（等价于全局平均池化卷积特征图），得到每帧一个2048维向量；然后再对**帧序列**取平均，得到每个轨迹段一个2048维向量；最后通过 `img_fc` 线性层映射到`feature_dim`大小。结果就是 `fusion_feat` 为 `[B, feature_dim]`（训练时未归一化）。

     - 如果 `self.training` 为 True，则直接返回该 `fusion_feat` ([model.py](file://file-qsvqjbjmlsvhiym459fo7w%23:~:text=fusion_feat = self,training: return fusion_feat else/))。如果为 False（测试阶段），则先 `F.normalize(fusion_feat, p=2, dim=-1)` 进行L2归一化后返回 ([model.py](file://file-qsvqjbjmlsvhiym459fo7w%23:~:text=if self,1) return fusion_feat/))。归一化是为了在测试时计算余弦相似度更方便。

   通过 `visual_local_global`，我们获得了**结合文本（视配置而定）的视觉特征** `visual_feat`。尺寸为 [B, feature_dim]，每个批次对应一个轨迹片段的整体表示。

3. **计算余弦相似度**：回到 `forward` ([model.py](file://file-qsvqjbjmlsvhiym459fo7w%23:~:text=else: visual_feat = self.visual_local_global(x,text_feat'] = textual_feat return output/))，有了 `visual_feat` 和之前的 `textual_feat`（对应相应描述短语），模型最后计算：

   ```python
   logits = F.cosine_similarity(visual_feat, textual_feat)  # 按行计算余弦相似度
   output = {'logits': logits, 'vis_feat': visual_feat, 'text_feat': textual_feat}
   ```

   这里 `visual_feat` 和 `textual_feat` 形状均为 [B, 1024]，`F.cosine_similarity` 默认对最后一个维度计算余弦值，返回 shape [B] 的张量`logits`。每个值对应该样本轨迹与描述的相似度分数。这个 `logits` **未经sigmoid**，可以为正或负（取决于夹角）。在训练时，会将其喂入带`with_logits`的二元交叉熵损失，用于判别描述是否匹配（正例希望相似度高正值，负例希望低甚至负值）。

   output除了logits，还返回了 `vis_feat` 和 `text_feat`，训练中不直接用，但在相似度校准等环节可能需要用到特征（例如 similarity_calibration基于text_feat等计算）。

**跨模态融合机制解析 (`cross_modal_fusion`)**：
 在 Model 类中，还有一个辅助函数 `cross_modal_fusion(vis_feat, text_feat, b, t, mode)` 实现了根据不同模式融合文本和视觉特征 ([model.py](file://file-qsvqjbjmlsvhiym459fo7w%23:~:text=def cross_modal_fusion,l bt c/)) ([model.py](file://file-qsvqjbjmlsvhiym459fo7w%23:~:text=fused_feat = self,elif mode == 'cross correlation/))。根据 mode 参数，可分为：

- **Cascade Attention 模式** ([model.py](file://file-qsvqjbjmlsvhiym459fo7w%23:~:text=def cross_modal_fusion,fusion/))：
   期望输入 `vis_feat` 形状 `[L, B*T, C]`（如 [49, B*T, 2048]），`text_feat` 形状 `[B, L_text, C_text]`（文本隐藏序列，L_text≈77，但代码中只用了前truncation=10长度）。实现：

  1. 扩展文本序列：`text_feat = text_feat.unsqueeze(1).repeat(1, t, 1, 1)` ([model.py](file://file-qsvqjbjmlsvhiym459fo7w%23:~:text=,fusion_visual_textual/))将文本序列从 [B, L_text, 1024] 扩展为 [B, T, L_text, 1024]（每个样本复制T次，表示假设每个时间帧都附加相同文本序列），再 `rearrange` 为 `[L_text, B*T, 1024]` ([model.py](file://file-qsvqjbjmlsvhiym459fo7w%23:~:text=text_feat = self,fusion_visual_textual/))方便attention使用。

  2. 文本线性投影：`text_feat = self.fusion_fc(text_feat)` 把文本embedding维度从1024线性映射到2048 ([model.py](file://file-qsvqjbjmlsvhiym459fo7w%23:~:text=text_feat = text_feat.repeat(,fusion_visual_textual( query=vis_feat, key=text_feat, value=text_feat/))。shape变为 [L_text, B*T, 2048]。

  3. 执行多头注意力：

     ```python
     fused_feat = self.fusion_visual_textual(query=vis_feat, key=text_feat, value=text_feat)[0]
     ```

     这里 `vis_feat` 是 [49, B*T, 2048]作为查询序列，`text_feat`作为键和值。注意力将输出一个 shape等于query的张量 `fused_feat` [49, B*T, 2048]，代表**每个视觉位置对文本序列的注意力聚合**。

  4. 融合：`vis_feat = vis_feat * fused_feat` ([model.py](file://file-qsvqjbjmlsvhiym459fo7w%23:~:text=value=text_feat, ),fused_feat/)) 将注意力输出与原视觉特征逐元素相乘（起到**选择性强调**的作用：如果某视觉位置与文本高度相关，fused_feat接近1则保留，否则降低）。

  5. 调整形状：`vis_feat = rearrange(vis_feat, 'l bt c -> bt c l')` 返回 [B*T, 2048, 49] ([model.py](file://file-qsvqjbjmlsvhiym459fo7w%23:~:text=vis_feat = vis_feat ,return vis_feat/))。

  6. 输出：返回经文本调制的视觉特征序列。

  这一模式整体就是**文本->视觉的交叉注意力**，先以文本特征指导每个视觉位置，再乘回视觉特征，达到“按描述信息重新加权视觉特征”的效果。级联注意是在局部-全局融合后做，所以这里 `vis_feat` 输入通常是融合后的特征。

- **Cross Correlation 模式** ([model.py](file://file-qsvqjbjmlsvhiym459fo7w%23:~:text=elif mode == 'cross correlation':,[bt,c,l/)) ([model.py](file://file-qsvqjbjmlsvhiym459fo7w%23:~:text=vis_feat = vis_feat + self,fusion_conv2(vis_feat) return vis_feat/))：
   期望输入 `vis_feat` 形状 `[L, B*T, C]`（如 [49, B*T, 2048]），`text_feat` 形状 `[B, C_text]`（文本embedding向量1024维）。实现：

  1. 文本扩展：`text_feat = text_feat.unsqueeze(1).repeat(1, t, 1)` 得 [B, T, 1024]，然后 `rearrange` 为 `[B*T, 1024, 1]` ([model.py](file://file-qsvqjbjmlsvhiym459fo7w%23:~:text=,[bt,c,l/))。相当于每帧都附加同一文本向量，作为1D卷积核。
  2. 文本卷积1：`text_feat = self.fusion_conv1(text_feat)` ([model.py](file://file-qsvqjbjmlsvhiym459fo7w%23:~:text=text_feat = rearrange,[bt,c,l/))。fusion_conv1是Conv1d(1024->2048)。这里 text_feat形状 (B*T, 1024, 1)，1D卷积核尺寸为1，相当于线性映射，输出 text_feat shape (B*T, 2048, 1)，并BatchNorm。这个输出可视为**为每个样本得到一个2048维的卷积核**。
  3. 视觉特征调整：`vis_feat = rearrange(vis_feat, 'L bt c -> bt c L')` 将vis_feat变为 [B*T, 2048, L] ([model.py](file://file-qsvqjbjmlsvhiym459fo7w%23:~:text=,[bt,c,l/))。例如 [B*T,2048,49]。
  4. 互相关：调用提供的 `xcorr_depthwise(vis_feat, kernel=text_feat)` ([model.py](file://file-qsvqjbjmlsvhiym459fo7w%23:~:text=vis_feat = rearrange,[bt,c,l/))。这个函数实现**深度互相关**：对于每个样本，将 text_feat 视作一个 depthwise卷积核（每个通道单独1x1卷积），对 vis_feat 在长度维度做卷积 ([model.py](file://file-qsvqjbjmlsvhiym459fo7w%23:~:text=def xcorr_depthwise(x, kernel): ,channel/))。由于kernel宽度1，效果相当于**对于每个通道 c，取 vis_feat[c] 与 text_feat[c] 的卷积**，实际上就是 `vis_feat * text_feat`（因为1-element conv就是乘法并求和）。考虑实现，xcorr_depthwise 输出 fused_feat shape [B*T, 2048, L]，基本相当于 channel-wise 的 text_feat 滤波结果。
  5. 融合：`vis_feat = vis_feat + self.fusion_drop(fused_feat)` 将互相关结果加回原特征形成残差 ([model.py](file://file-qsvqjbjmlsvhiym459fo7w%23:~:text=fused_feat = xcorr_depthwise,bt,c,l/))。Dropout随机丢弃一些通道以增强鲁棒性。这样达到**文本高相关的通道得到提升**，低相关的通道作用削弱的效果。
  6. 二次卷积：`vis_feat = self.fusion_conv2(vis_feat)` ([model.py](file://file-qsvqjbjmlsvhiym459fo7w%23:~:text=vis_feat = vis_feat + self,fusion_conv2(vis_feat/))。fusion_conv2是Conv1d(2048->2048)+BN，作用类似channel封装或平滑，进一步让融合后的特征适配。
  7. 输出 vis_feat shape仍为 [B*T, 2048, L]。在调用处会reshape回 [L, B*T, 2048] 或直接后续池化。

  Cross correlation 模式通过**将文本embedding当成滤波器去卷积视觉特征**，其效果类似注意力但形式不同：将文本向量看做2048维的通道权重分布，对视觉特征进行相关运算，相当于突出那些与文本向量方向对齐的视觉成分。因此这种融合更像Siamese网络中的模板匹配。

- **Text-First Modulation 模式** ([model.py](file://file-qsvqjbjmlsvhiym459fo7w%23:~:text=elif mode == 'text,[bt,c,hw/)) ([model.py](file://file-qsvqjbjmlsvhiym459fo7w%23:~:text=vis_feat = rearrange,return out_feat/))：
   期望 `vis_feat` 形状 `[L, B*T, C]`，`text_feat` 形状 `[B, C_text]`。实现：

  1. 文本扩展：与CrossCorr前两步类似，把 text_feat 扩展为 [B*T, 1024, 1]，再通过 `fusion_conv1` 卷积得到 [B*T, 2048, 1] ([model.py](file://file-qsvqjbjmlsvhiym459fo7w%23:~:text=l, _, _ = vis_feat,[bt,c,1/))。这样每样本每帧得到一条2048维“调制向量”。

  2. 不同的是，Text-first随后**将文本调制向量拓展到与视觉特征等长**：

     ```python
     text_feat = text_feat.repeat([1, 1, L])  # [B*T, 2048, L]
     ```

     复制调制向量 L 次，使其在每个空间位置都有相同的值。

  3. 视觉特征变换：`vis_feat = rearrange(vis_feat, 'L bt c -> bt c L')` 得 [B*T, 2048, L]。

  4. 调制融合：`out_feat = vis_feat * self.fusion_drop(text_feat)` ([model.py](file://file-qsvqjbjmlsvhiym459fo7w%23:~:text=,hw bt c/))逐元素相乘，将文本提供的权重作用在每个空间位置的每个通道。因为text_feat对一个frame的所有位置都一样，这相当于**为整幅特征图按通道加权**。也就是说，如果文本提示某种属性，对应视觉特征的一些通道会整体增强或抑制。

  5. 调整输出形状：`out_feat = rearrange(out_feat, 'bt c L -> L bt c')` 回到 [L, B*T, C]。 ([model.py](file://file-qsvqjbjmlsvhiym459fo7w%23:~:text=out_feat = vis_feat ,return out_feat/))

  6. 返回调制后的视觉序列。

  Text-first modulation 实际上将文本embedding直接作为**通道过滤器**（通过一次conv1映射后），在局部和全局特征图上分别执行**整图的通道级调制**。它发生在局部-全局注意力之前，因此称为“文本优先”。调制后的特征再进入局部-全局MHA融合。

以上 `cross_modal_fusion` 的三种策略实现了KUM的三类变体：

- **Cascade Attention**：用注意力机制让文本特征参与，对视觉特征序列进行精细调整（适合建模复杂文本-视觉对齐关系）。
- **Cross Correlation**：以卷积操作突出视觉特征中与文本embedding相似的部分，计算量小巧。
- **Text-First Modulation**：预先对视觉特征按文本进行全局门控，然后再融合局部和全局视觉信息。

### 模型输出与余弦相似度

如上所述，Model.forward 最终返回 `output` 字典，其中 `output['logits']` 是**预测的余弦相似度**。对于训练阶段，一个样本的 logits 是一个标量，表示“该轨迹片段与查询描述匹配的置信度”，不用激活函数就直接用于后续的`BCEwithLogitsLoss`。模型也提供了输出 `output['vis_feat']`（已投影的视觉特征）和 `output['text_feat']`（已投影的文本特征），在测试或校准时可用。

**模型小结**：iKUN 模型利用 CLIP 获取图像和文本的高层语义特征，通过**局部-全局注意力**整合视频帧中的目标局部信息和场景上下文，再通过 **KUM** 将文本线索融合进视觉特征，使模型能针对输入的描述调节视觉注意。最后计算视觉与文本特征的相似度评分，作为该目标符合描述程度的判断依据。通过这种架构，iKUN 在不改变跟踪器的前提下，实现了对每个跟踪目标进行语言查询的判别。

## 损失函数 (`loss.py`)

训练 iKUN 模型，我们将其视为一个**二分类任务**：判断给定的轨迹片段是否符合某条文本描述。`loss.py` 定义了 `SimilarityLoss`，用于度量模型输出的相似度(logits)与真实标签(1或0)之间的误差。这实际上是一个带**Focal Loss**机制的二元交叉熵损失。

- **基本定义**：`SimilarityLoss(nn.Module)` ([loss.py](file://file-rg6c8ztxfpecwfru12ss9m%23:~:text=class similarityloss(nn.module): """ ref: ,def __init/))接受参数 $\rho$、$\gamma$ 和 reduction。$\gamma$ 对应 Focal Loss 的聚焦因子，$\rho$ 用于调整正负样本权重。

- **前向计算**：`forward(self, scores, labels)` ([loss.py](file://file-rg6c8ztxfpecwfru12ss9m%23:~:text=def forward,weights = 1/)) ([loss.py](file://file-rg6c8ztxfpecwfru12ss9m%23:~:text=if self,labels/))
   其中 `scores` 是模型预测的 logit（未经sigmoid），`labels` 是对应的真值（0或1）的浮点张量。可能形状为 [B] 或 [B, N]（但训练设置每样本一个描述，故多为[batch]）。

  步骤：

  1. 计算基础的**二元交叉熵损失**：

     ```python
     loss = F.binary_cross_entropy_with_logits(scores, labels, reduction="none")
     ```

     `binary_cross_entropy_with_logits` 会内部对 `scores` 做 sigmoid，然后按 $-\big[y \log p + (1-y)\log(1-p)\big]$ 计算逐元素损失。因为 `reduction="none"`，保留每个样本的损失值张量，形状与scores相同。

  2. 初始化权重因子 `weights = 1`。

  3. **Focal Loss 调整**（易/难样本调权）：如果设置了 `self.gamma`（默认2），则计算：

     ```python
     logits = scores.sigmoid()         # 转为概率 p
     p_t = logits * labels + (1 - logits) * (1 - labels)
     weights *= ((1 - p_t) ** self.gamma)
     ```

     其中 $p_t$ 表示预测对真实类别的概率：若真实标签 y=1 则 $p_t = p$，否则 $p_t = 1-p$。$(1-p_t)^\gamma$ 就是 Focal Loss 中的因子，当 $p_t$ 很大（模型对真实类别预测准确且自信）时，这个因子很小，降低该样本损失；当模型犯错或不自信时 $p_t$小，因子接近1，不降低损失。因此 $\gamma>0$ 会**减少易分类样本的损失权重**，让模型更关注困难样本。

  4. **正负样本调衡**：如果设置了 `self.rho`（如 `rho<1`），则：

     ```python
     weights *= (self.rho * labels + (1 - labels))
     ```

     也就是对于正样本（label=1）乘以 $\rho$，负样本乘以1。当 $\rho<1$ 时，这会**降低正样本损失的权重**，相对提高负样本的重要性，反之若 $\rho>1`则强调正样本。这样可以平衡数据集正负样本不均衡的问题（如果正样本远少于负样本，设置 $\rho < 1$ 可以防止少数正样本的loss主导训练）。默认`loss_rho=None` 不启用该项。

  5. 将 loss 与 weights 相乘：`loss = loss * weights`，对每个样本应用前述调整因子。

  6. 根据 reduction 返回：如果 `self.reduction == 'mean'` 则取均值，`'sum'` 则求和。代码默认 `'sum'`，也就是**累加batch内所有样本的损失**作为最终loss标量。

- **参考**：类注释提到了参考实现链接，如 torchvision的 focal_loss 和 EQLv2（Equalization Loss v2）。这里 $\rho$ 类似于 EQLv2 中解决长尾问题的思想，对多数类（负例）不过度抑制损失。

**损失函数解读**：
 在 iKUN 训练中，每个训练样本只有一个 label（描述匹配或不匹配），且通常负例（描述不匹配的情况）数量远多于正例。因此使用普通的 BCE损失可能导致模型倾向于输出低相似度（猜负类）而忽略难正例。引入 $\gamma=2$ 的Focal Loss能降低容易判断的负例损失，迫使模型更多关注那些目前还分不清的正例。$\rho` 则可进一步缓解极端不均衡，如果正例太少适当降低它们loss占比，可避免梯度主导。另外loss使用sum意味着和batch size相关，实践中优化器学习率会配合batch大小来设置。

**训练目标**：通过最小化这个损失，模型被引导使对于真实匹配的 (轨迹,描述) 对输出的 `score` 尽可能大（BCE希望sigmoid(score)→1，意味着cosine_similarity→高正值），对于不匹配的对输出score尽可能小（sigmoid→0，即cosine_similarity→负或低）。这正符合我们要求——相似度判别正确与否。

## 训练过程 (`train.py`)

训练脚本 `train.py` 将前述组件组装起来，迭代优化模型参数。下面根据代码逻辑梳理训练流程：

- **初始化阶段**：

  1. **环境设置**：`import opts` 会执行 `opts.py` 中的解析，得到全局 `opt`，并设置GPU环境。接着调用 `set_seed(opt.seed)` ([train.py](file://file-uherntksgvjppb8vsbpmya%23:~:text=from utils import set_seed set_seed(opt/))来固定随机种子，使结果可复现（utils.py定义了set_seed函数，为random、numpy、torch等设置seed）。

  2. **导入依赖**：引入必要库 (torch, optim, torch.cuda.amp 等) 和自定义模块：loss, utils, model, test, dataloader 等 ([train.py](file://file-uherntksgvjppb8vsbpmya%23:~:text=import torch from torch import,tensorboard import summarywriter/))。这些import确保我们可以使用 `opt`, `SimilarityLoss`, `get_model` 等函数/类。

  3. **模型与优化器**：

     - 调用 `model = get_model(opt, 'Model')`。`get_model` 在 model.py 中实现，内部会 `model = Model(opt)` 实例化模型，然后 `.cuda()` 移动到GPU，并用 `nn.DataParallel` 包装以支持多GPU并行。因此得到的 model 已经并行化处理，可以用 model(input) 正常调用。

     - 实例化损失：

       ```python
       sim_loss = SimilarityLoss(rho=opt.loss_rho, gamma=opt.loss_gamma, reduction=opt.loss_reduction)
       ```

       带入配置中设置的loss参数（如 gamma=2等） ([train.py](file://file-uherntksgvjppb8vsbpmya%23:~:text=sim_loss = similarityloss( rho=opt,loss_reduction,/))。

     - 定义优化器：这里使用 **AdamW**（Adam的权重衰减版） ([train.py](file://file-uherntksgvjppb8vsbpmya%23:~:text=optimizer = optim.adamw( [,weight_decay,/))。传入 `model.parameters()`（包含并行wrapper的参数集合），学习率 `opt.base_lr`（1e-5）、权重衰减 `opt.weight_decay`（1e-5）。这会优化模型中**所有可训练参数**（CLIP未冻结的token_embedding、pos_embedding，fusion模块参数，投影层参数等）。由于学习率较小且有余弦调度，AdamW可稳定收敛。

  4. **可选模型恢复**：如果提供了 `opt.resume_path`，则调用 `load_from_ckpt(model, opt.resume_path)` ([train.py](file://file-uherntksgvjppb8vsbpmya%23:~:text=if opt,1 if exists(opt.save_dir): shutil.rmtree(opt.save_dir/))。`load_from_ckpt` 会加载checkpoint字典，恢复 `model.state_dict()` 并返回 (model, epoch)。如果没有提供，设定 `resume_epoch = -1` 并清理可能存在的旧实验目录 ([train.py](file://file-uherntksgvjppb8vsbpmya%23:~:text=else: resume_epoch = ,save_dir/))（如果上次有残留，则删除opt.save_dir以避免冲突）。

  5. **保存配置**：调用 `save_configs(opt)` ([train.py](file://file-uherntksgvjppb8vsbpmya%23:~:text=save_configs(opt) logger = get_logger(opt,save_dir/))将当前配置写入 `config.json` 文件存储在日志目录，方便记录超参数。

  6. **日志与监视**：

     - `logger = get_logger(opt.save_dir)` 获取logger对象，将日志输出保存到`log.txt`文件 ([train.py](file://file-uherntksgvjppb8vsbpmya%23:~:text=save_configs(opt) logger = get_logger(opt,save_dir/))。
     - `writer = SummaryWriter(opt.save_dir)` 初始化TensorBoard记录器，将在训练中记录损失和学习率曲线 ([train.py](file://file-uherntksgvjppb8vsbpmya%23:~:text=logger = get_logger(opt,save_dir/))。

  7. **数据加载**：

     - `dataloader_train = get_dataloader('train', opt, 'RMOT_Dataset', show=True)` ([train.py](file://file-uherntksgvjppb8vsbpmya%23:~:text=dataloader_train = get_dataloader,test', opt, 'rmot_dataset', show=false/))构建训练集DataLoader。show=True会打印数据集信息，如**“Refer-KITTI (train) Number of identities: X”** 等 ([dataloader.py](file://file-wbafkg6fnayniaswptm9mm%23:~:text=def show_information(self): print( f'===> refer,/)), 其中“identities”指目标轨迹数，即训练样本数。dataloader_train 会自动把数据按 opt.train_bs 分批，并在多个worker中并行加载。
     - `dataloader_test = get_dataloader('test', opt, 'RMOT_Dataset', show=False)` ([train.py](file://file-uherntksgvjppb8vsbpmya%23:~:text=dataloader_train = get_dataloader,test', opt, 'rmot_dataset', show=false/))构建测试集DataLoader。**注意**：这里用的是 RMOT_Dataset 的 test 模式，而不是 Track_Dataset。这意味着训练过程中评估使用的是**有真实标签**的测试集（比如KITTI的一部分作为测试）来计算Precision/Recall，而非真正推理跟踪输出。作者可能用这个来监控模型对**标注数据**的判别性能。

  8. 打印开始信息：根据 opt.kum_mode 打印 “Training (Text-Guided ON/OFF)” ([train.py](file://file-uherntksgvjppb8vsbpmya%23:~:text=print( '========== training (text,info('start training/)) 表示是否启用文本指导（kum_mode非None则ON）。`logger.info('Start training!')` 记录日志。

- **训练主循环**：

  ```python
  for epoch in range(resume_epoch + 1, opt.max_epoch):
      model.train()
      ...  # 准备计时和loss统计
      lr = get_lr(opt, epoch)      # 计算当前epoch学习率
      set_lr(optimizer, lr)        # 设置优化器学习率
      ...
      for batch_idx, data in enumerate(dataloader_train):
          ...
  ```

  训练循环从上次中断的epoch+1开始，一直到 max_epoch（默认100）。每个epoch：

  1. 调用 `model.train()` 切换模型到训练模式（启用dropout等，主要针对BatchNorm和Dropout）。

  2. **学习率调度**：使用 `get_lr(opt, epoch)` 计算当前学习率 ([train.py](file://file-uherntksgvjppb8vsbpmya%23:~:text=lr = get_lr,format(epoch, opt.max_epoch), lr=lr/))。`get_lr` 在utils中定义 ([utils.py](file://file-8rluj8uhpg816kgadnzbbx%23:~:text=def get_lr,warmup_epoch/)) ([utils.py](file://file-8rluj8uhpg816kgadnzbbx%23:~:text=return ( opt,0/))：

     - 如果 `epoch < opt.warmup_epoch`，则在起始学习率 opt.warmup_start_lr 和 base_lr 间线性增加。
     - 否则，采用**余弦退火**：
        $lr = opt.cosine_end_lr + \frac{opt.base_lr - opt.cosine_end_lr}{2} \big( \cos(\frac{\pi (epoch - warmup)}{max_epoch - warmup}) + 1 \big)$
        这样 epoch 越后 lr 越接近 `cosine_end_lr`（可为0）。
        本例子 opt.base_lr=1e-5，end_lr=0，max_epoch=100。如果不使用warmup，则lr会在训练中缓慢降低到0。
        `set_lr(optimizer, lr)` 则更新优化器中的学习率。

  3. **统计器**：创建 `AverageMeter('Loss')` 等实例记录每epoch的平均loss和耗时 ([train.py](file://file-uherntksgvjppb8vsbpmya%23:~:text=batch_time = averagemeter('time', ':6,format(epoch, opt.max_epoch/))。ProgressMeter用于美观打印进度。

  4. **批次训练**：遍历 `dataloader_train`：每次得到一个批次的数据 `data` 字典，含键有 `'cropped_images', 'global_images', 'target_expressions', 'target_labels', 'expression_id'` 等。

     - **加载数据到GPU**：

       ```python
       expression = data['target_expressions']            # 文本列表
       expression_ids = data['expression_id'].cuda()      # ID张量（训练用不上）
       inputs = {
           'local_img': data['cropped_images'].cuda(),
           'global_img': data['global_images'].cuda(),
           'exp': tokenize(expression).cuda(),
       }
       targets = data['target_labels'].view(-1).cuda()
       ```

       这里 `data['target_expressions']` 是一个包含 batch_size 个字符串的列表（每个样本一条描述），例如 `["left car in red", "pedestrian in the right", ...]`。通过 `clip.tokenize` 将其转为 token张量，再 .cuda()。同时将图像张量和标签张量都放到GPU。 ([train.py](file://file-uherntksgvjppb8vsbpmya%23:~:text=for batch_idx, data in enumerate,global_images'].cuda(), exp=tokenize(expression).cuda/))
        注意：`data['target_labels']` 原本shape是 [B, 1]（因为train每样本一个标签），view(-1)展开为 [B]，与模型输出logits对齐。

     - **前向传播**：`logits = model(inputs, epoch)['logits']` ([train.py](file://file-uherntksgvjppb8vsbpmya%23:~:text=) targets = data['target_labels'].view(,loss/))
        将拼装的 inputs 字典传入 model（DataParallel会自动分发到多GPU计算）。同时将当前 epoch 传入，以便模型内部决定是否启用文本融合。得到 output 字典，取出 logits 张量，shape [B]。

     - **计算损失**：`loss = sim_loss(logits, targets)` ([train.py](file://file-uherntksgvjppb8vsbpmya%23:~:text=logits = model(inputs, epoch)['logits'] ,backward/))
        将预测分数和真实标签传入前述 SimilarityLoss。因为我们使用 `binary_cross_entropy_with_logits`，这里 logits可以直接输入，无需手动sigmoid。损失函数会返回此 batch 的总loss（标量）。

     - **反向传播与更新**：

       ```python
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()
       ```

       清空梯度，反向传播计算参数梯度，然后AdamW优化一步 ([train.py](file://file-uherntksgvjppb8vsbpmya%23:~:text=loss = sim_loss(logits, targets) ,step/))。此时模型参数已经更新，提高下一次迭代的性能。

     - **记录训练指标**：更新计时和损失统计器 ([train.py](file://file-uherntksgvjppb8vsbpmya%23:~:text=%23 write batch_time.update(time.time() ,item(), iteration/))：

       - `BATCH_TIME.update(time.time() - end)` 记录每批耗时并更新平均。
       - `LOSS.update(loss.item(), opt.train_bs)` 记录loss总和及样本数（从而计算平均loss）。
       - `iteration += 1` 迭代计数（用于tensorboard横轴）。
       - `writer.add_scalar('Train/LR', lr, iteration)` 和 `writer.add_scalar('Loss/', loss.item(), iteration)` 将当前学习图**进行可视化**），当达到指定频率时输出进度到控制台和日志 ([train.py](file://file-uherntksgvjppb8vsbpmya%23:~:text=if (batch_idx + 1) ,item())/))。通过这些手段可以监控训练过程中的学习率和损失变化。

  5. **周期性验证**：每个 epoch 完成后，执行以下操作：

     - 调用 `torch.cuda.empty_cache()` 清理显存 ([train.py](file://file-uherntksgvjppb8vsbpmya%23:~:text=,model, dataloader_test/))（释放内存加速下个阶段）。

     - 如果 `(epoch+1) % opt.eval_frequency == 0`，则进行一次模型评估 ([train.py](file://file-uherntksgvjppb8vsbpmya%23:~:text=if (epoch + 1) ,format(p, r) logger.info(log_info) print(log_info/))：

       ```python
       p, r = test_accuracy(model, dataloader_test)
       log_info = f'precision: {p:.2f}% / recall: {r:.2f}%'
       logger.info(log_info); print(log_info)
       ```

       这里使用前面加载的 `dataloader_test`（RMOT_Dataset test模式）和 `test_accuracy` 函数。`test_accuracy` 会计算模型在真实标注测试集上的 Precision 和 Recall（稍后详细解释该函数），并返回百分比值。日志记录这些指标。**注意**：这是在训练过程中的验证评估，仍然使用真实标签数据，并未对最终跟踪任务输出进行评估，只是衡量模型判别文本正确性的性能。Precision/Recall可以衡量模型对正负样本的准确程度。

     - 如果 `(epoch+1) % opt.save_frequency == 0`，则保存模型检查点 ([train.py](file://file-uherntksgvjppb8vsbpmya%23:~:text=if (epoch + 1) ,epoch}.pth')) torch.cuda.empty_cache/))：
        组装 state_dict 包括当前模型参数`model.state_dict()`、优化器状态、当前epoch，将其保存为 `"epoch{epoch}.pth"` 文件。这样可以在训练中途存储模型，以防中断或用于不同阶段的比较。

     - 再次清理缓存 `torch.cuda.empty_cache()`，防止显存不断累积。

循环结束后（达到max_epoch），打印 `Finish training!` 并结束程序 ([train.py](file://file-uherntksgvjppb8vsbpmya%23:~:text=logger/))。

**训练过程要点**：

- 每个batch模型输出 logits，loss函数使其接近标签，优化器调整参数。经过若干epoch，模型学会将正确描述的相似度拉高，错误描述的相似度压低。
- 通过 `test_accuracy` 监控，可以观察Precision/Recall是否随epoch上升，判断模型性能是否改善。
- 学习率采用预热+余弦退火策略，起初可能保持较低学习率防止不稳定，逐渐提高，又在末期降低细调。
- DataParallel并行执行使多个GPU同时处理一个batch（增加有效batch size，提高吞吐），最终loss是各GPU loss之和，因此在更新统计时用 `opt.train_bs`（总batch大小）来更新AverageMeter。
- 由于embedding层未冻结，模型可以在训练集中稍微调整文本embedding，使更贴合特定描述分布；视觉CLIP全卷积层未冻结过多，也有适应空间。
- 每次取8帧片段、选1个描述的训练策略，使模型见到各种正负组合，增强了判别能力。负样本尤其多，有Focal Loss帮助更有效学习。

至此，iKUN 模型通过训练学到了在embedding空间**区分匹配/不匹配**的能力。接下来，在测试阶段，我们将利用训练好的模型，对未标注的视频进行文字指称目标的识别和输出。

## 测试与评估 (`test.py` 与 `similarity_calibration.py`)

训练完成后，iKUN 模型可以用于推断：给定一段视频和一条文字描述，它可以输出该描述对应的目标轨迹。`test.py` 脚本负责这一流程：加载模型、获取跟踪器输出并通过模型计算相似度分数，然后根据需要进行**相似度校准**，最终生成结果文件以评估跟踪性能。本节分两部分：一是`test.py`的主要功能，包括**准确率评估**和**跟踪输出生成**，二是`similarity_calibration.py`的**相似度校准**原理。

### 测试准确率评估 (test_accuracy)

`test.py` 中定义了两个测试准确率函数：`test_accuracy_v1` 和 `test_accuracy` ([test.py](file://file-uhs5udhj7tue5j9dngugj8%23:~:text=def test_accuracy_v1(model, dataloader, save_img=false): model,1/)) ([test.py](file://file-uhs5udhj7tue5j9dngugj8%23:~:text=def test_accuracy(model, dataloader, save_img=false): model,1/))。它们逻辑类似，只是`test_accuracy`使用了局部+全局双分支模型的输入形式，而`v1`可能对应单分支模型。这里重点说明 `test_accuracy(model, dataloader, save_img=False)`：

- **功能**：在标注数据上评估模型文本判别性能，计算 Precision 和 Recall。这个函数会使用 RMOT_Dataset（模式'test'）加载的样本，其中每个样本包含一个轨迹片段及**一个帧上的多条可能描述**（target_expressions 以逗号分隔，target_labels给出哪些描述存在）。它将模型对这些描述的输出与标签比较统计。

- **实现**： ([test.py](file://file-uhs5udhj7tue5j9dngugj8%23:~:text=def test_accuracy(model, dataloader, save_img=false): model,makedirs(save_dir, exist_ok=true) global_idx = 1/)) ([test.py](file://file-uhs5udhj7tue5j9dngugj8%23:~:text=expressions = data,forward inputs = dict/))

  ```python
  model.eval()
  TP, FP, FN = 0, 0, 0
  assert dataloader.batch_size == 1
  ...
  for batch_idx, data in enumerate(tqdm(dataloader)):
      expressions = data['target_expressions'][0].split(',')
      labels = data['target_labels'][0]        # shape [N_desc]
      # forward
      inputs = {
          'local_img': data['cropped_images'].cuda().repeat_interleave(len(expressions), dim=0),
          'global_img': data['global_images'].cuda().repeat_interleave(len(expressions), dim=0),
          'exp': tokenize(expressions).cuda(),
      }
      logits = model(inputs)['logits'].cpu()
      # evaluate
      TP += ((logits >= 0) * (labels == 1)).sum()
      FP += ((logits >= 0) * (labels == 0)).sum()
      FN += ((logits < 0) * (labels == 1)).sum()
      ...
  PRECISION = TP / (TP + FP) * 100
  RECALL = TP / (TP + FN) * 100
  return PRECISION, RECALL
  ```

  解释：

  - 函数假定 dataloader batch_size==1，每次只取一个轨迹样本的数据 ([test.py](file://file-uhs5udhj7tue5j9dngugj8%23:~:text=tp, fp, fn = 0,,dataloader/)) ([test.py](file://file-uhs5udhj7tue5j9dngugj8%23:~:text=assert dataloader,dataloader/))。这对计算Precision/Recall有好处（逐样本遍历统计）。
  - 对于取出的 `data`：
    - `data['target_expressions'][0]` 是形如 `"exp1,exp2,exp3..."` 的字符串，把它 `split(',')` 拆成列表 `expressions` ([test.py](file://file-uhs5udhj7tue5j9dngugj8%23:~:text=expressions = data,0/))。
    - `labels = data['target_labels'][0]` 是对应的标签张量，形状 [N_desc]（如 [1,0,1,...]）。由于batch=1，直接取第0项即可 ([test.py](file://file-uhs5udhj7tue5j9dngugj8%23:~:text=expressions = data,0/))。
  - **构建模型输入**：为了利用模型一次处理多个描述，我们**复制图像**若干份，将多条描述拼成一个batch：
    - `data['cropped_images']` 原shape [1, T, 3, H, W]，`repeat_interleave(len(expressions), dim=0)` 将batch维度复制N_desc次，得到 [N_desc, T, 3, H, W] ([test.py](file://file-uhs5udhj7tue5j9dngugj8%23:~:text=inputs = dict/))。每个描述对应一份相同的图像片段。
    - 同理复制 `global_img`。然后对 expressions 列表调用 `tokenize`（CLIP tokenize接受list），得到 shape [N_desc, 77] 的token张量。
    - 这样 `inputs` 字典构造好后，调用 `model(inputs)`。因为model DataParallel在eval下，不同desc会分到不同GPU并行计算余弦相似度。输出 `logits` shape [N_desc] 对应每条描述的分数。 ([test.py](file://file-uhs5udhj7tue5j9dngugj8%23:~:text=) logits = model(inputs)/))
  - **计算指标**：二分类阈值取0（因为sigmoid(0)=0.5）：
    - 预测正类条件：`logits >= 0`，预测负类条件：`< 0`。
    - 使用张量比较和布尔乘法：
      - `(logits >= 0) * (labels == 1)` 产生布尔张量，True在位置表示 真正例TP（预测正且实际正）。`.sum()`统计个数，然后累加到TP ([test.py](file://file-uhs5udhj7tue5j9dngugj8%23:~:text=logits = model(inputs)['logits'].cpu() ,(labels == 0)).sum/))。
      - `(logits >= 0) * (labels == 0)` 统计假正例FP（预测为正但实际为负）。
      - `(logits < 0) * (labels == 1)` 统计假负例FN（实际正但预测为负）。
    - 由于labels和比较结果都是向量，上述乘法按位与逻辑相同。最后得到TP,FP,FN总数。
  - 循环所有样本后，计算 Precision = TP/(TP+FP) * 100%，Recall = TP/(TP+FN) * 100% ([test.py](file://file-uhs5udhj7tue5j9dngugj8%23:~:text=precision = tp %2F ,100 return precision, recall/)) ([test.py](file://file-uhs5udhj7tue5j9dngugj8%23:~:text=precision = tp %2F ,return precision, recall/))返回。

  Precision表示模型输出为正的那些预测中有多少是真的（衡量误报警率），Recall表示实际为正的样本有多少被成功检出（衡量漏检率）。二者结合可反映模型在文字判别上的准确性。

  如果 `save_img=True` 参数开启，函数还会把每个预测的图像和结果保存 ([test.py](file://file-uhs5udhj7tue5j9dngugj8%23:~:text=if save_img: save_dir = join(opt,load/)) ([test.py](file://file-uhs5udhj7tue5j9dngugj8%23:~:text=,2f}.jpg'.format/))：它对输入images反归一化（un_norm），然后为每个表达叠加局部和全局图像并保存为一张图片，文件名包含表达内容、标签和预测得分 ([test.py](file://file-uhs5udhj7tue5j9dngugj8%23:~:text=,labels[i], logits[i]/)) ([test.py](file://file-uhs5udhj7tue5j9dngugj8%23:~:text=,cat( (local_img, global_img), dim=0/)) ([test.py](file://file-uhs5udhj7tue5j9dngugj8%23:~:text=imgs = imgs,i] ) save_image/))。这用于可视化模型对每个描述的判断情况。训练完成后可以查看这些图片了解模型倾向，这里不展开。

### 跟踪输出生成 (test_tracking & generate_final_results)

上述 test_accuracy 用于验证模型判别能力，而实际在无标注视频上应用时，需要利用模型输出构建最终的跟踪结果，即每条文本描述对应的目标轨迹输出。`test_tracking` 函数负责利用模型对**跟踪器输出的轨迹**进行打分，`generate_final_results` 则根据打分生成最终结果文件。

- **test_tracking(model, dataloader)** ([test.py](file://file-uhs5udhj7tue5j9dngugj8%23:~:text=def test_tracking,cropped_images'].cuda/)) ([test.py](file://file-uhs5udhj7tue5j9dngugj8%23:~:text=exp=tokenize(data,id/))：
   输入 model（已eval模式）和 dataloader（应为 Track_Dataset 提供的测试轨迹+描述数据）。过程：

  1. 初始化 `OUTPUTS = multi_dim_dict(4, list)` ([test.py](file://file-uhs5udhj7tue5j9dngugj8%23:~:text=outputs = multi_dim_dict,cropped_images'].cuda/))。即一个4层嵌套的默认字典结构，用于存 `OUTPUTS[video][obj_id][frame_id][expression] = [scores...]`。每个表达式对应一个score列表。

  2. 遍历 dataloader：每个 batch 是 Track_Dataset 的若干样本（默认 opt.test_bs=1，逐条处理）。取出 `data`：

     - 构造 inputs：与训练类似，把 data 中 `'expression_new'` 文本 tokenize 得到 tokens（这里每个样本只有一个 expression，无需 repeat图像） ([test.py](file://file-uhs5udhj7tue5j9dngugj8%23:~:text=,expression_new']).cuda(),/))。local_img, global_img 可以批量送入模型。因为Track_Dataset可能batch>1，则 inputs 内 local_img, global_img shape [batch*T, ...]，exp shape [batch, 77]。

     - 前向模型：`similarity = model(inputs)['logits'].cpu()` ([test.py](file://file-uhs5udhj7tue5j9dngugj8%23:~:text=) similarity = model(inputs),video/))。得到每个样本一个logit分数（tensor shape [batch]）。取到CPU上便于后处理。

     - 将每个样本结果记入 OUTPUTS：

       ```python
       for idx in range(len(data['video'])):
           for frame_id in range(data['start_frame'][idx], data['stop_frame'][idx] + 1):
               frame_dict = OUTPUTS[data['video'][idx]][int(data['obj_id'][idx])][int(frame_id)]
               frame_dict[data['expression_raw'][idx]].append(similarity[idx].item())
       ```

       逻辑：对于batch中第 idx 个样本，

       - 取出对应的 video编号、obj_id、起止帧。对于这个轨迹段的每个帧 frame_id（start到stop，包括stop） ([test.py](file://file-uhs5udhj7tue5j9dngugj8%23:~:text=for idx in range(len(data,id x].cpu().numpy().tolist/))：
         - 定位嵌套字典 OUTPUTS[video][obj_id][frame_id]，得到该帧对应一个defaultdict of list（针对表达式分类的字典）。
         - 在这个字典里，取键为当前样本的 expression_raw 字符串，append 当前模型输出的 similarity 分值 ([test.py](file://file-uhs5udhj7tue5j9dngugj8%23:~:text=frame_dict = outputs/))。
            也就是说，对每条样本（某目标某段 + 某描述），模型输出的置信度被登记到该目标对应帧下该描述的列表中。如果一个帧多次出现在不同片段（滑窗重叠）中，会累积多个分数。

  3. 循环结束返回 OUTPUTS 结构，包含每帧每描述的分数列表。

  OUTPUTS 可以被序列化为 JSON 保存（test.py主程序中确实dump了这个结果）。

- **generate_final_results(cls_dict, data_dir, track_dir, save_dir, thr_score=0.0)** ([test.py](file://file-uhs5udhj7tue5j9dngugj8%23:~:text=def generate_final_results,rmtree(save_dir/)) ([test.py](file://file-uhs5udhj7tue5j9dngugj8%23:~:text=,shape) == 2/)) ([test.py](file://file-uhs5udhj7tue5j9dngugj8%23:~:text=video_dict = cls_dict,video_dir_out, exp/)) ([test.py](file://file-uhs5udhj7tue5j9dngugj8%23:~:text=with open(join(exp_dir_out, 'predict,n/))：
   这是将模型评分转为实际跟踪输出文件的函数：

  1. 准备目录：`template_dir = join(data_dir, 'gt_template')` 指向数据集提供的 ground truth 模板目录，每个视频下有各描述短语子文件夹和对应真值`gt.txt`。`save_dir` 若存在则删除重建 ([test.py](file://file-uhs5udhj7tue5j9dngugj8%23:~:text=template_dir = join,video/))。

  2. 遍历每个视频文件夹：

     ```python
     for video in os.listdir(template_dir):
         if video not in cls_dict: continue
         video_dir_in = join(template_dir, video)
         video_dir_out = join(save_dir, video)
         ...
     ```

     只处理存在于cls_dict（即模型输出里有结果）的视频，创建输出文件夹结构。

  3. **准备输出文件结构**：对于模板中每个描述 `exp` 文件夹 ([test.py](file://file-uhs5udhj7tue5j9dngugj8%23:~:text=for exp in os,load tracks/))：

     - 创建对应输出目录 `exp_dir_out`。
     - 将模板中的 `gt.txt` 通过 `os.symlink` 符号链接复制到输出文件夹（这样评估时可以读取真值轨迹进行对比） ([test.py](file://file-uhs5udhj7tue5j9dngugj8%23:~:text=gt_path_in = join(exp_dir_in, 'gt,txt'), delimiter/))。
        这部分相当于复制 ground truth 结构，以便Evaluator能够找到`gt.txt`比较。链接方式避免实际复制。

  4. **加载跟踪器输出**：尝试读取 `tracks = np.loadtxt(track_dir/video/all/gt.txt')` ([test.py](file://file-uhs5udhj7tue5j9dngugj8%23:~:text=try: tracks = np,1/))。如果失败则像 Track_Dataset 那样读 car 和 pedestrian 的predict.txt合并 ([test.py](file://file-uhs5udhj7tue5j9dngugj8%23:~:text=tracks_1 = np,loadtxt(join(track_dir, video, 'pedestrian/))。总之得到 `tracks`，含视频所有目标的所有帧检测记录（格式 frame, id, x, y, w, h, ...）。

  5. **生成 predict.txt**：

     - 取出模型输出的该视频字典 `video_dict = cls_dict[video]` ([test.py](file://file-uhs5udhj7tue5j9dngugj8%23:~:text=video_dict = cls_dict,video_dir_out, exp/))。然后双重循环：

       ```python
       for obj_id, obj_dict in video_dict.items():
           for frame_id, frame_dict in obj_dict.items():
               for exp in EXPRESSIONS[video]:
                   if exp in EXPRESSIONS['dropped']: continue
                   if exp not in frame_dict: continue
                   exp_dir_out = join(video_dir_out, exp)
                   score = np.mean(frame_dict[exp])
                   with open(join(exp_dir_out, 'predict.txt'), 'a') as f:
                       if score > thr_score:
                           bbox = tracks[(tracks[:,0]==frame_id) & (tracks[:,1]==obj_id)][0]
                           ... 
                           f.write(','.join(map(str, bbox)) + '\n')
       ```

       逐个目标、逐帧检查：

       - 内层对每个 `exp` 表达式（从该视频的全表达式列表EXPRESSIONS[video]中遍历）：
         - 跳过 `EXPRESSIONS['dropped']` 列出的无意义短语 ([test.py](file://xn--file-uhs5udhj7tue5j9dngugj8%23:~:text=for frame_id, frame_dict in obj_dict,todo:-ci23mto3c/))。
         - 如果该帧的 frame_dict 没有此 exp 键，则跳过 ([test.py](file://xn--file-uhs5udhj7tue5j9dngugj8%23:~:text=if exp in expressions,todo: continue-o492lg41c/))（有TODO注释表示此情况理论上不应发生，因为frame_dict默认defaultdict，所有exp键都存在即使没有score应该也是空list。不过这里保守地略过空的情况）。
         - 计算score = 平均分数 ([test.py](file://file-uhs5udhj7tue5j9dngugj8%23:~:text=continue exp_dir_out = join,txt'), 'a') as f/))：取模型输出的 frame_dict[exp] 列表的均值。如果这个帧是从多个窗口覆盖得到的分数，会平均，得到更稳健的评分。
         - 打开对应exp的输出文件（append模式）`predict.txt`。如果 score > thr_score（阈值，默认为0.0，即只要score为正就认为匹配） ([test.py](file://file-uhs5udhj7tue5j9dngugj8%23:~:text=score = np.mean(frame_dict,0/))：
           - 从之前载入的 tracks 找到当前 video 中 frame_id 和 obj_id 对应的那一行检测数据 `bbox` ([test.py](file://file-uhs5udhj7tue5j9dngugj8%23:~:text=if score ,10,/))。tracks数组通过布尔索引筛选frame和id，取第一个匹配行。assert检查确保bbox长度9或10。
           - 判断 `MIN_FRAME < bbox[0] < MAX_FRAME` ([test.py](file://file-uhs5udhj7tue5j9dngugj8%23:~:text=assert bbox,n/))：这里 FRAMES[video] 可能给出视频起止帧号，用来排除序列的第一帧和最后一帧（因为 ground truth eval 通常不算开始和结束瞬间）。如果通过，则将bbox整行数据（frame,id,x,y,w,h,...）拼成字符串写入 predict.txt ([test.py](file://file-uhs5udhj7tue5j9dngugj8%23:~:text=if min_frame ,n/))。

       这样，对于满足条件的 (frame,obj_id) 若属于描述 exp，且模型信心score超阈，则输出该目标在该帧的检测结果到 exp 对应的 predict.txt 中。最终每个描述文件汇总了该描述出现的所有帧目标。

  6. 重复上述，对每个视频、每个描述都生成predict.txt完成后，输出目录`save_dir`下将组织成：

     ```
     save_dir/
       ├── video1/
       │    ├── exp1/
       │    │    ├── gt.txt (链接到模板)
       │    │    └── predict.txt (模型输出)
       │    ├── exp2/
       │    │    ├── gt.txt
       │    │    └── predict.txt
       │    └── ...
       └── video2/...
     ```

     这与输入模板结构一致，但 predict.txt 内容由模型决定。之后可以对这些predict与gt计算多目标跟踪指标（如MOTP/MOTA或更专门的语言指称指标），通常有评估脚本根据这些文件比较。

- **test.py 主程序**：在文件末尾 `if __name__ == '__main__':` ([test.py](file://file-uhs5udhj7tue5j9dngugj8%23:~:text=if __name__ == '__main__': print,opt.save_postfix}.json/)) ([test.py](file://file-uhs5udhj7tue5j9dngugj8%23:~:text=if not exists,test', opt, 'track_dataset/)) ([test.py](file://file-uhs5udhj7tue5j9dngugj8%23:~:text=save_dir = join(opt.save_root, opt.exp_name, f'results,load(open(output_path/)) ([test.py](file://file-uhs5udhj7tue5j9dngugj8%23:~:text=generate_final_results( cls_dict=cls_dict, data_dir=opt,/))将上述组件串联：

  1. 打印开始信息 (Text-Guided ON/OFF) 类似训练。
  2. 设置输出结果JSON路径 `output_path = save_root/exp_name/results{postfix}.json` ([test.py](file://file-uhs5udhj7tue5j9dngugj8%23:~:text=) output_path = join(opt,opt.save_postfix}.json/))。
  3. 如果该文件不存在：
     - 初始化模型 `model = get_model(opt, 'Model')` 并尝试 `model, _ = load_from_ckpt(model, opt.test_ckpt)` ([test.py](file://file-uhs5udhj7tue5j9dngugj8%23:~:text=if not exists,test', opt, 'track_dataset/))，加载指定的测试模型权重（默认 iKUN.pth）。若失败则提示未加载。
     - 准备 dataloader：`dataloader = get_dataloader('test', opt, 'Track_Dataset')` ([test.py](file://file-uhs5udhj7tue5j9dngugj8%23:~:text=print('the model is not loaded,model, dataloader/))，即用Track_Dataset产生需要评分的所有 (轨迹,描述) 组合。
     - 调用 `output = test_tracking(model, dataloader)` 得到嵌套结果字典，随后 `json.dump(output, open(output_path,'w'))` 保存为json文件 ([test.py](file://file-uhs5udhj7tue5j9dngugj8%23:~:text=output = test_tracking(model, dataloader) os,dump( output, open(output_path, 'w')/))。这样第一次运行会保存模型对每个帧每描述的score列表，避免重复计算。
  4. 读取结果：`CLS_DICT = json.load(open(output_path))` ([test.py](file://file-uhs5udhj7tue5j9dngugj8%23:~:text=save_dir = join(opt.save_root, opt.exp_name, f'results,load(open(output_path/))，将刚才算好的json装载为字典。
  5. **相似度校准**（若启用）：如果 `opt.similarity_calibration` 为 True ([test.py](file://file-uhs5udhj7tue5j9dngugj8%23:~:text=if opt,text_feat_dict, cls_dict, a=8/))：
     - 读取 `TEXT_FEAT_DICT = json.load(open(save_root/'textual_features.json'))`。这个json应预先由 similarity_calibration.encode_text 生成，包含训练集和测试集每条描述的embedding和概率信息。
     - 调用 `CLS_DICT = similarity_calibration(TEXT_FEAT_DICT, CLS_DICT, a=8, b=-0.1, tau=100)`。将文本频率信息和输出结果送入校准函数，得到**校准后的 CLS_DICT**。参数 a, b, tau 是校准超参（后文详述）。
  6. 最后调用 `generate_final_results(cls_dict=CLS_DICT, data_dir=opt.data_root, track_dir=opt.track_root, save_dir=SAVE_DIR)` ([test.py](file://file-uhs5udhj7tue5j9dngugj8%23:~:text=generate_final_results( cls_dict=cls_dict, data_dir=opt,/))，将cls_dict字典转为最终结果文件。SAVE_DIR定义为 `save_root/exp_name/results{postfix}`。这样就完成了跟踪结果的生成。

### 相似度校准 (`similarity_calibration.py`)

在开放环境中，测试时可能出现训练集中没出现过的全新描述短语，模型往往缺乏信心，且这些描述可能属于**长尾分布**。类似地，有些描述在训练中很常见，模型会倾向给出较高分。**相似度校准**的目的是利用训练集统计的信息，校正模型在测试时的相似度评分，以减少因描述频率差异造成的不公平。具体地，iKUN 提出了利用**伪频率**的方法：通过计算测试描述与训练描述的语义相似度，加权已知训练描述的出现概率，来推断测试描述的“先验概率”，然后据此对模型score做线性调整。

`similarity_calibration.py` 包含两个函数：`encode_text()` 用于离线生成文字特征和频率词典；`similarity_calibration(TEXT_FEAT_DICT, CLS_DICT, a, b, tau)` 用于调整得分。

- **encode_text()** ([similarity_calibration.py](file://file-dpttrqpkh3jdkxgrfmrexn%23:~:text=def encode_text(): exp_dir = join(opt,'feature': none/)) ([similarity_calibration.py](file://file-dpttrqpkh3jdkxgrfmrexn%23:~:text=num_bbox = 0 for video,of positive bboxes for each/)) ([similarity_calibration.py](file://file-dpttrqpkh3jdkxgrfmrexn%23:~:text=bbox_num = sum,feature'] = feat/)) ([similarity_calibration.py](file://file-dpttrqpkh3jdkxgrfmrexn%23:~:text=for exp in text_feat_dict,text_feat_dict[mode][exp]['bbox_num'] %2F num_bbox/))： 这不是在test.py运行时调用的函数，而是需要预先执行一次，生成 `textual_features.json` 文件。在README或附加说明中，应有指示调用它来生成校准所需数据。

  它所做的：

  1. 加载 CLIP 模型：`clip, _ = load('ViT-B/32')` ([similarity_calibration.py](file://file-dpttrqpkh3jdkxgrfmrexn%23:~:text=text_feat_dict = dict,b%2F32') clip.cuda() clip.eval/))。这里用了 CLIP 自带的`clip.load()`函数加载 ViT-B/32 模型并切换 eval 模式。与模型训练不同，这里用视觉Transformer版CLIP来获取文本embedding，或只是用其文本部分（因为embedding应该通用，选ViT还是RN50影响不大）。
  2. 初始化 `TEXT_FEAT_DICT = {'train': defaultdict(...), 'test': defaultdict(...)}` ([similarity_calibration.py](file://file-dpttrqpkh3jdkxgrfmrexn%23:~:text=for mode in ,/))。对每个描述，将存储三个信息：`feature`（文本特征向量），`bbox_num`（训练集中正样本框计数），`probability`（出现频率概率）。
  3. 遍历模式 in ('train','test') ([similarity_calibration.py](file://file-dpttrqpkh3jdkxgrfmrexn%23:~:text=num_bbox = 0 for video,of positive bboxes for each/))：
     - 对于每个视频 in VIDEOS[mode]，遍历 `opt.data_root/expression/<video>/` 目录下每个描述文件 `.json` ([similarity_calibration.py](file://file-dpttrqpkh3jdkxgrfmrexn%23:~:text=video_dir = join,of positive bboxes for each/))：
       - `exp = exp_file[:-5]`取出文件名作为描述字符串，`exp = expression_conversion(exp)` 规范化 ([similarity_calibration.py](file://file-dpttrqpkh3jdkxgrfmrexn%23:~:text=video_dir = join,of positive bboxes for each/))。
       - `exp_gt = json.load(open(file))` 读入该描述对应的GT数据（假设包含一个字典，其 'label' 键对应各帧标注，比如 frame->list of bboxIDs）。
       - 如果 mode=='train'：计算该描述的总bbox数量 `bbox_num` ([similarity_calibration.py](file://file-dpttrqpkh3jdkxgrfmrexn%23:~:text=,bbox_num num_bbox += bbox_num/))：对 exp_gt['label'] 的每个frame的列表取长度并求和（即此描述在训练集总共标注了多少个目标实例）。将这个数量累加到 TEXT_FEAT_DICT['train'][exp]['bbox_num']。另外 `NUM_BBOX` 统计训练集全体 bbox计数。
       - **文本特征**：如果当前 `TEXT_FEAT_DICT[mode][exp]['feature'] is None`（避免重复算），则：
         - `text = tokenize(exp).cuda()` 得到单条描述的token张量 ([similarity_calibration.py](file://file-dpttrqpkh3jdkxgrfmrexn%23:~:text=num_bbox += bbox_num if text_feat_dict,feature'] = feat/))。
         - `feat = clip.encode_text(text)` 用 CLIP 模型编码获取文本embedding ([similarity_calibration.py](file://file-dpttrqpkh3jdkxgrfmrexn%23:~:text=text = tokenize(exp),feature'] = feat/))。对于ViT-B/32，输出512维，但clip.encode_text返回未经normalize的embedding（猜测512维）。
         - `feat = F.normalize(feat, p=2)` 做L2正则化 ([similarity_calibration.py](file://file-dpttrqpkh3jdkxgrfmrexn%23:~:text=text = tokenize(exp),feature'] = feat/))（确保特征在比较相似度时用余弦距离）。
         - `feat = feat.detach().cpu().tolist()[0]` 将tensor转为Python列表存储 ([similarity_calibration.py](file://file-dpttrqpkh3jdkxgrfmrexn%23:~:text=text = tokenize(exp),feature'] = feat/))。
         - 保存到 TEXT_FEAT_DICT[mode][exp]['feature'] = feat。
     - 如果 mode=='train'：在处理完一个视频后，退出内层循环，遍历已经收集的 train exp，将每个exp的 `'probability'` 计算为 `bbox_num / NUM_BBOX` ([similarity_calibration.py](file://file-dpttrqpkh3jdkxgrfmrexn%23:~:text=for exp in text_feat_dict,text_feat_dict[mode][exp]['bbox_num'] %2F num_bbox/))（出现次数除以总次数，即频率概率）。
  4. 最后将 TEXT_FEAT_DICT dump 到 `text_feat_bboxNum.json` ([similarity_calibration.py](file://file-dpttrqpkh3jdkxgrfmrexn%23:~:text=text_feat_dict/))。根据 test.py，这个文件应该被重命名或复制为 `textual_features.json` 供加载。

  经过 encode_text，我们得到了：

  - **训练集**：每种描述短语的CLIP文本特征和其在训练集出现频率（probability）。
  - **测试集**：每种描述的CLIP文本特征（没有probability，因为测试没有GT计数，这部分用于和训练特征比相似度）。

- **similarity_calibration(TEXT_FEAT_DICT, CLS_DICT, a, b, tau)** ([similarity_calibration.py](file://file-dpttrqpkh3jdkxgrfmrexn%23:~:text=def similarity_calibration,x + b/)) ([similarity_calibration.py](file://file-dpttrqpkh3jdkxgrfmrexn%23:~:text=for frame, frame_value in obj_value,probs).sum/)) ([similarity_calibration.py](file://file-dpttrqpkh3jdkxgrfmrexn%23:~:text=new_exp_value = ,exp] = new_exp_value/))： 输入：

  - TEXT_FEAT_DICT：上面生成的结构（含 train和test的features和train概率）。
  - CLS_DICT：模型原始输出结果（嵌套字典，未校准的score列表）。
  - 标量参数：a, b, tau。论文中 likely 设 a=8, b=-0.1, tau=100，表示最终校准加权公式的参数。

  过程：

  1. 拷贝原结果：`cls_dict = deepcopy(CLS_DICT)` ([similarity_calibration.py](file://file-dpttrqpkh3jdkxgrfmrexn%23:~:text=cls_dict = deepcopy,probability'] for x in/)) 防止修改原数据。
  2. 将训练集所有描述的 feature 列表 转为 numpy数组 `FEATS` shape [N_train_exp, d]，将对应probability列表转为 `PROBS` 数组 ([similarity_calibration.py](file://file-dpttrqpkh3jdkxgrfmrexn%23:~:text=feats = np.array(,train'].values/))。
  3. 遍历 cls_dict: `for video, video_value in cls_dict.items(): ...` ([similarity_calibration.py](file://file-dpttrqpkh3jdkxgrfmrexn%23:~:text=for video, video_value in cls_dict,sim.min/))，深入每个video，每个obj_id，每个frame，每个exp：
     - 对于每个具体的描述字符串 exp（如"red car in left"这种） ([similarity_calibration.py](file://file-dpttrqpkh3jdkxgrfmrexn%23:~:text=for frame, frame_value in obj_value,probs).sum/))：
       - 规范化它：`exp_new = expression_conversion(exp)`，以确保和TEXT_FEAT_DICT索引一致。
       - 取出该描述的测试特征：`feat = np.array(TEXT_FEAT_DICT['test'][exp_new]['feature'])[None, :]` shape [1, d] ([similarity_calibration.py](file://file-dpttrqpkh3jdkxgrfmrexn%23:~:text=exp_new = expression_conversion,none,/))。
       - 计算与所有训练描述特征的相似度向量：`sim = (feat @ FEATS.T)[0]` ([similarity_calibration.py](file://file-dpttrqpkh3jdkxgrfmrexn%23:~:text=feat = np.array(text_feat_dict,0/))。这里feat是1xd, FEATS是 Nxd，乘积得到1xN，再取[0]为长度N的数组 sim。实质上 sim[i] 是该测试描述embedding与第i个训练描述embedding的内积（若embedding已归一，则为**余弦相似度**乘d，或者由于norm=1其实就是cosine相似度*d，再归一化会减小d影响）。
       - **归一化 sim**：`sim = (sim - sim.min()) / (sim.max() - sim.min())` ([similarity_calibration.py](file://file-dpttrqpkh3jdkxgrfmrexn%23:~:text=sim = (feat @ feats.t),sim.min/))。把sim中的值线性缩放到0~~1区间（因为embedding相似度本身范围-1~~1，归一后成为相对相似度权重)。
       - 计算 softmax 权重：`weight = np.exp(tau * sim) / np.exp(tau * sim).sum()` ([similarity_calibration.py](file://file-dpttrqpkh3jdkxgrfmrexn%23:~:text=sim = (sim ,probs).sum/))。$\tau$ 是放大系数，=100 会使最大的相似度对应权重非常接近1，其他几乎0（即接近 one-hot 选取最相似的训练描述）；较小 $\tau$ 则更平均。这样得到一个**权重分布** weight (长度N_train)，归一化和softmax确保 weight_i >=0 且 sum=1。
       - 计算加权概率：`prob = (weight * PROBS).sum()` ([similarity_calibration.py](file://file-dpttrqpkh3jdkxgrfmrexn%23:~:text=weight = np.exp(tau ,sum() new_exp_value =/))。即将训练集中各描述的频率 PROBS 按上述权重求和。这个 `prob` 可视为**根据训练集中相似描述推断出的当前描述的先验概率**。比如，如果测试描述跟几个常见的训练描述都类似，那prob会较大；如果跟只有罕见描述相似，那prob会较小；若是完全新颖的描述（embedding不接近任何已知描述），sim都很低均匀，softmax会平滑但结果仍可能较低。
       - 校准分数：`new_exp_value = [ x + fn(prob) for x in exp_value ]` ([similarity_calibration.py](file://file-dpttrqpkh3jdkxgrfmrexn%23:~:text=prob = (weight ,for x in exp_value/))。这里 exp_value 是原cls_dict里的score列表。`fn = lambda x: a * x + b` 定义了线性函数 ([similarity_calibration.py](file://file-dpttrqpkh3jdkxgrfmrexn%23:~:text=def similarity_calibration,x + b/))。给每个原始score加上 `a*prob + b` 的偏移量。a=8,b=-0.1，意味着当prob接近0（该描述很罕见）时，加偏移约 -0.1（略微降低score）；当prob较大（描述常见）时，加正偏移，比如prob=0.2则加1.5。这相当于**提升模型对高频描述的置信度，削弱对低频描述的置信度**，从而补偿模型自身可能的偏差。
       - 将 frame_value[exp] 更新为 new_exp_value 列表 ([similarity_calibration.py](file://file-dpttrqpkh3jdkxgrfmrexn%23:~:text=] frame_value/))。
  4. 返回校准后的 cls_dict。

举例：假设测试描述 "blue truck" 在训练集中基本没出现，但embedding和 "blue car"（常见）有些相似度0.5，那么 tau=100 softmax会给 "blue car"很高权重，其概率假设0.10，则 prob≈0.10，fn(prob)=8*0.10-0.1=0.7。这样模型对 "blue truck" 的score每个增加0.7，避免全被判低。另外如果描述非常生僻embedding也不像别的，prob会很低甚至0，加偏移-0.1，稍微降低score。当score阈值设0时，这可能使一些原本勉强为正的rare描述变负，从而**减少误报**。反之，常见描述embedding接近，比如 "red car"，train prob高，如0.05（5%），fn=8*0.05-0.1=0.3，增加0.3分，利于**召回**常见目标。

需要注意的是这些参数是经验设定，校准属于后处理，可根据验证集调整 a, b, tau 达到Precision/Recall的权衡。

### 总结

测试阶段iKUN的使用流程如下：

1. **准备数据**：先用一个现成多目标跟踪器（如NeuralSORT）跑视频，得到每帧每目标的检测轨迹（predict.txt）。
2. **模型打分**：Track_Dataset 枚举每个目标每段轨迹与每条描述组合，模型计算相似度。
3. **（可选）校准**：参考训练集频率，对模型打分做线性调整。
4. **生成结果**：对每个描述短语，汇总得分高于阈值的目标轨迹帧，输出predict.txt文件与gt.txt对比评估。

通过这种方式，iKUN 实现了**“说出描述，模型即可标出对应轨迹”**的功能，而无需重新训练跟踪器。核心在于利用视觉-语言共同空间和融合机制，将问题转化为相似度判别，从而与传统检测跟踪解耦。值得注意的是：

- iKUN 假定有基本的跟踪器提供候选轨迹，如果跟踪器漏检目标或轨迹错误，iKUN无法纠正，只能在提供的轨迹中筛选。
- 神经卡尔曼滤波（NKF）在代码中未直接出现，可能隐藏在NeuralSORT或不在开源范围内。不过iKUN本身不改变跟踪轨迹，只在score判别上发挥作用。
- CLIP 模型对跨模态对齐起关键作用，若想改进模型，可考虑使用更新的多模态模型（如CLIP-ViT大模型）或在CLIP基础上微调embedding以更好适应特定领域描述。
- KUM 模块提供了多种融合策略，使用者可以尝试调整 `opt.kum_mode` 选择不同模式，或者修改 `cross_modal_fusion` 来实现新的融合方法（如更多层的交叉注意力、不同卷积核大小等），以测试对性能的影响。
- 数据加载部分高度依赖预先准备的 JSON 和跟踪结果格式，如果迁移到其他数据集，需要适配解析逻辑。例如Refer-Dance数据集格式和KITTI有所不同，需要重新编写 RMOT_Dataset 的 parse_data。
- 相似度校准提供了一个有趣的思路：结合**先验知识**（训练数据分布）来优化模型决策。这个思想也可扩展到其它场景，比如可以利用语言模型对描述罕见程度的判断来调整置信度等等。

最后，整个项目的代码结构清晰、模块解耦明显：数据、模型、损失、训练、测试各司其职。通过本解析，相信读者已经理解了 iKUN 项目的工作原理和代码实现细节。从配置参数到模型架构、从训练策略到测试生成，都可以在此基础上尝试修改：比如调整采样长度、更换文本编码方式、或引入新的评价指标等。祝您在深入研究 iKUN 代码后，能够运用这些知识进行扩展和改进，实现更强大的跨模态多目标跟踪模型。