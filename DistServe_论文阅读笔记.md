# DistServe 论文阅读笔记

> **论文**：DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving    
> **阅读方式**：沿用 vLLM 笔记的“问题 → 概念 → 方法 → 实验 → 判断”结构；原图旁边配通俗解释。

**颜色说明**：<span style="color:#1d4ed8"><strong>蓝色＝原文解释</strong></span>；<span style="color:#15803d"><strong>绿色＝我的理解</strong></span>；<span style="color:#b91c1c"><strong>红色＝核心结论</strong></span>；<span style="color:#b45309"><strong>橙色＝边界与易错点</strong></span>。

“原文解释”是对作者论述的中文转述，不冒充逐字引文。“我的理解”中的比喻、教学数字和研究设想都不属于论文实验。以上颜色用于笔记文字，实验曲线按各图图例识别。部分 Markdown 软件会忽略颜色，文字标签仍可帮助区分；同目录 HTML 版保留彩色排版。

## 1. 论文信息与核心结论

| 项目 | 内容 |
| --- | --- |
| 题目 | DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving |
| 中文理解 | 把提示词处理与逐 token 生成分开，以提高满足时延要求的服务能力 |
| 作者 | Yinmin Zhong、Shengyu Liu、Junda Chen、Jianbo Hu、Yibo Zhu、Xuanzhe Liu、Xin Jin、Hao Zhang |
| 会议 | OSDI 2024 |
| 研究阶段 | 训练完成后的在线推理服务 |
| 主要优化对象 | Prefill/Decode 干扰、两阶段资源与并行策略耦合 |
| 核心机制 | P/D 分离、分别配置资源和并行度、根据带宽约束部署 |
| 主要目标 | 最大化每 GPU goodput，即在 SLO 达标率要求下可承载的请求率/GPU |
| 原文位置 | 摘要、§1，PDF 第 1–2 页 |

<span style="color:#b91c1c"><strong>一句话理解：先把“读题”和“接着写答案”放到不同 GPU 上，再分别配置资源，并把中间 KV 状态的搬运安排好。</strong></span>

论文的主要因果关系是：

**P/D 分离 → 减少相互干扰、解除资源耦合 → 分别优化首 token 和后续生成速度 → 用较少 GPU 满足给定服务要求。**

<span style="color:#b45309"><strong>不是把模型的前半层叫 Prefill、后半层叫 Decode。</strong></span>同一请求的两个阶段都要执行完整模型所需的各层；两个角色各自持有一套模型权重，必要时各自跨多个 GPU 切分。

## 2. 作者究竟在解决什么问题？

### 2.1 同一批次里，“读题”和“继续写”会互相拖慢

<span style="color:#1d4ed8"><strong>【原文解释】</strong></span>

传统混合部署把新请求的 Prefill 和已有请求的 Decode 放在相同 GPU 上。长 Prefill 执行时间较长，容易拖慢同批或等待中的 Decode；反过来，混入 Decode 工作也可能增加 Prefill 的耗时。优先照顾一个阶段，又可能损害另一个阶段的时延。

原文 Fig. 2 直接观察这种相互干扰：

![原文 Fig. 2：混合 Prefill 后，批次执行时间和干扰增大](assets/distserve/fig02_interference.png)

*原图：作者公开会议版，Fig. 2；对应用户版 PDF 第 4 页。原图仅作裁剪，数据和坐标未改。图源见文末 A。*

**怎么读这张图：**

- 横轴是批次大小，纵轴是批次执行耗时。
- 橙色表示只运行 Decode，蓝色表示加入一个 Prefill 工作后的情况。
- 两个子图的输入长度分别是 128 和 1024；长输入使干扰更明显。
- 这测的是执行时间，不是包含所有排队时间的端到端请求延迟。

<span style="color:#15803d"><strong>【我的理解】</strong></span>

把它想成同一位服务人员同时接两类任务：有人需要先读一大段材料，有人只等着继续接收几个字。为了提高总处理量，把任务凑在一起有好处，但已经开始“边写边返回”的用户，可能因为插入长任务而感到卡顿。

这里的关键不是某个阶段“没有用”，而是它们对资源和响应节奏的要求不同。

### 2.2 不只是干扰，连资源配置也被绑在一起

| 两个阶段的差别 | 混合部署的困难 |
| --- | --- |
| Prefill 一次处理很多已知输入 token | 可能需要更多计算并行，尽快产生首 token |
| Decode 每轮为每条活跃序列增加一个 token | 需要批处理来摊薄权重读取成本，又要控制每轮耗时 |
| 应用对 TTFT、TPOT 的要求不同 | 一套共同配置可能为了较严格的约束而过度分配资源 |

<span style="color:#15803d"><strong>【我的理解】</strong></span>

一个业务希望“马上开始答，后面慢一点也能接受”，另一个业务允许“先等一下，但开始后要很快生成完”。两种业务的最佳 P/D 资源比例，未必相同。

原文位置：§2.1–§2.3，PDF 第 3–4 页。

## 3. 理解方法前，需要掌握的概念

### 3.1 Prefill、Decode 与三个时延指标

| 概念 | 通俗解释 | 本文的关注点 |
| --- | --- | --- |
| Prefill | 处理提示词，建立其 KV cache，并生成首个输出 token | 尽快开始响应 |
| Decode | 利用历史 KV 继续逐个生成后续 token | 持续输出的速度 |
| TTFT | 从请求进入系统到拿到首个输出 token 的时间 | 包含排队与首 token 生成过程 |
| TPOT | 首 token 之后，平均每个后续输出 token 的耗时 | 衡量一个请求后续生成的平均速度 |
| 端到端延迟 | 从请求进入到全部输出完成 | 还受总输出长度影响 |

对输出 token 数 n 大于 1 的请求，可用时间戳理解：

$$
\mathrm{TPOT}
=
\frac{t_{\mathrm{last}}-t_{\mathrm{first}}}{n-1},
\qquad
T_{\mathrm{E2E}}
=
\mathrm{TTFT}+(n-1)\mathrm{TPOT}.
$$

这是帮助理解指标的表达。TPOT 是平均值，不能单独证明生成过程里从未出现一次很长的停顿。

### 3.2 SLO、达标率与 goodput

<span style="color:#1d4ed8"><strong>【原文解释】</strong></span>

SLO 是服务目标，例如 TTFT 不超过某个阈值、TPOT 不超过另一个阈值。SLO attainment 是满足这些要求的请求占比。论文主要关心达标率至少 90%，附录另测 99%。

将“两项同时达标”写出来：

$$
a_i
=
\mathbf{1}
\left[
\mathrm{TTFT}_i\leq \tau_P
\ \land\
\mathrm{TPOT}_i\leq \tau_D
\right],
\qquad
A=\frac{1}{N}\sum_{i=1}^{N}a_i.
$$

论文的每 GPU goodput 可以理解为：

$$
G_{\mathrm{GPU}}
=
\frac{\lambda^\star}{N_{\mathrm{GPU}}},
\qquad
\lambda^\star
=
\max\{\lambda:A(\lambda)\geq \alpha\}.
$$

其中，α 是要求的达标率，分母要统计参与服务的 P、D 及副本所使用的 GPU。

<span style="color:#b91c1c"><strong>重点：本文比较的不是“最多吐出多少 token”，而是“服务质量达标时，每张 GPU 能承载多少请求”。</strong></span>

<span style="color:#15803d"><strong>【我的理解】</strong></span>

假设系统每秒能接很多单，但大部分都严重超时，对在线用户仍可能不好用。因此，吞吐量高与 goodput 高不能直接画等号。

### 3.3 一个实例不一定是一张 GPU

原文把持有完整一套模型权重的资源单元称为 instance。一个实例内部可以用多张 GPU：

| 方式 | 做法 | 主要权衡 |
| --- | --- | --- |
| 算子内并行 Intra-op，通常对应 TP | 把同一个大算子拆到多个 GPU 上共同计算 | 可能缩短执行时间，但增加通信，收益会递减 |
| 算子间并行 Inter-op，通常对应 PP | 将不同层划成流水线阶段 | 可提高并发服务能力，但受阶段通信与流水线气泡影响 |
| 复制 Replication | 部署更多完整服务实例 | 分摊请求，但多占模型权重空间 |

<span style="color:#b45309"><strong>“TP 更多就一定更快”“PP 更多就一定提高吞吐量”都太绝对。</strong></span>端到端延迟还包括排队；不同输入长度、到达率和网络条件，会改变最佳选择。

原文位置：§1–§3，PDF 第 1–6 页。

## 4. 原有做法为什么不够？

| 做法 | 解决什么 | 原文指出的不足 |
| --- | --- | --- |
| Continuous batching | 将不同时间到达的请求持续加入批次，提高资源利用率 | 混合 P/D 仍会互相干扰 |
| 优先 Prefill | 尝试压低 TTFT | Decode 可能等待，损害 TPOT |
| 优先 Decode | 保护持续输出 | 新请求可能迟迟拿不到首 token |
| Chunked prefill | 把长 Prefill 分块，减少一次占用时间 | 减轻干扰，但仍有调度及重复访存代价 |
| 只调共同的并行配置 | 改善整体执行效率 | 两阶段仍被绑在同一套资源安排里 |

<span style="color:#15803d"><strong>【我的理解】</strong></span>

这些方案主要是在“同一个工作场所里如何排顺序”。DistServe 改变的是工作场所的组织方式：把两类任务分开，并给它们分别配资源。

<span style="color:#b45309"><strong>我的判断：不能因此说 chunked prefill 已被全面淘汰。</strong></span>作者在 §7 明确承认，对于不敏感于时延、主要追求离线吞吐量的场景，chunked prefill 等方法可能更合适；GPU 很少时，分离架构的选择空间也有限。

原文位置：§2.3、§7，PDF 第 3–4、12–13 页。

## 5. 核心方法：P/D 分离后，分别配置资源

### 5.1 系统架构

![原文 Fig. 6：Prefill 与 Decode 实例分离，通过 KV cache 传输衔接](assets/distserve/fig06_architecture.png)

*原图 Fig. 6；对应用户版 PDF 第 8 页。*

<span style="color:#1d4ed8"><strong>【原文解释】</strong></span>

控制器把新请求分派给 Prefill 实例。Prefill 完成后，将中间状态，主要是 KV cache，以及首个输出 token 交给 Decode 实例。Decode 实例继续生成后续 token。

| 请求的步骤 | 在哪里执行 | 需要保存或传递什么 |
| --- | --- | --- |
| 处理 prompt | Prefill 实例 | 为提示词计算 KV cache |
| 返回首 token | Prefill 阶段产生 | 首个采样结果 |
| 阶段切换 | P 实例与 D 实例之间 | KV cache 等状态 |
| 继续生成 | Decode 实例 | 读取已有 KV，持续追加新 KV |

<span style="color:#15803d"><strong>【我的理解】</strong></span>

P 实例负责“把题目读懂并建立上下文状态”，D 实例接着用这些状态继续写。交接的主要内容是大体积 KV，不是每个请求都把整套模型权重搬过去；权重已经在两种实例中分别部署好了。

### 5.2 为什么两个阶段的批处理策略不同？

![原文 Fig. 3：两阶段吞吐量对批次大小的反应不同](assets/distserve/fig03_batching.png)

*原图 Fig. 3；对应用户版 PDF 第 5 页。模型为 13B，曲线区分不同输入长度。*

<span style="color:#1d4ed8"><strong>【原文解释】</strong></span>

较长 prompt 的 Prefill 可能单个请求就接近算力饱和，再增加批次主要会延长等待和执行时间。Decode 的单条序列每轮只处理一个新 token，小批次下利用率较低，扩大批次通常更重要。

作者引入需要 profiling 得到的阈值 Lm：让 Prefill 尽量批到接近能充分利用 GPU 的 token 数；超过该阈值的长请求可单独处理。

<span style="color:#15803d"><strong>【我的理解】</strong></span>

Prefill 常常是一份题目就够忙了；Decode 则适合同时为许多用户各写下一个字，分摊一次读取模型权重的成本。

<span style="color:#b45309"><strong>这描述的是论文所测条件下的典型趋势。</strong></span>不能固定地把所有 Prefill 都叫计算受限、所有 Decode 都叫带宽受限。Decode 批次足够大时，也可能逐渐接近计算受限。

### 5.3 为什么不能预先规定 Prefill 一律用 TP、Decode 一律用 PP？

![原文 Fig. 4：不同到达率和 TP 加速系数会改变并行方案的优劣](assets/distserve/fig04_parallelism.png)

*原图 Fig. 4；对应用户版 PDF 第 5 页。以 66B 模型、两张 A100 为分析实例。*

作者用均匀输入长度、泊松到达等简化条件建立 M/D/1 排队模型。单设备情况下：

$$
\overline{\mathrm{TTFT}}
=
D+\frac{RD^2}{2(1-RD)},
\qquad RD<1.
$$

D 是固定执行时间，R 是到达率。第一项是执行，第二项是排队。

<span style="color:#15803d"><strong>【我的理解】</strong></span>

请求很少时，主要看“单次算得快不快”；请求很多时，还要看“能不能减少排队、保持流水线忙起来”。这就是同一个并行方案在不同负载下可能出现优劣变化的原因。

该排队模型用于解释机制。真实输入长度和到达过程更复杂，最终配置仍由带有 profiling 参数的模拟器搜索，不能把简化公式直接当作所有线上请求的精确预测。

原文位置：§3.1–§3.2，PDF 第 4–6 页。

## 6. 最关键的设计：配置搜索与带宽感知部署

### 6.1 算法到底输入什么、输出什么？

| 输入 | 输出 |
| --- | --- |
| 模型、GPU 容量与数量 | P、D 各自使用的并行配置 |
| 请求长度分布、到达过程与目标到达率 | 需要多少个 P、D 实例或实例组合 |
| TTFT/TPOT 阈值与达标率 | 这些实例或阶段片段如何放到集群里 |
| 节点内与节点间的通信条件 | 满足网络限制的部署方案 |

<span style="color:#1d4ed8"><strong>【原文解释】</strong></span>

DistServe 从历史请求拟合工作负载分布，采样轨迹给模拟器。模拟器结合 profiling 拟合的执行时延，估计一个配置在不同请求率下的 SLO 达标率，再通过二分搜索找出可承载的最大请求率。

因此，它既不是随机选配置，也不是逐一把所有配置部署到真实集群里完整试跑。

### 6.2 两个部署算法

![原文 Algorithm 1 与 Algorithm 2：根据跨节点通信条件选择搜索方式](assets/distserve/algorithms_placement.png)

*算法原图；对应用户版 PDF 第 7 页。先理解下面的步骤，再回看伪代码。*

| 方面 | Algorithm 1：High Node-Affinity | Algorithm 2：Low Node-Affinity |
| --- | --- | --- |
| 对应情形 | 跨节点网络很快，可承担 P→D KV 传输 | 跨节点带宽有限 |
| 核心思路 | P、D 分别搜索高性价比并行配置，再复制以匹配流量 | 将对应模型层的 P/D 阶段片段放在同节点，联合搜索 |
| KV 传输主要约束 | 可跨节点安排 | 尽量走节点内高速互联 |
| 配置自由度 | 较大 | 增加同节点与 GPU 数限制 |

这里的 High/Low 使用原文命名。理解时重点看对应的带宽条件，避免仅凭“节点亲和性”这个词猜测部署规则。

**Algorithm 1，通俗步骤：**

1. 枚举满足显存和设备数量约束的 TP/PP 配置。
2. 用模拟器估计每个配置的 Prefill goodput/GPU，保留合适方案。
3. 对 Decode 做相同的评估。
4. 根据总目标流量，计算需要复制多少个 P 和 D 实例。

**Algorithm 2，通俗步骤：**

1. 尝试不同的阶段划分。
2. 枚举一个节点中 P 片段与 D 片段的配置组合。
3. 只保留设备数量等约束允许、对应层的 KV 能在节点内交接的组合。
4. 用模拟器比较整个组合的 goodput/GPU。
5. 复制选择出的组合，以匹配目标流量。

<span style="color:#b91c1c"><strong>低带宽方案不是“把整个大模型的所有 P 和 D GPU 都塞到一台机器”。关键是让对应层的状态在同节点交接。</strong></span>

两个阶段可以使用不同 GPU 数和内部并行配置；“对应层放在同节点”不等于所有 TP/PP 参数必须一样。

### 6.3 真实选择的配置，帮助理解“分别优化”

原文附录 Table 3 的部分结果：

| 模型 / 负载 | P 的 TP × PP | 单个 P 实例 GPU 数 | D 的 TP × PP | 单个 D 实例 GPU 数 |
| --- | --- | --- | --- | --- |
| OPT-13B / ShareGPT | 2 × 1 | 2 | 1 × 1 | 1 |
| OPT-66B / ShareGPT | 4 × 1 | 4 | 2 × 2 | 4 |
| OPT-175B / ShareGPT | 3 × 3 | 9 | 4 × 3 | 12 |

这里统计的是对应单个实例的 GPU 数，不能忽略副本数量后直接当成整个部署的总资源。

<span style="color:#15803d"><strong>【我的理解】</strong></span>

“Prefill 分配几张，Decode 分配几张”不是先拍一个固定比例，而是由模型、负载、SLO 和互联条件共同决定。比如 66B 的例子中，两个阶段恰好都用 4 张，但组织方式不同。

### 6.4 为什么网络会影响这个方案能不能用？

原文给出：OPT-66B 的一个 512-token 请求，KV 约 1.13 GB。如果每秒 10 个请求需要交接，就要搬约 11.3 GB/s，约等于 90 Gbit/s。

<span style="color:#b45309"><strong>GB/s 与 Gbit/s 相差约 8 倍，不能混写。</strong></span>以上是原文量级估算，链路有效吞吐还会受到协议、并发和硬件路径影响。

<span style="color:#15803d"><strong>【我的理解】</strong></span>

拆开计算能减少干扰，却会产生交接流量。因此部署方案必须把“少等多少时间”与“多花多少搬运成本”一起算。这也是本文比单 GPU 显存管理更直接关联网络方向的地方。

原文位置：§3.3、§4.1–§4.2、附录 A–B，PDF 第 6–8、17–18 页。

## 7. 系统运行：不是算完配置就万事大吉

### 7.1 请求如何分派？

<span style="color:#1d4ed8"><strong>【原文解释】</strong></span>

集中式控制器先把请求送到队列较短的 P 实例，再送到负载较低的 D 实例。总体采用 FCFS，并配合针对两阶段的批处理策略。

### 7.2 为什么 KV 要“拉”，而不是一直“推”？

突发流量可能让多个 P 实例同时产出大量 KV，直接推给 D 会形成显存压力。DistServe 让 D 按需拉取，把 P 的显存暂时用作已经完成 Prefill 的状态缓冲区。

<span style="color:#15803d"><strong>【我的理解】</strong></span>

像下游工位有空位了再接货，而不是上游一做完就全塞过去。这样可以缓冲突发，但缓冲仍会占用 P 的显存，并不是免费、无限的队列。

### 7.3 工作负载改变后怎么办？

系统监测到输入/输出平均长度、到达率等发生明显变化时，可以重新执行部署搜索，再重新加载权重。原文描述的是较慢时间尺度的重规划，不是每个 token 到来时都重新部署模型。

### 7.4 作者没有完成什么？

原文明确将高级抢占和容错留作后续工作。FCFS 仍可能出现长任务挡住短任务的 convoy effect；P/D 依赖也使故障影响不再完全局限于一个独立副本。

<span style="color:#b45309"><strong>减少 P/D 干扰，不等于消除所有排队、长任务阻塞或系统故障。</strong></span>

原文位置：§4.3–§5，PDF 第 8–9 页。

## 8. 实验怎么做？先明确条件再看数字

### 8.1 硬件与对比系统

| 项目 | 原文条件 |
| --- | --- |
| 实机集群 | 4 个节点，共 32 张 A100 80GB |
| 每节点 | 8 张 GPU，节点内用 NVLink 互联 |
| 跨节点带宽 | 25 Gbit/s |
| 模型 | OPT-13B、OPT-66B、OPT-175B，FP16 |
| 主实验部署算法 | 主要使用低跨节点带宽对应的 Algorithm 2 |
| 基线 | vLLM、DeepSpeed-MII |
| 额外消融基线 | 搜索并行配置的 vLLM++，以及放宽部署限制的 DistServe-High |

原文所用 vLLM 只支持相应的 TP 设置；不能把这个历史条件当成当前所有 vLLM 版本的能力限制。DeepSpeed-MII 未参与 OPT-175B 比较，作者说明存在内核形状与显存限制。

### 8.2 业务、数据与 SLO

| 应用 | 模型 | 数据集 | TTFT 阈值 | TPOT 阈值 |
| --- | --- | --- | --- | --- |
| Chatbot | OPT-13B | ShareGPT | 0.25 s | 0.10 s |
| Chatbot | OPT-66B | ShareGPT | 2.5 s | 0.15 s |
| Chatbot | OPT-175B | ShareGPT | 4.0 s | 0.20 s |
| Code completion | OPT-66B | HumanEval | 0.125 s | 0.20 s |
| Summarization | OPT-66B | LongBench | 15 s | 0.15 s |

这些是作者根据业务目标选择的实验阈值，不是所有业务共同遵循的行业标准。

数据集提供文本和长度信息，但没有真实到达时间戳，因此作者使用泊松过程构造请求到达。LongBench 的输入被裁剪到 OPT 所支持的上下文范围，不能把这个实验当成百万 token 上下文的实测验证。

### 8.3 主图中的坐标怎么理解？

| 图中信息 | 解释 |
| --- | --- |
| Per-GPU Rate | 总请求率按 GPU 数归一后比较 |
| SLO Attainment | 同时满足 TTFT、TPOT 要求的请求占比 |
| SLO Scale | 按同一比例缩放两项时延阈值，越小越严格 |
| 90% 水平目标 | 找到维持至少该达标率的最大请求率，或最严格阈值 |

<span style="color:#b91c1c"><strong>提高达标比例（90%→99%）与收紧时延阈值，是两个不同的实验维度。</strong></span>

原文位置：§6.1、Table 1，PDF 第 9–10 页。

## 9. 主要实验结果：证明了什么，又没证明什么？

### 9.1 Chatbot：分离后可以承载更高的请求率

![原文 Fig. 8：ShareGPT 上不同模型规模的达标率与请求率、SLO 严格程度](assets/distserve/fig08_chatbot.png)

*原图 Fig. 8；对应用户版 PDF 第 10 页。上排改变每 GPU 请求率，下排改变 SLO Scale。*

<span style="color:#1d4ed8"><strong>【原文解释】</strong></span>

达到至少 90% SLO attainment 时，DistServe 相比 vLLM 可承载约 2.0–4.6 倍请求率；相比 DeepSpeed-MII，适用配置下为约 1.6–7.4 倍。

<span style="color:#15803d"><strong>【我的理解】</strong></span>

看上排时，关注谁能在更靠右的位置仍保持在达标率目标以上；看下排时，关注谁能在更严格的时延阈值下仍达标。

<span style="color:#b45309"><strong>不是每个请求都快 7.4 倍；7.4 倍是某些配置相对特定基线的最大容量收益。</strong></span>

### 9.2 不同业务：收益由哪个阶段的 SLO 更紧决定

![原文 Fig. 9：代码补全与摘要的实验](assets/distserve/fig09_applications.png)

*原图 Fig. 9；对应用户版 PDF 第 11 页。模型均为 OPT-66B，代码补全用 HumanEval，摘要用 LongBench。*

| 业务 | 相比 vLLM 可承载请求率 | 相比 vLLM 可支持的更严格 SLO |
| --- | --- | --- |
| Code completion | 5.7 倍 | 1.4 倍 |
| Summarization | 4.3 倍 | 12.6 倍 |

这些结果来自不同条件下的请求率扫描或 SLO 扫描，不能把一行的两个倍数理解为同一运行点必然同时获得。

代码补全的首 token 要求很紧；摘要允许更长 TTFT，却希望后续生成足够快。P/D 分离使资源配置可以分别服务这些要求。

### 9.3 传输开销确实小，但要连同部署条件一起讲

![原文 Fig. 10：总生命周期占比与 KV 传输时间分布](assets/distserve/fig10_communication.png)

*原图 Fig. 10；对应用户版 PDF 第 11 页。左侧使用 OPT-175B/ShareGPT，右侧包含三个 OPT 规模。*

作者报告，在所测部署下，KV 传输占总时延不到 0.1%，超过 95% 的请求传输延迟小于 30 ms。

<span style="color:#b45309"><strong>关键条件：对应层的 P/D 状态被安排在同节点，经节点内 NVLink 交接。不能将此图解释为“任意 25 Gbit/s 跨节点网络搬 KV 都几乎没有代价”。</strong></span>

<span style="color:#15803d"><strong>【我的理解】</strong></span>

这更像是“设计把最重的交接避开了慢链路”，而不是“KV 本来就很小”。此外，传输占端到端总耗时的比例，与它占 TTFT 或一次输出间隔的比例，也不是同一回事。

### 9.4 消融：收益不只是“给 DistServe 多试了几种配置”

![原文 Fig. 11：分离与部署限制的消融](assets/distserve/fig11_ablation.png)

*原图 Fig. 11；对应用户版 PDF 第 12 页。OPT-66B、ShareGPT；本组使用模拟器。*

- vLLM++ 也枚举并行配置，在该实验中默认配置已经较优。
- DistServe-Low 仍有收益，支持减少 P/D 干扰的作用。
- 放宽跨节点部署限制的 DistServe-High 进一步提高性能，说明互联条件限制了可选配置空间。

<span style="color:#b45309"><strong>这组是模拟结果，不是已经在真实高带宽集群上完成的同等实测。</strong></span>

### 9.5 附录补了什么？

| 附录 | 用途 |
| --- | --- |
| A，PDF 第 17 页 | 给出用于拟合 Prefill/Decode 执行时间的模型，并用 profiling 确定系数 |
| B，PDF 第 18 页 | 列出主实验选出的 TP/PP 配置 |
| C，PDF 第 18 页 | 将达标率目标改为 99%，查看更严格可靠性目标下的表现 |

原文位置：§6.2–§6.5、附录 A–C。

## 10. 原文图表索引：回看时应该看哪里？

| 图表 | 用户版 PDF 页 | 要回答的问题 |
| --- | --- | --- |
| Fig. 1 | 1 | 混合部署为何同时卡住 TTFT 和 TPOT？ |
| Fig. 2 | 4 | 混入 Prefill 会让执行时间增加多少？ |
| Fig. 3 | 5 | P、D 为什么需要不同批处理策略？ |
| Fig. 4 | 5 | 并行策略为何随到达率和通信效率改变？ |
| Fig. 5 | 6 | 大 Decode 批次下，TP/PP 有何权衡？ |
| Algorithm 1–2 | 7 | 高/低跨节点带宽下如何搜索部署？ |
| Fig. 6 | 8 | 控制器、两类实例与 KV 传输如何衔接？ |
| Table 1、Fig. 7 | 9 | 业务 SLO、数据集及长度分布是什么？ |
| Fig. 8 | 10 | 聊天业务在不同模型下收益如何？ |
| Fig. 9 | 11 | 代码补全、摘要有何差异？ |
| Fig. 10 | 11 | 传输是否成为主要开销？ |
| Table 2、Fig. 11 | 12 | 模拟器准确性及方法消融如何？ |
| Fig. 12 | 12 | 搜索部署方案需要多长时间？ |
| Table 3、Fig. 13–14 | 18 | 具体配置、99% 达标率目标下的结果如何？ |

## 11. 对论文的评价与我的理解

### 11.1 我认为最值得学的地方

<span style="color:#15803d"><strong>【个人判断】</strong></span>

1. **从用户体验定义系统目标。**首 token 和持续输出速度分开衡量，比只报总 token/s 更能反映在线服务要求。
2. **结构性消除一类干扰。**它不是仅在混合批次中调优顺序，而是改变部署结构。
3. **把网络约束纳入系统方案。**算法需要知道哪些状态必须搬运、哪里搬运更划算。
4. **同时考虑计算与排队。**单次执行更快，并不自动等于负载升高时端到端延迟最小。

### 11.2 需要保留的局限

| 局限 | 原文依据或我的判断 |
| --- | --- |
| P、D 各自需要模型权重副本 | 分离会增加权重存储和部署组织成本；显存很紧时不一定合算 |
| 依赖工作负载统计 | 历史分布不代表未来完全可预测；突发与快速漂移需要额外评估 |
| 网络结论有部署前提 | 通信小的主实验不能推广成任意网络上的免费交接 |
| 主实测上下文范围有限 | LongBench 做过截断；超长上下文讨论主要是分析，不是同等实测 |
| 主指标是请求平均 TPOT | 不能据此保证每个相邻输出间隔都平稳 |
| 抢占与故障处理未深入实现 | 原文 §4.3 明确列为后续方向 |
| 不存在所有场景统一最优架构 | 原文 §7 讨论离线、少 GPU 场景可能适合其他方案 |






