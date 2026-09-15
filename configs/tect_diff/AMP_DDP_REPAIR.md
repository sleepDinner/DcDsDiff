# 主训练 AMP 缓存与 DDP 梯度修复

## 异常及已授权处置

2026-09-15 前期监督发现 C (`TECT-DIFF-FULL-R512-S42-REFNORM-V2-PERF-20260915-C`) 在相同 optimizer step 上，两 rank 的梯度范数多次明显不同。例如 step1400 为 0.77301049 / 0.10962672。C 已通过身份核验的 controller stop 停止，并保持 operational hold；控制器、子进程退出且资源释放。C 的参考 epoch0–19、固定最终 probe 和2048图训练集校准均已完成并通过；主训练尚无完整 epoch、checkpoint 或 All8 结果。

用户已授权确认异常后停止并进行有界版本化修复。本次继承身份为 `TECT-DIFF-FULL-R512-S42-REFNORM-V2-AMPFIX-20260915-D`，配置仍为 `full_r512_s42_refnorm_v2.json`，hash `0b0358aa4908b852bb53113467184483fbb11d4be517eb58548ba80757052f88`。C 部分主训练无效，保留其日志和冻结源码，不恢复或用于结果。A/B/C均保持停止。

## 根因和修复边界

服务器 PyTorch2.1.2+cu121 在一个外层 autocast 内，先于 no_grad 下运行同一可训练网络，会缓存没有梯度连接的低精度权重。随后的正式前向复用这些缓存，部分参数不再得到梯度。自条件分支随机出现，使本应得到梯度的856个参数张量在部分 microbatch 中降为263个；原有按机制分组检查只要求每组存在部分非零梯度，未发现完整性缺失。

该动态缺失与 PyTorch 旧版 `no_sync` / `find_unused_parameters=True` 的累积同步问题组合，产生两 rank 参数分歧。官方实现背景见 [AMP no_grad cache issue](https://github.com/pytorch/pytorch/issues/160438) 和 [DDP accumulation issue](https://github.com/pytorch/pytorch/issues/69031)。根因结论以本机实际环境的CPU复现及512/BF16/双rank完整模型对照为依据。

修复仅在预测自条件的 no_grad 区域禁用 autocast 权重缓存，结束后恢复缓存设置。保留外层精度、RNG、前向数值、停梯度的预测反馈和全部科学配置。诊断中的同类预测路径一起修复。没有升级环境、改变数据、batch/microbatch、累积、损失、学习率、轮数、采样或选择规则。

增加三项保护：主模型实际 probe 要求完整梯度覆盖；每次主更新前核验梯度覆盖、两rank梯度张量数和范数一致；保存主 checkpoint 前核验两rank可训练参数SHA256一致。继承网络中8个从未调用的MMFF空间注意力权重仍保持None梯度语义；BN运行统计按既定协议保持rank局部，参数一致性检查不比较这些buffer。任何保护失败均阻断更新或checkpoint/evaluation，并保留失败收据。

## 依赖继承与新主训练

`controller.py launch --artifact-parent <C>` 在新run的hold内完成受限导入：C必须停止、持有hold，且没有任何完整主训练checkpoint；两运行配置必须相同，冻结源码必须完整，参考/校准相关未授权源码变化会拒绝。

导入核验已训练参考的epoch19收据、最终probe、训练manifest和校准内容及其参考hash绑定。参考/校准权重文件保持原路径和原hash，只逐字节复制小型依赖收据与manifest摘要；不重训参考、不重做校准。`artifact_continuation.json` 明示父运行、源版本、原artifact hashes及 `parent_partial_main_valid=false`。

D运行新版本双卡preflight和实际主模型probe，再按登记seed42/ImageNet初始化从main epoch0、step0开始全部100轮。这是对无效部分主训练的版本化修复，不是从C主状态exact resume，也不是额外科学对照臂。后续D的同版本恢复继续要求完整模型/optimizer/scheduler/scaler/两rank RNG/sampler/评测状态和固定来源相符。

## 有界证据和资源解释

原实现与缓存修复各8次实际完整模型更新中，初始化参数hash及首次前向loss相同；原实现第2次更新已出现两rank梯度及参数hash不一致，缓存修复的全部8次更新中二者完全一致，每microbatch均覆盖856个梯度张量。最终带运行保护的版本另外进行8次验证，核对梯度、参数和优化器hash。仅使用登记训练图；诊断权重仅在内存中，不写正式checkpoint。

修复前C的主阶段step1050–1650各50步窗口约5.63–6.70 images/s；14.5秒资源样本两GPU均值70.9%/40.6%，显存8048/7928MiB，CPU总体忙碌5.0%、I/O等待0.075%。失效主训练的速度不能作为有效训练加速基线。部分梯度缺失也使低资源占用失去代表性；以D恢复完整反传后的实际吞吐为准，不通过增加worker或占满显存猜测提速。

现有30分钟监督转向D，继续检查梯度覆盖和同步、学习诊断、真实吞吐和All8覆盖。前期终点仍是已继承的健康参考20轮与校准，加上D完整main epoch0–2及其既有All8检查。达到后暂停定时任务，服务器继续至epoch99。

## 正式启动核验

D于2026-09-15 19:51:01 Asia/Shanghai完成参考/校准继承，冻结执行提交为 `84fcf47e5721317d99fe186feec08d365fc3f72c`，159个来源文件核验通过。双卡preflight和实际主模型probe完成，两rank完整梯度覆盖通过，初始化状态和RNG恢复。19:52:26开始主训练更新；截至19:54:39记录的step100，两rank均有856个梯度张量、范数一致且保护通过，没有AMP跳步或同步失败。A/B/C仍停止并持有hold。详见 [启动与资源收据](../../analysis_reports/tect_diff/ampfix_startup.json)。

D实际数据加载下step50–100窗口为1.19861秒/更新、6.6744 images/s；随后14.56秒资源样本的GPU0/1利用率均值57.875%/69.75%，显存8106/8008MiB，CPU总体忙碌6.236%、I/O等待0.058%。这是修复后有效训练的短窗口记录，尚不足以确认瓶颈或宣称端到端加速。保留已验证的同步/统计优化，不因CPU或显存有空余而修改科学配置。`tect-v2`已确认ACTIVE、每30分钟检查D；目前没有完整主epoch，前期监督尚未通过。

`selection_protocol=test_selected`：All8参与选模，工程修复与早期学习均不代表独立泛化评估或最终创新收益。
