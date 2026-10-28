# PPO机械臂到点学习

## 本阶段做什么

原来的reach_demo.py使用10倍位置误差计算动作。新脚本ppo_reach.py使用PPO网络根据字典观测输出动作；没有混入原来的误差控制规则。FetchReach官方环境仍负责底层mocap目标和物理求解，不是从零学习关节逆运动学。

训练采用官方随机目标分布，而不是只记住原来的三个指定点。使用Dense奖励，即末端与目标距离的负值，距离越小，奖励越接近0。回合默认50个控制步，达到目标不会立即终止，继续尝试保持位置。

## 如何操作

打开本目录，依次运行：

1. `04_train_ppo.cmd`：训练约20万环境交互步。只有终端日志，没有仿真窗口，这是正常现象。每2万步进行一次验证，按验证集最终位置误差选择最佳模型。末尾自动比较随机动作、位置反馈和PPO。看到TRAINING COMPLETE表示结束。
2. `05_play_ppo.cmd`：加载最近一次训练的最佳模型及其配套归一化统计，通过MuJoCo原生查看器演示5个新目标，默认0.4倍速。界面有左右面板，可关闭窗口结束。
3. `06_evaluate_ppo.cmd`：无窗口独立评估30个目标，比较随机动作、位置反馈、PPO，输出结果表。
4. `07_training_curves.cmd`：可选，启动TensorBoard，浏览器打开终端显示的地址（通常http://localhost:6006）。看rollout/ep_rew_mean与validation/final_error_mm；前者通常越接近0越好，后者越低越好，不要求每次更新都单调改善。

每次训练新建ppo_runs下的时间戳目录，保留旧模型；latest.json记录最近完成的训练目录。演示和评估不更新模型。重新评估会重写该轮目录的test_summary和test_episodes文件；指标定义保持不变。

Anaconda Prompt逐段执行：

```bat
conda activate pytorch-2.13.0-GPU
cd /d C:\202621001534\Dexterous_Hand_Operation_Control\mujoco_rl\fetch_reach_demo
python -X utf8 ppo_reach.py train
```

训练结束后：

```bat
python -X utf8 ppo_reach.py play
python -X utf8 ppo_reach.py evaluate
```

只做短程代码检查可用`python -X utf8 ppo_reach.py train --steps 2048`，这不代表训练充分；它也会成为latest记录，正式演示要选择充分训练的目录。

选择某一轮训练：`python -X utf8 ppo_reach.py play --run ppo_runs/对应时间戳目录`。

## 文件和指标

- best_model.zip：根据验证误差选出的PPO模型。
- best_normalization.pkl：对应时刻的观测归一化统计，加载该模型时必须配套使用。
- final_model.zip、final_normalization.pkl：训练结束时的模型和统计，不一定优于验证最佳模型。
- best_checkpoint.json：最佳检查点的验证指标和训练步数。
- validation.csv：每2万步的验证结果。
- before_training.json：未训练网络的验证表现，不是随机动作基线。
- progress.csv、TensorBoard事件文件：训练过程日志。
- test_summary.csv：三种方法的汇总对比。
- test_episodes.csv：每个测试种子的目标、最终误差与回报。
- training_config.json：种子、训练步数、版本与耗时。

训练完成后运行`python plot_ppo_results.py`，生成validation_curve.png和test_comparison.png，分别显示验证误差变化与相同测试目标上的三方法对比。

final_success_rate_50mm指回合最后是否小于官方5厘米阈值；final_within_10mm_rate是最后位置误差是否小于1厘米；last_five_within_10mm_rate要求最后5个控制采样均小于1厘米；mean_final_error_mm是最终距离误差的平均值。返回的mean_return是原始奖励尺度，不是训练时归一化后的奖励。

训练seed为42，验证目标使用10000至10009，测试使用20000至20029，演示使用30000开始的种子。三种评估方法使用相同测试目标，测试集不用于挑模型。反复检查同一测试集不会增加独立样本；不同训练种子的统计结论需要以后另做实验。

## 需要理解的代码

1. `PPO('MultiInputPolicy', ...)`：观测是字典，所以用字典输入策略。
2. `VecNormalize(...)`：按训练统计调整输入与训练奖励尺度，帮助优化。演示时冻结统计，并保留原始奖励尺度用于评价。
3. `model.learn(...)`：收集数据，更新Actor和Critic。
4. `model.predict(normalizer.normalize_obs(obs), deterministic=True)`：用网络输出动作，不再使用10*error规则。

模型文件和归一化文件缺一不可，不能把最佳模型与最终归一化文件混配。训练完成不等于已经达到毫米精度，也不保证PPO优于简单位置反馈。该任务上普通控制器已知误差关系，表现优于PPO并不反常。

## 这一轮的学习目标

体验训练前后的区别，理解Actor怎样根据目标输出XYZ移动动作、Critic怎样辅助训练、密集奖励为何与距离有关。再比较PPO与手写控制器，而不是只判断是否有画面。

## 2026年10月28日验证结果

首轮训练实际200704步，耗时约443秒。30个未用于选模型的测试目标中，PPO在30个目标上最终误差均小于官方5厘米阈值，24个目标小于1厘米，平均最终误差5.651毫米。相同目标下位置反馈控制平均4.788毫米、随机动作243.627毫米。此前三个固定目标的结果不能与这里30个随机目标直接混为同一测试集。

原生查看器已使用加载后的模型实际运行验证；模型推理配套加载best_normalization.pkl，统计冻结，未使用手写位置误差规则。该结果是一个训练种子、一个固定测试集上的学习实验，不是多训练种子的统计结论。
