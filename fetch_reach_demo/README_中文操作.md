# FetchReach 机械臂到点复现

## 复现范围

Gymnasium-Robotics是一套环境集合，不是一个下载后就学会操作的模型。这里先使用官方FetchReachDense-v4完成基础控制，再考虑PPO训练。

reach_demo.py对应第一阶段：随机动作对照、位置反馈控制三个指定目标、记录位置误差、保存图像。ppo_reach.py对应第二阶段：PPO随机目标训练、模型保存、加载演示和独立评估。两阶段均已运行验证，不是复现完整仓库的全部任务或论文性能。

## 已安装环境

解释器：`D:\software\Anaconda\envs\pytorch-2.13.0-GPU\python.exe`。

Gymnasium-Robotics 1.4.2、Gymnasium 1.4.0、MuJoCo 3.15.0。官方Robotics包含有模型和网格，本阶段无需手动克隆整个GitHub仓库。`_downloads`保留了已核对PyPI SHA256的官方wheel，`download_robotics.py`只是网络中断时的下载辅助脚本，日常学习无需运行。

官方主要测试Linux/macOS；本机Windows的无窗口运行和图像渲染已经通过。图形演示现已改为MuJoCo原生查看器，默认打开左侧设置面板和右侧Joint、Control等面板。

## 操作步骤

打开文件夹：

`C:\202621001534\Dexterous_Hand_Operation_Control\mujoco_rl\fetch_reach_demo`

1. 双击 `01_random.cmd`：观察机械臂随机运动，末端通常不会停在目标处。三个目标各50个控制步，循环2轮；窗口默认0.4倍速播放，约慢2.5倍，整段约30秒。
2. 双击 `02_reach_targets.cmd`：观察反馈控制下末端接近三个目标。每个目标为独立回合，从相同初始状态开始，循环3轮；回合切换时重置，不是连续经过三点的轨迹规划。
3. 双击 `03_evaluate_headless.cmd`：没有窗口，统计三个目标误差，并保存场景截图。

末尾按任意键关闭命令窗口。每回合约2秒仿真时间；随机演示0.4倍播放对应约5秒现实时间，实际耗时还受渲染影响。只调整展示节奏，不修改物理步长、随机动作或控制频率，CSV中的time_s仍是仿真时间。

需要更慢可运行 `python -X utf8 reach_demo.py random --cycles 2 --playback-speed 0.2`，恢复正常速度用 `--playback-speed 1`。无窗口模式不等待，不受播放速度影响。

原生查看器使用launch_passive连接同一个环境的model和data；仿真步进由Python脚本执行，因此部分Simulation按钮禁用属于这种模式的正常表现。关闭查看器即可结束演示。每个目标切换时会更新目标标记，--headless与截图功能保持可用。

Anaconda Prompt逐段执行：

```bat
conda activate pytorch-2.13.0-GPU
cd /d C:\202621001534\Dexterous_Hand_Operation_Control\mujoco_rl\fetch_reach_demo
python -X utf8 reach_demo.py reach --cycles 3
```

无窗口运行：

```bat
python -X utf8 reach_demo.py reach --headless --snapshot
python -X utf8 reach_demo.py random --headless
python plot_comparison.py
```

## 预期看到什么

画面中有Fetch机械臂、桌面和目标标记。程序指定三个目标，机械臂末端逐步接近并停留。终端输出每个目标的最终误差；results下各次时间戳目录保存summary.csv、tracking.csv、experiment.json，截图模式另存preview.png。plot_comparison.py把最近一次随机和反馈控制的第一轮曲线画到results/tracking_comparison.png。

2026年10月28日实测反馈控制三个目标最终误差约0.376、0.431、0.419毫米；随机动作约282.070、197.336、331.866毫米。这是三个指定目标的学习演示结果，不是广泛随机目标统计。

官方成功阈值为5厘米。示例额外记录最后五个控制采样是否均小于10毫米；最终误差仍原样输出，不把官方success当作毫米级精度证据。

启动时可能出现针对Adroit任务历史奖励版本的警告；该警告不是FetchReach模型加载失败。如果出现Traceback，保留完整文字以便排查。

## 理解控制原理

观测是字典：observation包含10维末端/夹爪状态；achieved_goal为当前末端XYZ；desired_goal为目标XYZ。

动作4维，前三个为归一化笛卡尔目标位移，环境内部按0.05缩放，最后一个夹爪动作在Reach中不生效。它不是7维关节角或关节力矩。

位置反馈过程：目标减当前位置得到误差，乘以增益并限制到动作范围，传给环境执行，读取新位置，再次计算误差。底层mocap约束与物理求解由官方环境提供，因此此实验不意味着已经自行实现逆运动学或关节力矩控制。

核心代码是reach_demo.py中的TARGET_OFFSETS和action[:3] = clip(10.0*error, -1, 1)。目标偏移相对固定初始末端位置设置，每次运行会将实际世界坐标写入experiment.json。

## 后续强化学习阶段

基础到点控制理解后，再接入SB3 PPO，使用适配字典观测的MultiInputPolicy，先采用Dense奖励。完成训练、保存、加载、独立目标测试，并与上述反馈控制对照。训练结果不保证优于简单控制器，运行步数和配置需要依据实际评估调整。

该阶段现已实现。详见同目录PPO阶段操作说明.md；先双击05_play_ppo.cmd观看已有模型，再用04_train_ppo.cmd自己训练；06_evaluate_ppo.cmd独立评估，07_training_curves.cmd查看训练曲线。

本阶段的汇报成果为机械臂到点演示、三个目标坐标、随机与反馈误差曲线，以及对观测和动作含义的解释。

官方来源：
- https://github.com/Farama-Foundation/Gymnasium-Robotics
- https://robotics.farama.org/envs/fetch/reach/
- https://robotics.farama.org/content/installation/
