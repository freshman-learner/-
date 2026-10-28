"""FetchReach 第一步：随机控制与三点位置反馈控制。"""
import argparse
import csv
from contextlib import nullcontext
from datetime import datetime
import json
from pathlib import Path
import time

import gymnasium as gym
import gymnasium_robotics
import numpy as np

ROOT = Path(__file__).resolve().parent
ENV_ID = 'FetchReachDense-v4'
# 相对于同一个初始末端位置的偏移，单位为米；首次运行会保存绝对目标坐标。
TARGET_OFFSETS = np.array([[0.08, -0.06, -0.03], [-0.08, 0.06, 0.06], [0.03, 0.10, -0.07]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['check', 'random', 'reach'])
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--cycles', type=int, default=1)
    parser.add_argument('--snapshot', action='store_true')
    parser.add_argument('--playback-speed', type=float, default=1.0,
                        help='Window playback speed; 0.4 means 2.5 times slower. Physics is unchanged.')
    args = parser.parse_args()
    if args.cycles < 1:
        parser.error('cycles must be positive')
    if not np.isfinite(args.playback_speed) or args.playback_speed <= 0:
        parser.error('playback-speed must be finite and positive')
    gym.register_envs(gymnasium_robotics)
    # 不再启用Gymnasium的human窗口；图形显示交给MuJoCo原生查看器。
    env = gym.make(ENV_ID, render_mode=None)
    viewer = None
    folder = ROOT/'results'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    folder.mkdir(parents=True)
    traces, summary = [], []
    try:
        obs, _ = env.reset(seed=42)
        start = obs['achieved_goal'].copy()
        targets = start+TARGET_OFFSETS
        print('环境：', ENV_ID, flush=True)
        print('当前末端坐标：', start, flush=True)
        print('三个指定目标坐标：\n', targets, flush=True)
        print('观测字段：', {k: v.shape for k,v in obs.items()}, flush=True)
        print('动作空间：', env.action_space, flush=True)
        if args.mode == 'check':
            env.step(np.zeros(4, dtype=np.float32))
            print('环境重置和物理推进通过。', flush=True)
            return
        if not args.headless:
            from mujoco import viewer as native_viewer
            viewer = native_viewer.launch_passive(
                env.unwrapped.model, env.unwrapped.data,
                show_left_ui=True, show_right_ui=True)
            with viewer.lock():
                viewer.cam.lookat[:] = [1.1, 0.75, 0.55]
                viewer.cam.distance = 2.0
                viewer.cam.azimuth = 135
                viewer.cam.elevation = -25
            print('已打开MuJoCo原生查看器：左右面板开启，关闭窗口可结束演示。', flush=True)
            print(f'画面播放速度：{args.playback_speed:g}倍；仿真步长和控制动作保持原样。', flush=True)
        for cycle in range(args.cycles):
            for target_id, target in enumerate(targets):
                if viewer is not None and not viewer.is_running():
                    print('查看器已关闭，结束演示。', flush=True)
                    return
                with viewer.lock() if viewer is not None else nullcontext():
                    obs, _ = env.reset(seed=42)
                    env.action_space.seed(1000+target_id)
                    # 仅设置任务目标；没有直接改关节或末端物理位置。
                    env.unwrapped.goal = target.copy()
                    obs['desired_goal'] = target.copy()
                    if viewer is not None:
                        # 原先由Gymnasium.render调用；原生窗口需要显式更新目标site。
                        env.unwrapped._render_callback()
                if viewer is not None:
                    viewer.sync()
                first_official_success = None
                last_errors = []
                for step in range(env.spec.max_episode_steps):
                    if viewer is not None and not viewer.is_running():
                        print('查看器已关闭，结束演示。', flush=True)
                        return
                    tick = time.perf_counter()
                    error = obs['desired_goal']-obs['achieved_goal']
                    if args.mode == 'random':
                        action = env.action_space.sample()
                        action[3] = 0.0
                    else:
                        # 官方环境将前三个动作乘0.05后用作笛卡尔位置目标增量。
                        # 10*误差对应约0.5*误差的目标增量，不直接控制关节力矩。
                        action = np.zeros(4, dtype=np.float32)
                        action[:3] = np.clip(10.0*error, -1, 1)
                    with viewer.lock() if viewer is not None else nullcontext():
                        obs, reward, terminated, truncated, info = env.step(action)
                    if viewer is not None:
                        viewer.sync()
                    distance = float(np.linalg.norm(obs['achieved_goal']-target))
                    last_errors.append(distance)
                    if bool(info['is_success']) and first_official_success is None:
                        first_official_success = (step+1)*env.unwrapped.dt
                    traces.append({'mode': args.mode, 'cycle': cycle+1, 'target_id': target_id+1,
                                   'step': step+1, 'time_s': (step+1)*env.unwrapped.dt,
                                   'error_m': distance, 'reward': float(reward),
                                   'official_success': bool(info['is_success']),
                                   **{f'target_{a}_m': float(target[i]) for i,a in enumerate('xyz')},
                                   **{f'actual_{a}_m': float(obs['achieved_goal'][i]) for i,a in enumerate('xyz')},
                                   **{f'action_{i}': float(a) for i,a in enumerate(action)}})
                    if not args.headless:
                        time.sleep(max(0, env.unwrapped.dt/args.playback_speed-(time.perf_counter()-tick)))
                    if terminated or truncated:
                        break
                row = {'mode': args.mode, 'cycle': cycle+1, 'target_id': target_id+1,
                       'final_error_mm': last_errors[-1]*1000,
                       'final_official_success': bool(info['is_success']),
                       'first_official_success_time_s': first_official_success,
                       'last_five_samples_within_10mm': bool(np.all(np.array(last_errors[-5:]) < 0.01))}
                summary.append(row)
                print(f"目标{target_id+1}：末端误差 {row['final_error_mm']:.3f} mm，"
                      f"官方达标={row['final_official_success']}，末5步小于10mm={row['last_five_samples_within_10mm']}", flush=True)
        for name, rows in [('tracking.csv', traces), ('summary.csv', summary)]:
            with (folder/name).open('w', encoding='utf-8-sig', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        (folder/'experiment.json').write_text(json.dumps({
            'environment': ENV_ID, 'mode': args.mode, 'seed': 42,
            'initial_position_m': start.tolist(), 'targets_m': targets.tolist(),
            'cycles': args.cycles, 'control_timestep_s': env.unwrapped.dt,
            'viewer': 'none' if args.headless else 'mujoco_native_passive',
            'playback_speed': None if args.headless else args.playback_speed,
            'official_distance_threshold_m': env.unwrapped.distance_threshold,
            'results': summary,
            'note': 'Scripted feedback control, not a trained RL policy. '
                    'Repeated cycles use identical initial states and goals.'}, ensure_ascii=False, indent=2), encoding='utf-8')
        if args.snapshot:
            from PIL import Image
            image_env = gym.make(ENV_ID, render_mode='rgb_array')
            try:
                obs, _ = image_env.reset(seed=42)
                target = targets[-1]
                image_env.unwrapped.goal = target.copy()
                obs['desired_goal'] = target.copy()
                for _ in range(image_env.spec.max_episode_steps):
                    action = np.zeros(4, dtype=np.float32)
                    action[:3] = np.clip(10*(target-obs['achieved_goal']), -1, 1)
                    obs, _, terminated, truncated, _ = image_env.step(action)
                    if terminated or truncated:
                        break
                Image.fromarray(image_env.render()).save(folder/'preview.png')
            finally:
                image_env.close()
        print('结果目录：', folder, flush=True)
    finally:
        if viewer is not None:
            viewer.close()
        env.close()


if __name__ == '__main__':
    main()
