"""PPO训练、独立评估和原生查看器演示；使用官方随机目标分布。"""
import argparse
from contextlib import nullcontext
import csv
from datetime import datetime
from importlib.metadata import version
import json
from pathlib import Path
import time

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

ROOT = Path(__file__).resolve().parent
ENV_ID = 'FetchReachDense-v4'


def csv_file(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def evaluate(model, normalizer, seeds, method='ppo'):
    """每种控制方法使用完全相同的初始种子和目标；奖励使用原环境尺度。"""
    env = gym.make(ENV_ID)
    results = []
    try:
        for seed in seeds:
            obs, _ = env.reset(seed=seed)
            env.action_space.seed(seed+100000)
            target = obs['desired_goal'].copy()
            total, errors = 0.0, []
            for step in range(env.spec.max_episode_steps):
                if method == 'random':
                    action = env.action_space.sample(); action[3] = 0
                elif method == 'feedback':
                    action = np.zeros(4, dtype=np.float32)
                    action[:3] = np.clip(10*(obs['desired_goal']-obs['achieved_goal']), -1, 1)
                else:
                    # 必须使用与该模型配套的观测归一化统计。
                    action, _ = model.predict(normalizer.normalize_obs(obs), deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total += float(reward)
                errors.append(float(np.linalg.norm(obs['achieved_goal']-target)))
                if terminated or truncated:
                    break
            results.append({'method': method, 'seed': seed,
                            **{f'target_{a}_m': float(target[i]) for i,a in enumerate('xyz')},
                            'final_error_mm': errors[-1]*1000,
                            'final_success_50mm': bool(info['is_success']),
                            'final_within_10mm': errors[-1]<0.01,
                            'last_five_within_10mm': bool(np.all(np.array(errors[-5:])<0.01)),
                            'return': total, 'steps': len(errors)})
    finally:
        env.close()
    summary = {'method': method, 'episodes': len(results),
               'mean_final_error_mm': float(np.mean([r['final_error_mm'] for r in results])),
               'final_success_rate_50mm': float(np.mean([r['final_success_50mm'] for r in results])),
               'final_within_10mm_rate': float(np.mean([r['final_within_10mm'] for r in results])),
               'last_five_within_10mm_rate': float(np.mean([r['last_five_within_10mm'] for r in results])),
               'mean_return': float(np.mean([r['return'] for r in results]))}
    return summary, results


class ValidationCallback(BaseCallback):
    def __init__(self, folder, interval=20000):
        super().__init__()
        self.folder, self.interval, self.next_eval = folder, interval, interval
        self.best_error = float('inf')
        self.rows = []

    def _on_step(self):
        if self.num_timesteps < self.next_eval:
            return True
        self.next_eval += self.interval
        normalizer = self.model.get_vec_normalize_env()
        summary, _ = evaluate(self.model, normalizer, range(10000, 10010))
        summary['timesteps'] = self.num_timesteps
        self.rows.append(summary)
        csv_file(self.folder/'validation.csv', self.rows)
        self.logger.record('validation/final_error_mm', summary['mean_final_error_mm'])
        self.logger.record('validation/success_50mm', summary['final_success_rate_50mm'])
        print(f"验证 {self.num_timesteps}步：平均误差 {summary['mean_final_error_mm']:.2f}mm，"
              f"5cm达标 {summary['final_success_rate_50mm']:.0%}", flush=True)
        if summary['mean_final_error_mm'] < self.best_error:
            self.best_error = summary['mean_final_error_mm']
            self.model.save(self.folder/'best_model')
            normalizer.save(self.folder/'best_normalization.pkl')
            (self.folder/'best_checkpoint.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        return True


def load_run(folder, raw_env):
    model = PPO.load(folder/'best_model.zip', device='cpu')
    normalizer = VecNormalize.load(folder/'best_normalization.pkl', DummyVecEnv([lambda: raw_env]))
    normalizer.training = False
    normalizer.norm_reward = False
    return model, normalizer


def select_run(args):
    if args.run:
        return args.run.resolve()
    return Path(json.loads((ROOT/'ppo_runs'/'latest.json').read_text(encoding='utf-8'))['run'])


def train(args):
    folder = ROOT/'ppo_runs'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    folder.mkdir(parents=True)
    env = VecNormalize(make_vec_env(ENV_ID, n_envs=4, seed=args.seed), norm_obs=True, norm_reward=True)
    try:
        model = PPO('MultiInputPolicy', env, device='cpu', seed=args.seed,
                    learning_rate=3e-4, n_steps=512, batch_size=256, n_epochs=10,
                    gamma=0.98, gae_lambda=0.95, ent_coef=0.0,
                    policy_kwargs={'net_arch': dict(pi=[64,64], vf=[64,64]), 'log_std_init': -0.5},
                    verbose=1)
        model.set_logger(configure(str(folder), ['stdout', 'csv', 'tensorboard']))
        before, rows = evaluate(model, env, range(10000,10010))
        csv_file(folder/'before_training.csv', rows)
        (folder/'before_training.json').write_text(json.dumps(before, indent=2), encoding='utf-8')
        callback = ValidationCallback(folder)
        print('开始训练：随机目标，无图形窗口。', flush=True)
        started = time.perf_counter()
        model.learn(total_timesteps=args.steps, callback=callback)
        model.save(folder/'final_model')
        env.save(folder/'final_normalization.pkl')
        if not (folder/'best_model.zip').exists():
            model.save(folder/'best_model'); env.save(folder/'best_normalization.pkl')
        (folder/'training_config.json').write_text(json.dumps({
            'env_id': ENV_ID, 'seed': args.seed, 'requested_steps': args.steps,
            'actual_steps': model.num_timesteps, 'elapsed_seconds': time.perf_counter()-started,
            'training_goal_distribution': 'official random goals',
            'validation_seeds': [10000,10009], 'heldout_test_seeds': [20000,20029],
            'checkpoint_selection': 'minimum validation mean final position error',
            'normalization': 'VecNormalize observation and training reward',
            'versions': {n:version(n) for n in ['gymnasium-robotics','gymnasium','mujoco','stable-baselines3','torch']}
        }, indent=2), encoding='utf-8')
        (ROOT/'ppo_runs'/'latest.json').write_text(json.dumps({'run': str(folder)}, indent=2), encoding='utf-8')
    finally:
        env.close()
    args.run = folder
    evaluate_run(args)
    print('TRAINING COMPLETE：', folder, flush=True)


def evaluate_run(args):
    folder = select_run(args)
    raw_env = gym.make(ENV_ID)
    model, normalizer = load_run(folder, raw_env)
    try:
        summaries, records = [], []
        for method in ['random','feedback','ppo']:
            summary, rows = evaluate(model, normalizer, range(20000,20030), method)
            summaries.append(summary); records.extend(rows)
            print(f"{method}：平均最终误差 {summary['mean_final_error_mm']:.2f}mm，"
                  f"5cm达标 {summary['final_success_rate_50mm']:.0%}，"
                  f"1cm达标 {summary['final_within_10mm_rate']:.0%}", flush=True)
        csv_file(folder/'test_summary.csv', summaries)
        csv_file(folder/'test_episodes.csv', records)
        (folder/'test_summary.json').write_text(json.dumps(summaries, indent=2), encoding='utf-8')
    finally:
        normalizer.close()


def play(args):
    folder = select_run(args)
    env = gym.make(ENV_ID)
    model, normalizer = load_run(folder, env)
    viewer = None
    traces = []
    try:
        obs, _ = env.reset(seed=30000)
        from mujoco import viewer as native_viewer
        viewer = native_viewer.launch_passive(env.unwrapped.model, env.unwrapped.data,
                                            show_left_ui=True, show_right_ui=True)
        with viewer.lock():
            viewer.cam.lookat[:] = [1.1,0.75,0.55]
            viewer.cam.distance = 2.0
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -25
        for episode in range(args.episodes):
            if not viewer.is_running():
                break
            with viewer.lock():
                obs, _ = env.reset(seed=30000+episode)
                env.unwrapped._render_callback()
            viewer.sync()
            for step in range(env.spec.max_episode_steps):
                if not viewer.is_running():
                    return
                tick = time.perf_counter()
                # 此处动作完全来自PPO，不混入之前的10*error规则。
                action, _ = model.predict(normalizer.normalize_obs(obs), deterministic=True)
                with viewer.lock():
                    obs, _, terminated, truncated, info = env.step(action)
                viewer.sync()
                distance = float(np.linalg.norm(obs['desired_goal']-obs['achieved_goal']))
                traces.append({'episode':episode+1,'step':step+1,'error_mm':distance*1000,
                               **{f'action_{i}':float(a) for i,a in enumerate(action)}})
                time.sleep(max(0, env.unwrapped.dt/args.playback_speed-(time.perf_counter()-tick)))
                if terminated or truncated:
                    break
            print(f'回合{episode+1}：最终误差 {distance*1000:.2f}mm，官方达标={bool(info["is_success"])}', flush=True)
    finally:
        if traces:
            csv_file(folder/f'play_{datetime.now():%Y%m%d_%H%M%S}.csv', traces)
        if viewer is not None:
            viewer.close()
        normalizer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['train','evaluate','play'])
    parser.add_argument('--steps', type=int, default=200000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--playback-speed', type=float, default=0.4)
    args = parser.parse_args()
    if args.steps < 1 or args.episodes < 1 or not np.isfinite(args.playback_speed) or args.playback_speed<=0:
        parser.error('steps, episodes, and playback-speed must be positive and finite')
    gym.register_envs(gymnasium_robotics)
    torch.set_num_threads(1)
    if args.mode != 'train' and not args.run and not (ROOT/'ppo_runs'/'latest.json').exists():
        parser.error('请先运行04_train_ppo.cmd，或通过--run指定训练结果目录。')
    {'train':train,'evaluate':evaluate_run,'play':play}[args.mode](args)


if __name__ == '__main__':
    main()
