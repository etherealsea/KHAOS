import argparse
import contextlib
import os
import sys

PROJECT_SRC = r'd:\《基于物理信息增强神经网络（PI-KAN）的金融时间序列相变探测研究》\Finance\02_核心代码\源代码'
SAVE_DIR = r'd:\《基于物理信息增强神经网络（PI-KAN）的金融时间序列相变探测研究》\Finance\02_核心代码\模型权重备份\iter9_round'
LOG_PATH = r'd:\《基于物理信息增强神经网络（PI-KAN）的金融时间序列相变探测研究》\Finance\04_项目文档\99_归档\测试日志\2026-03-26_iter9_full_train.log'
RESUME_PATH = r'd:\《基于物理信息增强神经网络（PI-KAN）的金融时间序列相变探测研究》\Finance\02_核心代码\模型权重备份\iter9_round\khaos_kan_resume.pth'

sys.path.append(PROJECT_SRC)

from khaos.模型训练.train import train

def main():
    args = argparse.Namespace(
        data_dir=r'd:\《基于物理信息增强神经网络（PI-KAN）的金融时间序列相变探测研究》\Finance\01_数据中心\03_研究数据\research_processed',
        save_dir=SAVE_DIR,
        epochs=16,
        batch_size=256,
        lr=1e-3,
        window_size=20,
        horizon=10,
        hidden_dim=64,
        layers=3,
        grid_size=10,
        seed=42,
        deterministic=True,
        test_mode=False,
        fast_full=False,
        early_stop_patience=4,
        early_stop_min_delta=0.0015,
        resume=True,
        resume_path=RESUME_PATH,
    )
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    os.makedirs(SAVE_DIR, exist_ok=True)
    log_mode = 'a' if os.path.exists(LOG_PATH) else 'w'
    with open(LOG_PATH, log_mode, encoding='utf-8', buffering=1) as log_file:
        log_file.write('\nrunner_start\n')
        log_file.flush()
        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            train(args)
        log_file.write('runner_end\n')
        log_file.flush()

if __name__ == '__main__':
    main()
