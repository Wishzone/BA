"""回测脚本

功能：
1) 从 config.json 读取 method 与 TRate；根据 method 选择 *_train 文件（如 MB_train / TS_train / 其它前缀_train）。
2) 从 config.json 读取 symbol 与 interval，加载 data/ 下对应 csv（包含 timestamp, price）。
3) 按 TRate 进行时间切分，从训练集长度开始，依次扩展 +1 interval、+2 interval… 到末尾；
   每一步重新训练并预测下一个点，收集预测并输出到 data/{METHOD}_{SYMBOL}_{INTERVAL}.csv。

使用：
  python -m src.backtest
或
  python src/backtest.py
"""

from __future__ import annotations

import importlib
import sys
import time
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Any, Dict

import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / 'src'
# 确保无论以 "python -m src.backtest" 还是 "python src/backtest.py" 执行，都能解析到模块
for p in (ROOT, SRC_DIR):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)
CONFIG_PATH = ROOT / 'config.json'
DATA_DIR = ROOT / 'data'
MODEL_DIR = ROOT / 'models'
MODEL_DIR.mkdir(exist_ok=True)


def load_config() -> Dict[str, Any]:
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    # 合理默认
    cfg.setdefault('method', 'MB')
    cfg.setdefault('TRate', 0.8)
    if 'symbol' not in cfg or 'interval' not in cfg:
        raise ValueError('config.json 必须包含 symbol 与 interval')
    return cfg


def load_price_series(symbol: str, interval: int) -> pd.DataFrame:
    file_symbol = symbol.replace('/', '')
    path = DATA_DIR / f"{file_symbol}_{interval}.csv"
    if not path.exists():
        raise FileNotFoundError(f'未找到数据文件: {path}')
    df = pd.read_csv(path, comment='#')
    if 'timestamp' not in df.columns or 'price' not in df.columns:
        raise ValueError('CSV 需要包含 timestamp, price 列')
    df['dt'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.set_index('dt', inplace=True)
    df = df.sort_index()
    # 保持 float32，与训练脚本一致
    return df[['timestamp', 'price']].astype({'price': 'float32'})


@dataclass
class SelectedTrainer:
    module: Any
    name: str  # 方法名，如 "MB"、"TS"、其它


def select_trainer(method: str) -> SelectedTrainer:
    """根据 method 动态导入对应 *_train 模块。

    规则：优先尝试 `src.{method}_train`（大小写不敏感），若失败，再对常见别名做兜底映射。
    """
    method_clean = method.strip()
    base = method_clean.lower()
    # 支持既可加 src. 前缀，也可不加（脚本方式运行时只能找到无前缀模块名）
    candidates = [
        f"{base}_train",
        f"{method_clean}_train",
        f"src.{base}_train",
        f"src.{method_clean}_train",
    ]
    # 兜底：常见别名
    alias = {
        'mb': 'MB_train',
        'ts': 'TS_train',
    }
    if base in alias:
        # 同时插入有无前缀两个版本，确保最大兼容
        ali = alias[base]
        if ali not in candidates:
            candidates.insert(0, ali)
        pref = f"src.{ali}"
        if pref not in candidates:
            candidates.insert(1, pref)

    last_err = None
    for modname in candidates:
        try:
            mod = importlib.import_module(modname)
            return SelectedTrainer(module=mod, name=method_clean.upper())
        except Exception as e:  # 继续尝试其它候选
            last_err = e
            continue
    raise ImportError(f'无法根据 method={method!r} 定位训练模块：尝试 {candidates} 失败，最后错误：{last_err}')


def train_and_predict_once(tr: SelectedTrainer, df_slice: pd.DataFrame) -> float:
    """使用给定训练片段（完整重建模型）训练并预测下一步价格。仅使用模块默认 TrainConfig。"""
    mod = tr.module
    cfg = mod.TrainConfig()
    dataset, norm_series, mean, std = mod.prepare_data(df_slice.set_index('timestamp')[['price']], cfg.window)
    drop_last = True if len(dataset) >= cfg.batch_size else False
    loader = DataLoader(dataset, batch_size=min(cfg.batch_size, max(1, len(dataset))), shuffle=True, drop_last=drop_last)
    if hasattr(mod, 'MambaTimeSeries'):
        try:
            model = mod.MambaTimeSeries(device=cfg.device)
        except TypeError:
            model = mod.MambaTimeSeries()
    elif hasattr(mod, 'TimeSeriesTransformer'):
        model = mod.TimeSeriesTransformer()
    else:
        raise AttributeError('训练模块缺少可用模型类。')
    mod.train(model, loader, cfg.epochs, cfg.device, cfg.lr)
    recent = norm_series[-cfg.window:]
    return float(mod.predict_next(model, recent, cfg.device, mean, std))


def backtest():
    cfg = load_config()
    method = str(cfg.get('method', 'MB'))
    TRate = float(cfg.get('TRate', 0.8))
    symbol = cfg['symbol']
    interval = int(cfg['interval'])

    # 载入数据
    df = load_price_series(symbol, interval)
    n = len(df)
    if n < 10:
        raise ValueError('数据量过小，无法回测。')

    # 切分点
    split_idx = max(1, min(n - 1, int(n * TRate)))

    trainer = select_trainer(method)
    print(f"回测设置: method={trainer.name} TRate={TRate} split_idx={split_idx}/{n}")

    # 回测循环
    results = []  # 每步一个 dict：timestamp, price(真实), pred(预测)

    # 要求：从训练集长度开始，依次扩展到末尾；每步都重新训练并预测下一个点
    step_counter = 0
    start_time_all = time.time()
    for train_end in range(split_idx, n):
        # 训练数据为 [0, train_end) 的价格序列
        df_train = df.iloc[:train_end].copy()

        # 确保窗口长度足够
        # 这里使用训练模块的默认窗口
        tmp_cfg = trainer.module.TrainConfig()
        if len(df_train) < tmp_cfg.window + 1:
            # 若不足窗口，跳过直到足够为止
            continue

        # 预测目标是索引 train_end 的点（即下一个点）
        if train_end >= n:
            break
        target_row = df.iloc[train_end]
        target_ts = int(target_row['timestamp'])
        target_price = float(target_row['price'])

        # 训练 + 预测
        t0 = time.time()
        pred_price = train_and_predict_once(trainer, df_train[['timestamp', 'price']])
        dt = time.time() - t0

        results.append({
            'timestamp': target_ts,
            'price': target_price,
            'pred': float(pred_price),
        })
        step_counter += 1

        if (step_counter % 10) == 0:
            elapsed = time.time() - start_time_all
            print(f"已完成 {step_counter} 步 / {n - split_idx}  (累计 {elapsed:.1f}s) 最新真实={target_price:.4f} 预测={pred_price:.4f} 单步耗时≈{dt:.2f}s")

    # 输出 CSV：放在 data/ 下，命名如 MB_BTCUSDT_3600.csv
    outfile = DATA_DIR / f"{trainer.name}_{symbol.replace('/', '')}_{interval}.csv"
    out_df = pd.DataFrame(results, columns=['timestamp', 'pred', 'price'])
    out_df = out_df[['timestamp', 'pred', 'price']]
    out_df.to_csv(outfile, index=False)
    print(f"回测完成：共 {len(results)} 条预测，已写入 {outfile}")

    # 简单指标
    if len(results) > 0:
        mae = (out_df['pred'] - out_df['price']).abs().mean()
        rmse = ((out_df['pred'] - out_df['price']) ** 2).mean() ** 0.5
        print(f"MAE={mae:.6f} RMSE={rmse:.6f}")


if __name__ == '__main__':
    backtest()
