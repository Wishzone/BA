"""时间序列预测脚本

功能:
1. 读取 `config.json` 中的 symbol, interval
2. 根据配置读取 `data/` 下 csv (格式: SYMBOLINTERVAL 例如 BTCUSDT_3600.csv)
3. 使用 Transformer 训练并预测向后 1 个 interval 的价格

无需命令行参数, 直接运行即可: `python -m src.TS_train` 或 `python src/TS_train.py`
"""

from __future__ import annotations

import json
from pathlib import Path
import math
from dataclasses import dataclass
from typing import Tuple
import os

import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / 'config.json'
DATA_DIR = ROOT / 'data'
MODEL_DIR = ROOT / 'models'
MODEL_DIR.mkdir(exist_ok=True)


def load_config() -> Tuple[str, int]:
	with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
		cfg = json.load(f)
	symbol = cfg.get('symbol')  # e.g. BTC/USDT
	interval = int(cfg.get('interval'))
	if not symbol or not interval:
		raise ValueError('config.json 必须包含 symbol 与 interval')
	return symbol, interval


def load_price_series(symbol: str, interval: int) -> pd.DataFrame:
	file_symbol = symbol.replace('/', '')
	fname = f"{file_symbol}_{interval}.csv"
	path = DATA_DIR / fname
	if not path.exists():
		raise FileNotFoundError(f'未找到数据文件: {path}')
	# 文件前两行: meta 注释 + header
	df = pd.read_csv(path, comment='#')
	# 期望列 timestamp, price
	if 'timestamp' not in df.columns or 'price' not in df.columns:
		raise ValueError('CSV 需要包含 timestamp, price 列')
	df['dt'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
	df.set_index('dt', inplace=True)
	df = df.sort_index()
	return df[['price']].astype('float32')


class SeriesWindowDataset(Dataset):
	def __init__(self, series: torch.Tensor, window: int):
		self.series = series  # shape [T]
		self.window = window

	def __len__(self):
		return len(self.series) - self.window

	def __getitem__(self, idx):
		x = self.series[idx:idx + self.window]
		y = self.series[idx + self.window]
		return x.unsqueeze(-1), y.unsqueeze(-1)


class TimeSeriesTransformer(nn.Module):
	def __init__(self, d_model=32, nhead=2, num_layers=1, dim_feedforward=64, dropout=0.1):
		super().__init__()
		self.input_proj = nn.Linear(1, d_model)
		encoder_layer = nn.TransformerEncoderLayer(
			d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,
			batch_first=True, norm_first=False
		)
		self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
		self.positional = PositionalEncoding(d_model, dropout)
		self.head = nn.Linear(d_model, 1)

	def forward(self, x):  # x: [B, L, 1]
		h = self.input_proj(x)
		h = self.positional(h)
		h = self.encoder(h)
		# 取最后一个时间步输出
		out = self.head(h[:, -1])
		return out


class PositionalEncoding(nn.Module):
	def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10000):
		super().__init__()
		self.dropout = nn.Dropout(dropout)
		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0)  # [1, max_len, d_model]
		self.register_buffer('pe', pe, persistent=False)

	def forward(self, x):  # x: [B, L, D]
		x = x + self.pe[:, :x.size(1)]
		return self.dropout(x)


@dataclass
class TrainConfig:
	window: int = 64
	batch_size: int = 256
	epochs: int = 1
	lr: float = 2e-3
	device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


def prepare_data(df: pd.DataFrame, window: int) -> Tuple[SeriesWindowDataset, torch.Tensor, float, float]:
	prices = torch.tensor(df['price'].values, dtype=torch.float32)
	mean = prices.mean()
	std = prices.std() + 1e-8
	norm = (prices - mean) / std
	dataset = SeriesWindowDataset(norm, window)
	return dataset, norm, float(mean), float(std)


def train(model: nn.Module, loader: DataLoader, epochs: int, device: str, lr: float) -> None:
	optim = torch.optim.Adam(model.parameters(), lr=lr)
	loss_fn = nn.MSELoss()
	model.to(device)
	for ep in range(1, epochs + 1):
		model.train()
		total = 0.0
		for xb, yb in loader:
			xb = xb.to(device)
			yb = yb.to(device)
			pred = model(xb)
			loss = loss_fn(pred, yb)
			optim.zero_grad()
			loss.backward()
			optim.step()
			total += loss.item() * xb.size(0)
		avg = total / len(loader.dataset)
		print(f'Epoch {ep}/{epochs} loss={avg:.6f}')


def predict_next(model: nn.Module, recent_window: torch.Tensor, device: str, mean: float, std: float) -> float:
	model.eval()
	with torch.no_grad():
		x = recent_window.unsqueeze(0).unsqueeze(-1).to(device)
		y_norm = model(x).cpu().item()
	y = y_norm * std + mean
	return float(y)


def main():
	symbol, interval = load_config()
	df = load_price_series(symbol, interval)
	print(f'{symbol} interval={interval}, {len(df)}行, {df.index[0]} -> {df.index[-1]}')

	cfg = TrainConfig()
	dataset, norm_series, mean, std = prepare_data(df, cfg.window)
	loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

	model = TimeSeriesTransformer()
	train(model, loader, cfg.epochs, cfg.device, cfg.lr)

	recent = norm_series[-cfg.window:]
	pred_price = predict_next(model, recent, cfg.device, mean, std)
	last_price = df['price'].iloc[-1]
	print(f'最后一个已知价格: {last_price:.2f}')
	print(f'预测下一个 interval 价格: {pred_price:.2f}')

	model_path = MODEL_DIR / f"ts_{symbol.replace('/', '')}_{interval}.pt"
	torch.save({
		'model_state': model.state_dict(),
		'mean': mean,
		'std': std,
		'window': cfg.window,
		'interval': interval,
		'symbol': symbol,
	}, model_path)
	print(f'模型已保存: {model_path}')


if __name__ == '__main__':
	main()

