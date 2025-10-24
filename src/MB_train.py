"""时间序列预测脚本 (Mamba 版本)

功能:
1. 读取 `config.json` 中的 symbol, interval
2. 根据配置读取 `data/` 下 csv (格式: SYMBOLINTERVAL 例如 BTCUSDT_3600.csv)
3. 使用 Mamba 风格的序列模型训练并预测向后 1 个 interval 的价格

无需命令行参数, 直接运行即可: `python -m src.MB_train` 或 `python src/MB_train.py`

说明:
- 若本机已安装 `mamba-ssm` 库，将优先使用其官方实现；
- 若未安装，则回退到一个轻量化的 Mamba 风格 SSM 块（深度可分离卷积 + 门控 + 残差），以保持无依赖可运行。
"""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple
import math
import os

# 降低 C++ 侧日志等级，抑制 NNPACK 等硬件不支持的告警
os.environ.setdefault("GLOG_minloglevel", "2")      # 2=ERROR，只显示 ERROR/FATAL
os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")

import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

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
	# 文件前两行: 可能有 meta 注释 + header
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


# ============== 可选: 使用官方 mamba-ssm ==============
_HAS_MAMBA_LIB = False
try:
	# 尝试导入 mamba-ssm 官方实现
	from mamba_ssm import Mamba as MambaBlockLib  # type: ignore
	_HAS_MAMBA_LIB = True
except Exception:
	_HAS_MAMBA_LIB = False


class TinyMambaBlock(nn.Module):
	"""轻量 Mamba 风格块: 深度可分离因果卷积 + 门控 + 残差 + 归一化

	不是官方实现，但具备线性时序建模与门控选择的特征，适合无依赖场景。
	输入/输出: [B, L, D]
	"""

	def __init__(self, d_model: int, d_conv: int = 4, dropout: float = 0.1):
		super().__init__()
		self.d_model = d_model
		self.k = d_conv
		# depthwise conv (causal)
		self.dw = nn.Conv1d(d_model, d_model, kernel_size=d_conv, groups=d_model, bias=True)
		# pointwise convs for mixing and gating
		self.pw1 = nn.Conv1d(d_model, d_model, kernel_size=1)
		self.pw_gate = nn.Conv1d(d_model, d_model, kernel_size=1)
		self.out = nn.Conv1d(d_model, d_model, kernel_size=1)
		self.norm = nn.LayerNorm(d_model)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# x: [B, L, D]
		residual = x
		x = self.norm(x)
		x_t = x.transpose(1, 2)  # [B, D, L]
		# causal padding: pad left with k-1
		x_pad = F.pad(x_t, (self.k - 1, 0))
		h = self.dw(x_pad)
		h = F.silu(self.pw1(h))
		g = torch.sigmoid(self.pw_gate(x_t))
		y = self.out(h * g).transpose(1, 2)  # back to [B, L, D]
		y = self.dropout(y)
		return residual + y


class MambaTimeSeries(nn.Module):
	"""时间序列回归模型: 输入 [B, L, 1] -> 预测下一个值 [B, 1]

	优先使用 mamba-ssm 官方模块；若不可用则回退到 TinyMambaBlock 堆叠。
	"""

	def __init__(
		self,
		d_model: int = 64,
		n_layers: int = 4,
		d_state: int = 16,
		d_conv: int = 4,
		expand: int = 2,
		dropout: float = 0.1,
		device: str | None = None,
	):
		super().__init__()
		self.input_proj = nn.Linear(1, d_model)
		# 仅当显卡可用且目标设备为 cuda 时启用官方 mamba-ssm
		if device is None:
			device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.target_device = device
		self.use_lib = bool(_HAS_MAMBA_LIB and torch.cuda.is_available() and str(device).startswith('cuda'))
		self.dropout = nn.Dropout(dropout)

		if self.use_lib:
			# 使用官方 Mamba 模块栈（若 API 不同，可根据本地版本调整）
			layers = []
			for _ in range(n_layers):
				# 常见构造签名: Mamba(d_model, d_state=16, d_conv=4, expand=2)
				layers.append(MambaBlockLib(d_model, d_state=d_state, d_conv=d_conv, expand=expand))
			self.backbone = nn.ModuleList(layers)
		else:
			self.backbone = nn.ModuleList([TinyMambaBlock(d_model, d_conv=d_conv, dropout=dropout) for _ in range(n_layers)])

		self.head = nn.Linear(d_model, 1)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# x: [B, L, 1]
		h = self.input_proj(x)
		if self.use_lib:
			# 官方实现通常是全序列到全序列: [B, L, D]
			for layer in self.backbone:
				h = layer(h)
		else:
			for layer in self.backbone:
				h = layer(h)
		# 取最后一个时间步
		out = self.head(h[:, -1])  # [B, 1]
		return out


@dataclass
class TrainConfig:
	window: int = 64
	batch_size: int = 256
	epochs: int = 3
	lr: float = 2e-3
	device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


def prepare_data(df: pd.DataFrame, window: int):
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
	print(f'配置: symbol={symbol} interval={interval}')
	df = load_price_series(symbol, interval)
	print(f'数据加载完成: {len(df)} 行, 时间范围 {df.index[0]} -> {df.index[-1]}')

	cfg = TrainConfig()
	dataset, norm_series, mean, std = prepare_data(df, cfg.window)
	loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

	model = MambaTimeSeries(device=cfg.device)

	train(model, loader, cfg.epochs, cfg.device, cfg.lr)

	recent = norm_series[-cfg.window:]
	pred_price = predict_next(model, recent, cfg.device, mean, std)
	last_price = df['price'].iloc[-1]
	print(f'最后一个已知价格: {last_price:.2f}')
	print(f'预测下一个 interval 价格: {pred_price:.2f}')

	model_path = MODEL_DIR / f"mb_{symbol.replace('/', '')}_{interval}.pt"
	torch.save({
		'model_state': model.state_dict(),
		'mean': mean,
		'std': std,
		'window': cfg.window,
		'interval': interval,
		'symbol': symbol,
		'use_mamba_lib': _HAS_MAMBA_LIB,
	}, model_path)
	print(f'模型已保存: {model_path}')


if __name__ == '__main__':
	main()

