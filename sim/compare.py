from __future__ import annotations

import argparse
import csv
import json
import math
import operator
import sys
from functools import reduce
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / 'config.json'


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='对比原始与预测结果，输出指标与图表')
    ap.add_argument('--source', help='原始价格 CSV 路径，缺省时自动根据配置推断')
    ap.add_argument('--pred', help='预测结果 CSV 路径，缺省时按 method_<source> 推断')
    ap.add_argument('--out', help='输出 HTML 路径，缺省为 compare_<source_stem>.html')
    ap.add_argument('--fee', type=float, help='覆盖手续费率，默认采用配置文件中的 fee')
    ap.add_argument('--no-open', action='store_true', help='生成图表后不自动打开浏览器')
    return ap.parse_args()


def load_config(path: Path = CONFIG_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f'配置文件不存在: {path}')
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def normalize_path(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def resolve_source_path(cfg: dict, override: str | None) -> Path:
    if override:
        return normalize_path(override)
    cfg_path = cfg.get('compare_source') or cfg.get('source')
    if cfg_path:
        return normalize_path(cfg_path)
    symbol = str(cfg.get('symbol', '')).replace('/', '')
    interval = cfg.get('interval')
    if symbol and interval is not None:
        return normalize_path(Path('data') / f'{symbol}_{interval}.csv')
    raise ValueError('缺少源数据路径，请通过 --source 或在 config.json 中配置 compare_source')


def resolve_pred_path(source_path: Path, method_prefix: str, override: str | None) -> Path:
    if override:
        return normalize_path(override)
    prefix = method_prefix.strip().upper() if method_prefix else 'MB'
    return source_path.with_name(f'{prefix}_{source_path.name}')


def resolve_output_path(source_path: Path, override: str | None) -> Path:
    if override:
        return normalize_path(override)
    return PROJECT_ROOT / f'compare_{source_path.stem}.html'


def resolve_fee(args: argparse.Namespace, cfg: dict) -> float:
    if args.fee is not None:
        return max(0.0, args.fee)
    fee = cfg.get('fee', 0.001)
    try:
        return max(0.0, float(fee))
    except (TypeError, ValueError):
        return 0.001


def should_open_browser(args: argparse.Namespace, cfg: dict) -> bool:
    return (not args.no_open) and bool(cfg.get('open_browser', True))


def load_price_series(path: Path, label: str) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(f'{label} 文件不存在: {path}')
    with path.open('r', encoding='utf-8') as fh:
        first_line = fh.readline()
    skiprows = 1 if first_line.startswith('#') else 0
    df = pd.read_csv(path, skiprows=skiprows)
    if 'timestamp' not in df.columns or 'price' not in df.columns:
        raise ValueError(f'{label} 文件缺少 timestamp 或 price 列: {path}')
    df['ts'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True, errors='coerce')
    df = df.dropna(subset=['ts']).sort_values('ts').set_index('ts')
    return df['price'].astype(float)


def build_figure(actual: pd.Series, pred: pd.Series, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual.index, y=actual.values, mode='lines', name='Actual'))
    if len(pred):
        fig.add_trace(go.Scatter(x=pred.index, y=pred.values, mode='lines+markers', name='Pred(backtest)'))
    fig.update_layout(title=title, template='plotly_white', xaxis_title='Time', yaxis_title='Price')
    return fig


def calculate_god_return(actual: pd.Series, pred: pd.Series, fee: float):
    actual_sorted = actual.sort_index()
    pred_sorted = pred.sort_index()
    actual_index = actual_sorted.index
    records = []
    factors = []
    for ts, pred_price in pred_sorted.items():
        pos = actual_index.searchsorted(ts) - 1
        if pos < 0:
            continue
        prev_ts = actual_index[pos]
        prev_price = actual_sorted.iloc[pos]
        if prev_price <= 0 or not math.isfinite(prev_price) or not math.isfinite(pred_price):
            continue
        ratio = float(pred_price / prev_price)
        net_ratio = ratio - fee
        used = ratio >= (1.0 + fee)
        if used:
            factors.append(net_ratio)
        records.append({
            'predict_ts': ts,
            'prev_actual_ts': prev_ts,
            'prev_price': prev_price,
            'pred_price': pred_price,
            'ratio': ratio,
            'net_ratio': net_ratio,
            'used': int(used),
        })
    god_return = reduce(operator.mul, factors, 1.0) if factors else 1.0
    return records, god_return


def write_god_return_details(records: list[dict], output_path: Path) -> None:
    if not records:
        return
    detail_path = output_path.with_name(output_path.stem + '_god_return_details.csv')
    with detail_path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['predict_ts', 'prev_actual_ts', 'prev_price', 'pred_price', 'ratio', 'net_ratio', 'used'])
        for rec in records:
            writer.writerow([
                int(rec['predict_ts'].timestamp() * 1000),
                int(rec['prev_actual_ts'].timestamp() * 1000),
                f"{rec['prev_price']:.8f}",
                f"{rec['pred_price']:.8f}",
                f"{rec['ratio']:.6f}",
                f"{rec['net_ratio']:.6f}",
                rec['used'],
            ])
    print('god return details saved', detail_path)


def main():
    args = parse_args()
    try:
        cfg = load_config()
    except Exception as exc:
        print(f'加载配置失败: {exc}', file=sys.stderr)
        sys.exit(1)

    try:
        source_path = resolve_source_path(cfg, args.source)
    except Exception as exc:
        print(f'解析源数据路径失败: {exc}', file=sys.stderr)
        sys.exit(1)

    try:
        actual_series = load_price_series(source_path, '源数据')
    except Exception as exc:
        print(f'加载源数据失败: {exc}', file=sys.stderr)
        sys.exit(1)

    method_prefix = str(cfg.get('method', 'MB')).strip().upper() or 'MB'
    try:
        pred_path = resolve_pred_path(source_path, method_prefix, args.pred)
        pred_series = load_price_series(pred_path, '预测数据')
    except Exception as exc:
        print(f'加载预测数据失败: {exc}', file=sys.stderr)
        sys.exit(1)

    fee = resolve_fee(args, cfg)
    title = f'Compare {source_path.name} (actual={len(actual_series)} pred={len(pred_series)})'
    figure = build_figure(actual_series, pred_series, title)
    output_path = resolve_output_path(source_path, args.out)
    figure.write_html(str(output_path))
    print('saved', output_path)

    if len(pred_series) == 0:
        print('无预测数据, 跳过上帝收益计算')
    else:
        records, god_return = calculate_god_return(actual_series, pred_series, fee)
        kept = sum(rec['used'] for rec in records)
        total = len(records)
        print(f'预期上帝收益 (fee={fee:.4f}) = {god_return:.6f} (保留 {kept}/{total} 条, 仅累乘满足 ratio>=1+fee 的记录)')
        write_god_return_details(records, output_path)

    if should_open_browser(args, cfg):
        try:
            figure.show()
        except Exception:
            pass


if __name__ == '__main__':
    main()
