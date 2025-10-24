""" 描述：抓取交易对市场价格，支持向后续写模式，向前补历史数据需要删除文件重新获取。
    输出：(timestamp_ms, price)格式csv文件。
    参数：'--config', default='config_prices.json', help='配置文件路径(JSON). 默认: config_prices.json'
          '--symbol', help='交易对, 例: BTC/USDT'
          '--start', help='开始时间 (ISO 或时间戳秒/毫秒)，默认: start=2020-09-01 00:00:00 UTC'
          '--end', help='结束时间 (ISO 或时间戳秒/毫秒)，默认end=当前时间'
          '--interval', type=int, help='间隔秒'
          '--out', help='输出文件路径 (csv) ，默认: data/<symbol>_<interval>.csv'
"""
from __future__ import annotations
import os
import sys
import time
import csv
import argparse
import datetime as dt
from typing import Optional, Dict, Any, List, Tuple
import json
import ccxt
import requests

# -------------------- 工具函数 --------------------

def parse_time(value: str) -> int:
    """解析时间字符串/时间戳为毫秒。支持:
    - 纯数字(视为秒或毫秒: >=1e12 认为已是毫秒)
    - ISO 格式: 2025-09-07 00:00:00 / 2025-09-07T00:00:00 / 带时区(+08:00 / Z)
    - 若为空字符串, 返回当前时间(ms)
    """
    if value is None or value == "":
        return int(time.time() * 1000)
    value = value.strip()
    if value.isdigit():
        iv = int(value)
        return iv if iv >= 1_000_000_000_000 else iv * 1000
    # 兼容空格分隔
    value = value.replace(" ", "T")
    # 若无时区假定本地时间
    try:
        if value.endswith("Z"):
            dt_obj = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        else:
            # fromisoformat 支持 ±HH:MM
            dt_obj = dt.datetime.fromisoformat(value)
            if dt_obj.tzinfo is None:
                # 当地时间 -> 转换为 UTC 再毫秒
                dt_obj = dt_obj.astimezone()
        return int(dt_obj.timestamp() * 1000)
    except Exception as e:
        raise ValueError(f"无法解析时间: {value} ({e})")


def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

# -------------------- 核心抓取 --------------------

TIMEFRAME_MAP = {60: '1m', 180: '3m', 300: '5m', 900: '15m', 1800: '30m', 3600: '1h', 86400: '1d'}

def build_exchange():
    api_key = os.getenv('BINANCE_API_KEY')
    secret = os.getenv('BINANCE_SECRET')
    proxy = os.getenv('SOCKS5_PROXY')
    cfg = {'enableRateLimit': True, 'options': {'adjustForTimeDifference': True}}
    if api_key and secret:
        cfg['apiKey'] = api_key
        cfg['secret'] = secret
    if proxy:
        cfg['proxies'] = {'http': proxy, 'https': proxy}
        os.environ['HTTP_PROXY'] = proxy
        os.environ['HTTPS_PROXY'] = proxy
    return ccxt.binance(cfg)

# -------------------- 市值相关工具 --------------------

def fetch_btc_market_caps_range(start_ms: int, end_ms: int) -> List[Tuple[int, float]]:
    """获取指定时间范围 BTC 市值；自动按窗口循环拼接，避免分辨率被降到日级。
    逻辑:
      - 若范围 <= 89 天: 单次请求
      - 若 > 89 天: 以 89 天窗口分段请求 (略小于 90 天安全阈值) 并合并去重
    返回: [(timestamp_ms, market_cap), ...] 按时间排序
    失败: 抛异常
    """
    WINDOW_DAYS = 89  # 保证 < 90 天, 避免被 CoinGecko 自动降采样为日线
    WINDOW_MS = WINDOW_DAYS * 86400 * 1000

    def _one_range(f_ms: int, t_ms: int) -> List[Tuple[int,float]]:
        url = (
            "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range"
            f"?vs_currency=usd&from={int(f_ms/1000)}&to={int(t_ms/1000)}"
        )
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"CoinGecko range 请求失败: {r.status_code} {r.text[:120]}")
        data = r.json()
        caps = data.get('market_caps') or []
        part: List[Tuple[int,float]] = []
        for item in caps:
            if not isinstance(item, list) or len(item) < 2:
                continue
            ts, cap = item[0], item[1]
            part.append((int(ts), float(cap)))
        return part

    if end_ms <= start_ms:
        return []

    total_span = end_ms - start_ms
    if total_span <= WINDOW_MS:
        merged = _one_range(start_ms, end_ms)
        merged.sort(key=lambda x: x[0])
        return merged

    # 分段循环
    cursor = start_ms
    store: Dict[int, float] = {}
    seg_index = 0
    while cursor < end_ms:
        seg_index += 1
        seg_end = min(end_ms, cursor + WINDOW_MS)
        try:
            part = _one_range(cursor, seg_end)
        except Exception as e:
            # 可选择重试，这里先直接抛出
            raise
        for ts, cap in part:
            store[ts] = cap  # 去重覆盖
        # 打印进度（可选）
        print(
            f"[mc] segment {seg_index} "
            f"{dt.datetime.fromtimestamp(cursor/1000, dt.timezone.utc):%Y-%m-%d} -> "
            f"{dt.datetime.fromtimestamp(seg_end/1000, dt.timezone.utc):%Y-%m-%d} points={len(part)}"
        )
        # 轻量节流，避免触碰速率限制
        time.sleep(0.6)
        # 下一个窗口：避免重叠过多，+1ms 跳过已取末尾
        cursor = seg_end + 1

    out = sorted(store.items(), key=lambda x: x[0])
    return out

def lookup_market_cap(series: List[Tuple[int,float]], ts: int) -> Optional[float]:
    """在已排序 (ts, cap) 序列中找到最接近 ts 的市值。若序列为空返回 None。"""
    if not series:
        return None
    import bisect
    times = [t for t,_ in series]
    i = bisect.bisect_left(times, ts)
    if i == 0:
        return series[0][1]
    if i == len(times):
        return series[-1][1]
    before_t, before_c = series[i-1]
    after_t, after_c = series[i]
    return before_c if (ts - before_t) <= (after_t - ts) else after_c

def fetch_realtime_btc_market_cap() -> Optional[float]:
    """实时获取 BTC 市值。请求失败返回 None。"""
    url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd&include_market_cap=true"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return None
        data = r.json()
        return data.get('bitcoin', {}).get('usd_market_cap')
    except Exception:
        return None

def read_last_timestamp(out_file: str) -> Optional[int]:
    if not os.path.exists(out_file):
        return None
    last = None
    try:
        with open(out_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or line.lower().startswith('timestamp'):
                    continue
                last = line
        if last:
            return int(last.split(',')[0])
    except Exception:
        return None
    return None

def read_first_timestamp(out_file: str) -> Optional[int]:
    """读取文件中第一条数据的时间戳 (毫秒)。
    跳过 meta / 表头行。若无法读取则返回 None。
    """
    if not os.path.exists(out_file):
        return None
    try:
        with open(out_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or line.lower().startswith('timestamp'):
                    continue
                return int(line.split(',')[0])
    except Exception:
        return None
    return None

def parse_meta(line: str) -> Dict[str, str]:
    # line example: # meta symbol=BTC/USDT interval=60 marketcap=1 created=...
    parts = line.strip('# ').split()
    kv = {}
    for p in parts[1:]:
        if '=' in p:
            k,v = p.split('=',1)
            kv[k]=v
    return kv

def check_existing_file(out_file: str, symbol: str, interval_sec: int, want_market_cap: bool) -> bool:
    if not os.path.exists(out_file) or os.path.getsize(out_file)==0:
        return False
    try:
        with open(out_file,'r',encoding='utf-8') as f:
            first = f.readline()
            if first.startswith('# meta'):
                meta = parse_meta(first)
                ms = meta.get('symbol')
                mi = meta.get('interval')
                mc_flag = meta.get('marketcap','0')
                if ms == symbol and mi == str(interval_sec) and (mc_flag == ('1' if want_market_cap else '0')):
                    return True
                return False
            else:
                # 无元数据则视为不兼容
                return False
    except Exception:
        return False

def write_new_file(out_file: str, symbol: str, interval_sec: int, include_market_cap: bool):
    ensure_dir(out_file)
    with open(out_file,'w',newline='',encoding='utf-8') as f:
        f.write(f"# meta symbol={symbol} interval={interval_sec} marketcap={'1' if include_market_cap else '0'} created={int(time.time())}\n")
        w = csv.writer(f)
        header = ['timestamp','price']
        if include_market_cap:
            header.append('market_cap')
        w.writerow(header)

def prepare_output(out_file: str, symbol: str, interval_sec: int, include_market_cap: bool) -> bool:
    compatible = check_existing_file(out_file, symbol, interval_sec, include_market_cap)
    if not compatible:
        # 如果已经存在但不兼容，重命名备份
        if os.path.exists(out_file) and os.path.getsize(out_file)>0:
            bak = out_file + f".bak_{int(time.time())}"
            try:
                os.replace(out_file, bak)
                print(f"旧文件与当前参数不符，已备份为 {bak}")
            except Exception:
                pass
        write_new_file(out_file, symbol, interval_sec, include_market_cap)
        return False  # 不可续写
    return True  # 可续写

def loop_realtime(exchange, symbol: str, start_ms: int, end_ms: int, interval_sec: int, out_file: str, include_market_cap: bool):
    next_ms = start_ms
    with open(out_file, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        while next_ms <= end_ms:
            now_ms = int(time.time() * 1000)
            if now_ms < next_ms:
                time.sleep((next_ms - now_ms)/1000)
            attempt = 0
            while True:
                try:
                    ticker = exchange.fetch_ticker(symbol)
                    break
                except Exception:
                    attempt += 1
                    time.sleep(min(10, 0.5 * 2 ** attempt))
            row = [int(time.time() * 1000), ticker.get('last')]
            if include_market_cap and symbol.upper().startswith('BTC'):
                mc = fetch_realtime_btc_market_cap()
                row.append(mc)
            w.writerow(row)
            f.flush()
            next_ms += interval_sec * 1000

def fetch_historical_ohlcv(exchange, symbol: str, start_ms: int, end_ms: int, interval_sec: int, out_file: str, include_market_cap: bool):
    timeframe = TIMEFRAME_MAP[interval_sec]
    limit = 1000
    mc_lookup = None
    if include_market_cap and symbol.upper().startswith('BTC'):
        try:
            mc_lookup = fetch_btc_market_caps_range(start_ms, end_ms)
        except Exception as e:
            print(f"获取市场市值历史失败: {e}")
    with open(out_file, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        since = start_ms
        while since <= end_ms:
            attempt = 0
            while True:
                try:
                    batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
                    break
                except Exception:
                    attempt += 1
                    time.sleep(min(10, 0.5 * 2 ** attempt))
            if not batch:
                break
            for o in batch:
                ts, o_, h_, l_, c_, v_ = o
                if ts < start_ms:
                    continue
                if ts > end_ms:
                    return
                if include_market_cap and mc_lookup:
                    mc_value = lookup_market_cap(mc_lookup, ts)
                    w.writerow([ts, c_, mc_value])
                else:
                    w.writerow([ts, c_])
            f.flush()
            last_end = batch[-1][0]
            since = last_end + interval_sec * 1000 if last_end != since else since + interval_sec * 1000
            if last_end >= end_ms:
                break

def load_config(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            raise SystemExit(f"配置文件 JSON 解析失败: {e}")

# -------------------- 主流程 --------------------

def main():
    parser = argparse.ArgumentParser(description="抓取交易对市场价格 (支持配置文件)")
    parser.add_argument('--config', default='config.json', help='配置文件路径(JSON). 默认: config_prices.json')
    args = parser.parse_args()
    # 读取配置
    cfg: Dict[str, Any] = {}
    cfg = load_config(args.config)
    print(f"加载配置文件: {args.config}")

    # 合并: 命令行优先
    symbol      = cfg.get('symbol')
    start       = cfg.get('start')
    end         = cfg.get('end')
    interval_sec= int(cfg.get('interval'))
    out_file    = cfg.get('out')
    proxy       = cfg.get('proxy')
    api_key     = cfg.get('apiKey')
    secret      = cfg.get('secret')
    include_market_cap = bool(cfg.get('add_btc_market_cap'))

    # 默认参数填写
    if not out_file:
        sym_clean = symbol.replace('/', '').upper() # 规范化
        out_file = os.path.join('data', f"{sym_clean}_{interval_sec}.csv")
    if not end:
        end = ''  # parse_time 里空串=当前时间
    if proxy:
        os.environ['SOCKS5_PROXY'] = proxy
    if api_key:
        os.environ['BINANCE_API_KEY'] = api_key
    if secret:
        os.environ['BINANCE_SECRET'] = secret

    # 检查必要参数
    missing = [k for k, v in [('symbol', symbol), ('start', start), ('interval', interval_sec)] if v in (None, '')]
    if missing:
        print(f"缺少必要参数: {missing}", file=sys.stderr)
        sys.exit(1)

    # 解析时间
    try:
        start_ms = parse_time(start)
        end_ms = parse_time(end)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    ensure_dir(out_file)                                            # 创建或确保输出目录存在
    exchange = build_exchange()                                     # 创建交易所实例
    now_ms = int(time.time() * 1000)                                # 当前时间
    can_resume = prepare_output(out_file, symbol, interval_sec, include_market_cap)     # 检测元数据是否匹配，可否续写
    if can_resume:
        # 若用户要求的 start 早于现有文件最早时间，则删除并重建，避免错误认为可续写
        first_ts = read_first_timestamp(out_file)
        if first_ts is not None and start_ms < first_ts:
            print(f"请求的 start({start_ms}) 早于现有文件最早时间({first_ts})，将删除并重建文件以回溯历史数据。")
            try:
                os.remove(out_file)
            except Exception as e:
                print(f"删除旧文件失败: {e}")
            write_new_file(out_file, symbol, interval_sec, include_market_cap)
            can_resume = False
    if can_resume:
        # 仍然处于续写模式，则根据最后时间戳调整起点
        last_ts = read_last_timestamp(out_file)
        if last_ts is not None and last_ts + interval_sec * 1000 > start_ms:
            start_ms = last_ts + interval_sec * 1000                # 续写模式，从最后一个点之后继续
    can_hist = (interval_sec in TIMEFRAME_MAP) and (end_ms < now_ms or start_ms < now_ms)
    print(f"start {symbol} interval={interval_sec}s -> {out_file} resume={'yes' if can_resume else 'no'} market_cap={'on' if include_market_cap else 'off'}")

    # 如果完全在过去且支持 timeframe -> 批量历史; 否则实时循环
    if can_hist and end_ms <= now_ms:
        fetch_historical_ohlcv(exchange, symbol, start_ms, end_ms, interval_sec, out_file, include_market_cap)
    else:
        # 若开始在过去且支持 timeframe，可先补历史到当前，再转实时
        if can_hist and start_ms < now_ms:
            hist_end = min(now_ms - interval_sec * 1000, end_ms)
            if hist_end > start_ms:
                fetch_historical_ohlcv(exchange, symbol, start_ms, hist_end, interval_sec, out_file, include_market_cap)
                start_ms = hist_end + interval_sec * 1000
        # 进入实时阶段 (可能可提前结束)
        if start_ms <= end_ms:
            loop_realtime(exchange, symbol, start_ms, end_ms, interval_sec, out_file, include_market_cap)

    print('done')


if __name__ == '__main__':
    main()