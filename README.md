# BA 价格分析流程

项目实现了一条轻量级的研究流程，用于抓取币安现货数据、训练神经网络预测器，并基于预测结果进行回测和可视化分析。

- **数据获取**：`src/fetch.py` 通过 CCXT 与 CoinGecko 下载历史价格（可选附带 BTC 市值），写入 `data/` 目录下的 CSV。
- **预测算法**：`src/MB_train.py`（Mamba模型）与 `src/TS_train.py`（Transformer模型）读取 CSV，训练 PyTorch 模型并将权重保存到 `models/`，该算法只向后预测一个interval，为了适配回测代码，请按照该输入输出标准编写别的算法文件。
- **回测文件**：`sim/backtest.py` 调用预测算法进行回测，并可视化结果。
- **模拟与报告**：`sim/compare.py` 对比实际价格与预测结果，并在扣除手续费后计算“上帝收益”乘数。

所有脚本共用根目录下的 `config.json`，统一管理运行参数，方便在一个地方修改交易对、周期等配置。

## 仓库结构

```
data/        # 原始价格 CSV，命名为 <SYMBOL><INTERVAL>.csv
models/      # 训练得到的 PyTorch 模型
sim/         # 回测与可视化脚本
extra/       # 用于确定代理IP地址
src/         # 数据抓取与训练脚本
requirements.txt
config.json  # 全局运行配置
```

## 配置文件说明

`config.json` 集中管理运行参数，常用字段如下：

| 字段 | 说明 |
| --- | --- |
| `symbol` | 交易对，例如 `BTC/USDT`。|
| `start` | 抓取起始时间，可用 ISO 时间或毫秒时间戳。|
| `interval` | K 线周期，单位秒，支持 60/180/300/900/1800/3600/86400。|
| `fee` | 回测使用的单笔手续费率（如 `0.001`）。|
| `proxy` | 可选 SOCKS5 代理，例如 `socks5h://127.0.0.1:7890`。|
| `apiKey` / `secret` | 币安 API 凭证，抓取脚本需要认证请求时使用。留空则仅用公开数据。|
| `add_btc_market_cap` | 为 `true` 时追加 BTC 市值列。|
| `method` | 预测结果前缀（如 `MB`, `TS`），`sim/compare.py` 会据此寻找 `MB_<source>.csv` 等文件。|
| `TRate` | 回测脚本使用的训练集与回测集比例。|

请务必妥善保管 API Key

## 注意事项

- 当fetch的时间跨度超过 89 天时，脚本会分段调用 CoinGecko 接口，避免被降采样到日线。
- 修改训练参数请直接修改训练算法本身的窗口、批量大小等参数。
- `sim/compare.py` 仅对满足 `ratio >= 1 + fee` 的记录乘积，并在乘积前先扣除手续费，得到更保守的上帝收益估计。
- 请先运行 `src/fetch.py`，以获取基本市场数据。
