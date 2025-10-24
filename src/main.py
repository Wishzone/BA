import os
import json
import ccxt
import pandas as pd

pd.set_option('display.max_columns', 5000)

CONFIG_PATH = os.getenv('PRICE_CONFIG', 'config_prices.json')

def load_credentials(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f'配置文件不存在: {path}')
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    api_key = data.get('apiKey') or data.get('api_key') or data.get('BINANCE_API_KEY')
    secret = data.get('secret') or data.get('apiSecret') or data.get('api_secret') or data.get('BINANCE_SECRET')
    proxy = data.get('proxy') or data.get('SOCKS5_PROXY')
    return api_key, secret, proxy

def build_exchange(api_key: str | None, secret: str | None, proxy: str | None):
    cfg = {'enableRateLimit': True, 'options': {'adjustForTimeDifference': True}}
    if api_key and secret:
        cfg['apiKey'] = api_key
        cfg['secret'] = secret
    if proxy:
        cfg['proxies'] = {'http': proxy, 'https': proxy}
        os.environ['HTTP_PROXY'] = proxy
        os.environ['HTTPS_PROXY'] = proxy
    return ccxt.binance(cfg)

def filter_balances(exchange):
    bal = exchange.private_get_account()
    df = pd.DataFrame(bal['balances'])
    df = df[df['asset'].isin(['BTC', 'USDT', 'BUSD', 'ETH', 'USDC'])]
    # 只显示非零
    if 'free' in df.columns:
        try:
            df = df[(df['free'].astype(float) > 0) | (df['locked'].astype(float) > 0)]
        except Exception:
            pass
    return df.reset_index(drop=True)

def main():
    api_key, secret, proxy = load_credentials(CONFIG_PATH)
    exch = build_exchange(api_key, secret, proxy)
    df = filter_balances(exch)
    print(df)

if __name__ == '__main__':
    main()





# # 指定下单币种
# symbol = 'BTCUSDT'
# # 获取当前价格
# ticker = exchange.fetch_ticker(symbol)
# # 指定下单价格
# price = 38000.0
# # 指定下单数量
# quantity = 0.001
# # 下单
# exchange.private_post_order(
#     params={
#         'symbol': symbol,
#         'quantity': quantity,
#         'price': price,
#         'side': 'BUY',
#         'type': 'LIMIT',
#         'timeInForce': 'GTC',
#         'timestamp':int(time.time() * 1000)}
# )

# # 撤单
# symbol = 'BTCUSDT'
# exchange.private_delete_openorders(
#     params={
#         'symbol': symbol,
#         'timestamp':int(time.time() * 1000)}   
# )