import os
import sys
import json
import requests

PROXY = "socks5h://127.0.0.1:7890"
proxies = {
    "http": PROXY,
    "https": PROXY,
}

TEST_URLS = [
    "https://api.ipify.org?format=json",
    "https://ifconfig.me/all.json",
    "https://ipinfo.io/json",
]

def get_ip():
    for url in TEST_URLS:
        try:
            r = requests.get(url, proxies=proxies, timeout=10)
            r.raise_for_status()
            data = r.json() if "json" in r.headers.get("Content-Type", "").lower() else {"raw": r.text.strip()}
            # Try common fields
            ip = data.get("ip") or data.get("ip_addr") or data.get("ip_address") or data.get("ip_addr_v4") or data.get("raw")
            if not ip and isinstance(data, dict):
                # ifconfig.me/all.json contains "ip_addr"
                ip = data.get("ip_addr")
            if ip:
                return {"service": url, "ip": ip, "raw": data}
        except Exception as e:
            last_err = str(e)
    raise RuntimeError(f"Failed to detect IP via proxy {PROXY}. Last error: {last_err}")

if __name__ == "__main__":
    try:
        result = get_ip()
        print(json.dumps({"proxy": PROXY, **result}, ensure_ascii=False))
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
