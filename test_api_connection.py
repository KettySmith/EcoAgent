"""独立检查 EcoAgent 模型 API；不加载向量库、不访问数据库。

运行：python test_api_connection.py
自定义 CA：python test_api_connection.py --ca-file /path/to/company-ca.pem
仅检查 TLS（不发送密钥、不调用模型）：python test_api_connection.py --tls-only
"""

import argparse
import ast
import asyncio
import os
from pathlib import Path
import ssl
from urllib.parse import urlsplit

import httpx
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent


def default_model():
    # 读取配置字面量，避免导入应用时加载模型、数据库等依赖。
    tree = ast.parse((ROOT / "src/eco_agent/context.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "model" and isinstance(node.value, ast.Call):
                for keyword in node.value.keywords:
                    if keyword.arg == "default":
                        return ast.literal_eval(keyword.value)
    raise ValueError("无法从 context.py 读取默认模型，请用 --model 指定")


def error_kind(exc):
    parts = []
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        parts.append(str(exc))
        exc = exc.__cause__ or exc.__context__
    message = " ".join(parts).lower()
    if "certificate_verify_failed" in message or "certificate verify failed" in message:
        return "TLS 证书验证失败（可能是服务端或中间代理证书不受信任）"
    if "name or service not known" in message or "getaddrinfo" in message:
        return "DNS 解析失败"
    if "timeout" in message or "timed out" in message:
        return "连接或响应超时"
    return "网络连接失败；检查服务监听、端口、代理和防火墙"


async def probe(label, trust_env, base_url, model, key, args):
    print(f"\n=== {label} ===", flush=True)
    # 自定义 CA 追加到系统信任库；始终启用证书校验。
    verify = True
    if args.ca_file:
        verify = ssl.create_default_context()
        verify.load_verify_locations(args.ca_file)
    async with httpx.AsyncClient(
        trust_env=trust_env, verify=verify,
        timeout=httpx.Timeout(args.timeout), follow_redirects=False,
    ) as client:
        try:
            # 不附带认证；即使返回 401/404/405，也说明 HTTPS 已连接成功。
            response = await client.get(base_url + "/models")
            print(f"连接检查：HTTP {response.status_code}，已收到 HTTP 响应")
        except httpx.RequestError as exc:
            print(f"连接检查失败：{type(exc).__name__}；{error_kind(exc)}")
            return "connection_failed"

        if args.tls_only:
            return "connected"
        if not key:
            print("未设置 OPENAI_API_KEY，跳过模型请求。")
            return "missing_key"
        try:
            response = await client.post(
                base_url + "/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={"model": model, "messages": [
                    {"role": "user", "content": "Reply only with OK."}
                ], "max_tokens": 16, "stream": False},
            )
        except httpx.RequestError as exc:
            print(f"模型请求失败：{type(exc).__name__}；{error_kind(exc)}")
            return "connection_failed"

        status = response.status_code
        print(f"模型请求：HTTP {status}")
        if 200 <= status < 300:
            try:
                data = response.json()
                message = data["choices"][0]["message"]
                if not isinstance(message, dict):
                    raise ValueError("invalid message")
            except (ValueError, KeyError, IndexError, TypeError):
                print("响应不是预期的 Chat Completions JSON，请检查 API 网关和接口兼容性。")
                return "invalid_response"
            print("成功：收到 Chat Completions 格式的模型响应。")
            return "ok"
        reasons = {
            400: "请求参数或模型不兼容（包括 max_tokens 参数）",
            401: "密钥无效或认证方式不匹配",
            403: "访问权限不足或网关拒绝",
            404: "API 路径或模型不存在",
            429: "限流或额度不足",
        }
        print("检查方向：" + reasons.get(status, "服务端/网关异常" if status >= 500 else "接口响应异常"))
        print("为避免泄露敏感信息，不输出密钥、请求头或原始响应正文。")
        return f"http_{status}"


async def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--model", help="覆盖模型名称，支持 provider/model 或 model")
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--ca-file", help="额外信任的 PEM 格式 CA 文件")
    parser.add_argument("--tls-only", action="store_true")
    args = parser.parse_args()
    load_dotenv(args.env_file, override=False)
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    parsed = urlsplit(base_url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        parser.error("OPENAI_BASE_URL 必须是无认证信息、查询参数和 fragment 的 HTTP(S) URL")
    configured = args.model or os.environ.get("MODEL") or default_model()
    model = configured.split("/", 1)[-1]
    key = os.environ.get("OPENAI_API_KEY", "")
    print(f"环境文件：{args.env_file}（已有环境变量优先）")
    print(f"API 地址：{base_url}\n模型：{model}\n密钥：{'已设置' if key else '未设置'}")
    names = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "all_proxy", "no_proxy", "SSL_CERT_FILE", "SSL_CERT_DIR")
    print("已设置的代理/证书变量：" + (", ".join(n for n in names if os.environ.get(n)) or "无"))
    if not args.tls_only:
        print("将分别测试两种连接方式，每种最多调用模型一次，可能产生少量费用。")
    normal = await probe("默认环境（读取代理与证书环境变量）", True, base_url, model, key, args)
    direct = await probe("直连（忽略代理与证书环境变量）", False, base_url, model, key, args)
    print(f"\n结果：默认环境={normal}；直连={direct}")
    if normal == "connection_failed" and direct in ("ok", "connected"):
        print("直连成功，优先检查代理或证书环境变量。")
    elif direct == "connection_failed" and normal in ("ok", "connected"):
        print("默认环境成功，网络可能需要代理或环境配置的 CA。")
    elif normal == direct == "connection_failed":
        print("两种方式均失败；结合上方错误检查服务证书、网络或透明网关，不能仅凭此认定 API 服务宕机。")
    return 0 if normal in ("ok", "connected") else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
