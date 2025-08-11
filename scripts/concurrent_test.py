#!/usr/bin/env python3
import asyncio
import json
import time
import uuid
import argparse
from statistics import mean
from typing import List, Tuple, Optional
import aiohttp
from Cryptodome.PublicKey import RSA
from Cryptodome.Cipher import PKCS1_v1_5 as Cipher_pkcs1_v1_5
import base64
import os
def encrypt_password(plain: str, pubkey_path: str) -> str:
    with open(pubkey_path, "rb") as f:
        rsa_key = RSA.importKey(f.read(), "Welcome")
    cipher = Cipher_pkcs1_v1_5.new(rsa_key)
    b64_pw = base64.b64encode(plain.encode("utf-8")).decode("utf-8")
    enc = cipher.encrypt(b64_pw.encode("utf-8"))
    return base64.b64encode(enc).decode("utf-8")

async def login(session: aiohttp.ClientSession, base_url: str, email: str, password: str, pubkey_path: str) -> str:
    enc_pwd = encrypt_password(password, pubkey_path)
    url = f"{base_url}/v1/user/login"
    async with session.post(url, json={"email": email, "password": enc_pwd}) as resp:
        if resp.status != 200:
            text = await resp.text()
            raise RuntimeError(f"login failed: {resp.status} {text}")
        # token 在响应头 Authorization 中
        token = resp.headers.get("Authorization")
        if not token:
            raise RuntimeError("no Authorization header in login response")
        return token

async def pick_dialog_id(session: aiohttp.ClientSession, base_url: str, auth: str, prefer_dialog_id: Optional[str]=None) -> str:
    if prefer_dialog_id:
        return prefer_dialog_id
    url = f"{base_url}/v1/dialog/list"
    async with session.get(url, headers={"Authorization": auth}) as resp:
        if resp.status != 200:
            text = await resp.text()
            raise RuntimeError(f"list dialogs failed: {resp.status} {text}")
        data = await resp.json()
        if data.get("code") != 0 or not data.get("data"):
            raise RuntimeError("no accessible dialog found; please create one first")
        return data["data"][0]["id"]

async def create_conversation(session: aiohttp.ClientSession, base_url: str, auth: str, dialog_id: str) -> str:
    conv_id = uuid.uuid4().hex
    url = f"{base_url}/v1/conversation/set"
    payload = {
        "conversation_id": conv_id,
        "dialog_id": dialog_id,
        "is_new": True,
        "name": "bench-conv"
    }
    async with session.post(url, headers={"Authorization": auth}, json=payload) as resp:
        if resp.status != 200:
            text = await resp.text()
            raise RuntimeError(f"create conversation failed: {resp.status} {text}")
        data = await resp.json()
        if data.get("code") != 0:
            raise RuntimeError(f"create conversation error: {data}")
        return conv_id

def perc(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    k = max(0, min(len(values) - 1, int(round((p / 100.0) * (len(values) - 1)))))
    return sorted(values)[k]

async def do_completion_sse(session: aiohttp.ClientSession, base_url: str, auth: str, conv_id: str, question: str) -> Tuple[float, float]:
    url = f"{base_url}/v1/conversation/completion"
    body = {
        "conversation_id": conv_id,
        "messages": [
            {"role": "user", "content": question, "id": uuid.uuid4().hex}
        ],
        "stream": True,
        # 你可在此追加 LLM 相关参数，如 temperature、top_p 等
    }
    t0 = time.perf_counter()
    async with session.post(url, headers={"Authorization": auth}, json=body) as resp:
        if resp.status != 200:
            text = await resp.text()
            raise RuntimeError(f"completion(sse) failed: {resp.status} {text}")
        ttfb = None
        async for raw in resp.content:
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            if not line.startswith("data:"):
                continue
            if ttfb is None:
                ttfb = time.perf_counter() - t0  # 首个 data 行到达时间
            try:
                payload = json.loads(line[5:].strip())
            except Exception:
                continue
            # 结束信号：{"code":0,"data":True}
            if payload.get("code") == 0 and payload.get("data") is True:
                total = time.perf_counter() - t0
                return (ttfb or 0.0, total)
        # 流意外结束
        raise RuntimeError("SSE stream closed unexpectedly")

async def do_completion_nonstream(session: aiohttp.ClientSession, base_url: str, auth: str, conv_id: str, question: str) -> Tuple[float, float]:
    url = f"{base_url}/v1/conversation/completion"
    body = {
        "conversation_id": conv_id,
        "messages": [
            {"role": "user", "content": question, "id": uuid.uuid4().hex}
        ],
        "stream": False
    }
    t0 = time.perf_counter()
    async with session.post(url, headers={"Authorization": auth}, json=body) as resp:
        if resp.status != 200:
            text = await resp.text()
            raise RuntimeError(f"completion(non-stream) failed: {resp.status} {text}")
        ttfb = time.perf_counter() - t0
        data = await resp.json()
        if data.get("code") != 0:
            raise RuntimeError(f"completion(non-stream) error: {data}")
        total = time.perf_counter() - t0
        return (ttfb, total)

async def worker(idx: int, base_url: str, auth: str, dialog_id: str, question: str, stream: bool, results_ttfb: List[float], results_total: List[float], sem: asyncio.Semaphore):
    async with sem:
        async with aiohttp.ClientSession() as s2:
            conv_id = await create_conversation(s2, base_url, auth, dialog_id)
            if stream:
                ttfb, total = await do_completion_sse(s2, base_url, auth, conv_id, question)
            else:
                ttfb, total = await do_completion_nonstream(s2, base_url, auth, conv_id, question)
            results_ttfb.append(ttfb)
            results_total.append(total)

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:9380", help="服务基础地址")
    parser.add_argument("--email", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--pubkey", default=os.path.abspath("conf/public.pem"), help="RSA 公钥路径")
    parser.add_argument("--dialog-id", default=None, help="指定 dialog_id；不指定则自动选取可访问的一个")
    parser.add_argument("--concurrency", type=int, default=10, help="并发协程数")
    parser.add_argument("--requests", type=int, default=50, help="总请求数（每个请求会创建一个 conversation）")
    parser.add_argument("--stream", action="store_true", help="使用 SSE 流式（推荐）")
    parser.add_argument("--question", default="请简要介绍一下RAGFlow是什么？")
    args = parser.parse_args()

    async with aiohttp.ClientSession() as session:
        auth = await login(session, args.base_url, args.email, args.password, args.pubkey)
        dialog_id = await pick_dialog_id(session, args.base_url, auth, args.dialog_id)

    results_ttfb: List[float] = []
    results_total: List[float] = []
    sem = asyncio.Semaphore(args.concurrency)
    tasks = []
    t_start = time.perf_counter()
    for i in range(args.requests):
        tasks.append(worker(i, args.base_url, auth, dialog_id, args.question, args.stream, results_ttfb, results_total, sem))
    await asyncio.gather(*tasks)
    t_elapsed = time.perf_counter() - t_start

    def fmt(stats: List[float]) -> str:
        if not stats:
            return "n/a"
        return (
            f"count={len(stats)} "
            f"avg={mean(stats)*1000:.1f}ms "
            f"p50={perc(stats,50)*1000:.1f}ms "
            f"p90={perc(stats,90)*1000:.1f}ms "
            f"p95={perc(stats,95)*1000:.1f}ms "
            f"p99={perc(stats,99)*1000:.1f}ms "
            f"max={max(stats)*1000:.1f}ms"
        )

    print(f"Done {args.requests} requests in {t_elapsed:.2f}s, concurrency={args.concurrency}")
    print(f"Throughput ≈ {args.requests / t_elapsed:.2f} req/s")
    print(f"TTFB:  {fmt(results_ttfb)}")
    print(f"Total: {fmt(results_total)}")

if __name__ == "__main__":
    asyncio.run(main())