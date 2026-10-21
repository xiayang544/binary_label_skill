# -*- coding: utf-8 -*-
"""LLM 打标客户端：并发 + 限速 + 重试 + 断点续跑。

结构参考 law1/LLM打标_违反法律法规_label.py（限速器 / 线程池 /
输出按 sample_id 去重续跑），输出字段对齐 data_schema.yaml 的
llm_annotation_fields：

    llm_label / llm_scene_id / llm_reason
    label_model / label_prompt_version
    llm_raw_response / llm_parse_success

接入配置（优先级：task.yaml llm_api > 环境变量 > 默认值）：
    llm_api:
      base_url: "https://<llm-gateway>/v1/openai/native"
      api_key: "..."
      gateway_names:            # 显示名 → 网关模型名（缺省取小写）
        "LongCat-Flash-Chat": "longcat-flash-chat"
"""

import json
import os
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

# openai SDK 在部分环境装在 /tmp/pylibs2（用户 site-packages 被禁用）。
# 仅当默认路径找不到 openai 时才添加该回退路径，避免其残缺依赖
# （如旧版 anyio）覆盖正式 site-packages 中的完整版本。
_FALLBACK_PATH = "/tmp/pylibs2"

try:
    from openai import OpenAI
except ImportError:
    if os.path.isdir(_FALLBACK_PATH) and _FALLBACK_PATH not in sys.path:
        sys.path.insert(0, _FALLBACK_PATH)
    from openai import OpenAI

DEFAULT_BASE_URL = "https://<llm-gateway>/v1/openai/native"
DEFAULT_API_KEY = os.environ.get("LLM_API_KEY", "")

VALID_LABELS = {"positive", "negative"}


def resolve_api(task_cfg=None):
    api_cfg = (task_cfg or {}).get("llm_api", {}) or {}
    base_url = api_cfg.get("base_url") or os.environ.get("LLM_BASE_URL") or DEFAULT_BASE_URL
    api_key = api_cfg.get("api_key") or os.environ.get("LLM_API_KEY") or DEFAULT_API_KEY
    return base_url, api_key


def resolve_gateway_name(model, task_cfg=None):
    names = ((task_cfg or {}).get("llm_api", {}) or {}).get("gateway_names", {}) or {}
    return names.get(model, model.lower())


class RateLimiter:

    def __init__(self, max_calls_per_minute):
        self.max_calls = max_calls_per_minute
        self.period = 60.0
        self.lock = threading.Lock()
        self.call_timestamps = []

    def acquire(self):
        while True:
            with self.lock:
                now = time.monotonic()
                self.call_timestamps = [t for t in self.call_timestamps if now - t < self.period]
                if len(self.call_timestamps) < self.max_calls:
                    self.call_timestamps.append(now)
                    return
                sleep_time = self.period - (now - self.call_timestamps[0]) + 0.01
            time.sleep(max(sleep_time, 0.01))


def extract_json(text):
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"无 JSON: {text[:200]}")
    return json.loads(text[start: end + 1])


def parse_label_response(content):
    """解析并校验 LLM 输出 → (label, scene_id, reason)。非法抛 ValueError 触发重试。"""
    obj = extract_json(content or "")
    label = str(obj.get("label", "")).strip().lower()
    if label not in VALID_LABELS:
        raise ValueError(f"非法 label: {label}")
    scene_id = str(obj.get("scene_id") or "").strip()
    if label == "positive" and not scene_id:
        raise ValueError("positive 缺少 scene_id")
    if label == "negative":
        scene_id = ""
    reason = str(obj.get("reason") or "").strip()
    return label, scene_id, reason


def call_llm(model, system_prompt, user_prompt, task_cfg=None,
             rate_limiter=None, max_retries=5, max_tokens=1024):
    """通用单次调用（合成 / 归因等非打标任务复用），返回原始文本。

    rate_limiter 由调用方持有，保证多任务共享同一限速额度。
    失败重试与打标一致（指数退避，429 加大）。
    """
    base_url, api_key = resolve_api(task_cfg)
    gateway = resolve_gateway_name(model, task_cfg)
    client = OpenAI(api_key=api_key, base_url=base_url)
    last_err = None
    for attempt in range(max_retries):
        try:
            if rate_limiter is not None:
                rate_limiter.acquire()
            resp = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=gateway,
                max_tokens=max_tokens,
                temperature=0.7,
                timeout=180,
            )
            content = resp.choices[0].message.content
            if content is None and getattr(resp.choices[0].message, "reasoning_content", None):
                content = resp.choices[0].message.reasoning_content
            return content or ""
        except Exception as e:
            last_err = str(e)[:200]
            sleep_s = 2 ** attempt
            if "429" in last_err:
                sleep_s = max(sleep_s, 30 * (attempt + 1))
            time.sleep(sleep_s)
    raise RuntimeError(f"LLM 调用失败 [{model}]: {last_err}")


def label_records(records, model, system_prompt, build_user_fn, out_path,
                  task_cfg=None, workers=4, rpm=20, max_retries=5, verbose=True):
    """多线程打标，输出追加写 out_path（jsonl），按 sample_id 断点续跑。

    返回本次新增的完成记录列表。失败样本也落盘（llm_parse_success=false，
    label_error 记录原因），不抛异常。
    """
    base_url, api_key = resolve_api(task_cfg)
    gateway = resolve_gateway_name(model, task_cfg)
    client = OpenAI(api_key=api_key, base_url=base_url)
    rate_limiter = RateLimiter(rpm)

    done_keys = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done_keys.add(json.loads(line).get("sample_id"))
                except Exception:
                    continue
    todo = [r for r in records if r.get("sample_id") not in done_keys]
    if verbose:
        print(f"模型 {model}（网关名 {gateway}）| 共 {len(records)} 条，"
              f"已完成 {len(done_keys)}，待打标 {len(todo)} | 并发 {workers}，限速 {rpm} rpm")
    if not todo:
        return []

    def _label_one(row):
        user_prompt = build_user_fn(row)
        last_err = None
        for attempt in range(max_retries):
            try:
                rate_limiter.acquire()
                resp = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    model=gateway,
                    max_tokens=512,
                    temperature=0.1,
                    timeout=120,
                )
                content = resp.choices[0].message.content
                if content is None and getattr(resp.choices[0].message, "reasoning_content", None):
                    content = resp.choices[0].message.reasoning_content
                label, scene_id, reason = parse_label_response(content or "")
                return dict(row), {
                    "llm_label": label, "llm_scene_id": scene_id, "llm_reason": reason,
                    "llm_raw_response": content, "llm_parse_success": True, "label_error": None,
                }
            except Exception as e:
                last_err = str(e)[:200]
                sleep_s = 2 ** attempt
                if "429" in last_err:
                    sleep_s = max(sleep_s, 30 * (attempt + 1))
                time.sleep(sleep_s)
        return dict(row), {"llm_parse_success": False, "label_error": last_err}

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    out_lock, stats, done_count = threading.Lock(), Counter(), [0]
    finished = []

    with open(out_path, "a", encoding="utf-8") as fout:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_label_one, r): r for r in todo}
            for fut in as_completed(futures):
                try:
                    row, llm_fields = fut.result()
                except Exception as e:
                    row, llm_fields = dict(futures[fut]), {"llm_parse_success": False,
                                                           "label_error": f"future异常: {str(e)[:200]}"}
                out = dict(row)
                out["label_model"] = model
                out.update({k: v for k, v in llm_fields.items() if v is not None or k == "label_error"})
                with out_lock:
                    fout.write(json.dumps(out, ensure_ascii=False, default=str) + "\n")
                    fout.flush()
                    finished.append(out)
                    if out.get("llm_parse_success"):
                        stats["done"] += 1
                        stats[f"label:{out['llm_label']}"] += 1
                    else:
                        stats["error"] += 1
                    done_count[0] += 1
                    if verbose and (done_count[0] % 50 == 0 or done_count[0] == len(todo)):
                        print(f"[{done_count[0]}/{len(todo)}] " +
                              " ".join(f"{k}={v}" for k, v in sorted(stats.items())), flush=True)

    if verbose:
        print("==== LLM 打标统计 ====")
        for k, v in sorted(stats.items()):
            print(f"  {k}: {v}")
    return finished
