# -*- coding: utf-8 -*-
"""配置驱动的 Regex 初筛引擎。

规则保存在 v_k/standard/regex/regex_config.yaml，格式：

    regex_version: "v_0"
    rules:
      - rule_id: "R001"
        name: "教唆超速"
        scene_ids: ["scene_001"]     # 命中时归属的子场景
        match:
          all:  ["超速", "闯红灯"]    # 必须全部命中的模式（AND）
          any:  ["没事|没关系|放心"]  # 至少命中一个的模式（OR，可省略）
          none: ["不允许|禁止"]      # 不得命中的模式（NOT，可省略）
        scope: "current_turn"        # current_turn / context / both，默认 current_turn
        case_sensitive: false

通用执行逻辑与本文件绑定；某轮无法用配置描述时，允许在
standard/regex/regex_prefilter.py 中存放定制逻辑（见 SKILL.md §11）。

回归测试：regex_cases.jsonl 每行
    {"case_id": "...", "current_turn": "...", "context": "...",
     "expect_hit": true, "expect_rule_ids": ["R001"], "note": "..."}
"""

import re

import yaml


class RegexEngine:
    def __init__(self, config):
        self.config = config or {}
        self.regex_version = self.config.get("regex_version", "")
        self.rules = []
        for rule in self.config.get("rules", []) or []:
            compiled = self._compile_rule(rule)
            if compiled:
                self.rules.append(compiled)

    @staticmethod
    def _compile_patterns(patterns, flags):
        return [re.compile(p, flags) for p in (patterns or [])]

    def _compile_rule(self, rule):
        match = rule.get("match", {}) or {}
        flags = 0 if rule.get("case_sensitive") else re.IGNORECASE
        try:
            return {
                "rule_id": str(rule.get("rule_id", "")),
                "name": rule.get("name", ""),
                "scene_ids": list(rule.get("scene_ids", []) or []),
                "scope": rule.get("scope", "current_turn"),
                "all": self._compile_patterns(match.get("all"), flags),
                "any": self._compile_patterns(match.get("any"), flags),
                "none": self._compile_patterns(match.get("none"), flags),
            }
        except re.error as e:
            raise ValueError(f"规则 {rule.get('rule_id')} 正则编译失败: {e}")

    @staticmethod
    def _scope_text(scope, current_turn, context):
        if scope == "context":
            return context or ""
        if scope == "both":
            return (current_turn or "") + "\n" + (context or "")
        return current_turn or ""

    def match(self, current_turn, context):
        """返回 {"hit": bool, "rule_ids": [...], "scene_ids": [...], "names": [...]}。"""
        hit_rule_ids, hit_scenes, hit_names = [], [], []
        for rule in self.rules:
            text = self._scope_text(rule["scope"], current_turn, context)
            if not text:
                continue
            if not all(p.search(text) for p in rule["all"]):
                continue
            if rule["any"] and not any(p.search(text) for p in rule["any"]):
                continue
            if any(p.search(text) for p in rule["none"]):
                continue
            hit_rule_ids.append(rule["rule_id"])
            hit_names.append(rule["name"])
            for s in rule["scene_ids"]:
                if s not in hit_scenes:
                    hit_scenes.append(s)
        return {"hit": bool(hit_rule_ids), "rule_ids": hit_rule_ids,
                "scene_ids": hit_scenes, "names": hit_names}


def load_engine(config_path):
    with open(config_path, encoding="utf-8") as f:
        return RegexEngine(yaml.safe_load(f) or {})


def run_regression(engine, cases):
    """对 regex_cases.jsonl 执行回归测试，返回报告 dict。

    判定：expect_hit 与实际 hit 一致；若给出 expect_rule_ids，
    还要求实际命中规则集合与预期一致。
    """
    total, passed, failures = 0, 0, []
    for case in cases:
        total += 1
        result = engine.match(case.get("current_turn", ""), case.get("context", ""))
        ok = result["hit"] == bool(case.get("expect_hit"))
        expect_rules = case.get("expect_rule_ids")
        if ok and expect_rules:
            ok = sorted(result["rule_ids"]) == sorted(expect_rules)
        if ok:
            passed += 1
        else:
            failures.append({
                "case_id": case.get("case_id"),
                "note": case.get("note", ""),
                "expect_hit": case.get("expect_hit"),
                "expect_rule_ids": expect_rules,
                "actual_hit": result["hit"],
                "actual_rule_ids": result["rule_ids"],
            })
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round(passed / total, 4) if total else None,
        "failures": failures,
    }
