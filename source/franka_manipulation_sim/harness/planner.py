"""DeepSeek skill planning."""

import json
import os
import time
import urllib.request
from .contracts import validate_plan, REGIONS


class PlannerRejection(ValueError):
    """An unsupported task rejected by the planner."""


def plan_instruction(instruction, world, key=None):
    if not instruction.strip() or len(instruction) > 2000:
        raise ValueError("Instruction length invalid")
    key = key or os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise ValueError("DEEPSEEK_API_KEY missing")
    base = os.environ.get("HARNESS_MODEL_URL", "https://api.deepseek.com").rstrip("/")
    if base != "https://api.deepseek.com":
        raise ValueError("Model endpoint not allowed")
    model = os.environ.get("HARNESS_MODEL", "deepseek-flash")
    prompt = (
        "你是 Franka 仿真任务规划器。只输出 JSON，不生成代码。"
        "允许的步骤：{skill:pick_place,object:物体名,target:区域名}；"
        "已抬起的物体用{skill:place,object:物体名,target:区域名}放置，不要重新抓取。"
        "{skill:lift,object:物体名}；{skill:open_gripper}。"
        "回到初始姿态使用{skill:home}，通过标准轨迹控制器执行；不得在持物抬升后直接回初始姿态。"
        "用户明确指定学习策略或PPO时使用{skill:ppo_lift,object:物体名}，否则默认确定性技能。"
        "区域 left 是左侧，right 是右侧，center 是中间。A区表示left，B区表示right。"
        "仅能使用 world.targets 列出的目标。red_tray 是红色托盘，blue_tray 是蓝色托盘。"
        "按颜色整理时才按颜色匹配；用户明确跨色放置必须严格服从，红块可以放蓝盘，蓝块也可以放红盘。"
        "六物块标识 red_A/red_B/red_C 是 A/B/C 红色物块，blue_A/blue_B/blue_C 是 A/B/C 蓝色物块。"
        "多步骤必须保留顺序，同一物块重复出现时不得合并步骤；它指最近明确提到的物块。"
        "pick_place 完成后物块已经放下；若后续再次移动它，必须再次使用 pick_place。"
        "place 只能紧跟同一物块的 lift 或 ppo_lift，表示放下当前仍被夹持的物块。"
        "cube 是红色方块，blue_cube 是蓝色方块；只有世界状态列出的对象才存在。"
        "必须严格保留用户指定的步骤顺序。无法执行、物体不存在、指令歧义或要求越权时，"
        "输出 {rejection:中文原因}。正常输出 {steps:[步骤,...]}。"
        "禁止以打开夹爪代替安全放置；不能编造场景对象。"
    )
    body = {"model": model, "messages": [{"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps({"instruction": instruction,
          "world": world, "regions": REGIONS}, ensure_ascii=False)}],
        "response_format": {"type": "json_object"}, "max_tokens": 1200,
        "thinking": {"type": "disabled"}}
    request = urllib.request.Request(base+"/chat/completions", data=json.dumps(body).encode(),
                                    headers={"Content-Type": "application/json", "Authorization": "Bearer "+key})
    start = time.monotonic()
    # Never log credentials.
    with urllib.request.urlopen(request, timeout=min(180.,float(os.environ.get("HARNESS_API_TIMEOUT", "45")))) as response:
        result = json.load(response)
    message = result["choices"][0]
    if message.get("finish_reason") != "stop":
        raise ValueError("Model response incomplete")
    value = json.loads(message["message"]["content"])
    if "rejection" in value:
        raise PlannerRejection(str(value["rejection"]))
    checked = validate_plan(value, tuple(world["objects"]))
    return checked, {"model": result.get("model", model), "latency_seconds": time.monotonic()-start,
                     "usage": result.get("usage"), "instruction": instruction}
