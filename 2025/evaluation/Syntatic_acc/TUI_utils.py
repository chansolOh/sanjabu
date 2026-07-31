#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, time, os, math, shutil, random



RESET = "\x1b[0m"
BOLD  = "\x1b[1m"
DIM   = "\x1b[2m"

FG = {
    "green": "\x1b[32m",
    "cyan":  "\x1b[36m",
    "yellow":"\x1b[33m",
    "magenta":"\x1b[35m",
    "blue":  "\x1b[34m",
    "red":   "\x1b[31m",
    "gray":  "\x1b[90m",
    "white": "\x1b[97m",
}

def term_width(default=80):
    try:
        return shutil.get_terminal_size().columns
    except:
        return default

def clr_line():
    sys.stdout.write("\x1b[2K")  # clear entire line
    sys.stdout.flush()

def move_up(n=1):
    if n>0:
        sys.stdout.write(f"\x1b[{n}A")
        sys.stdout.flush()

def move_down(n=1):
    if n>0:
        sys.stdout.write(f"\x1b[{n}B")
        sys.stdout.flush()

# ---------- 진행바 ----------
def progress_bar(current, total, width=30, color="cyan"):
    ratio = 0 if total == 0 else min(max(current/total, 0.0), 1.0)
    filled = int(width * ratio)
    empty  = width - filled
    bar = f"{FG[color]}{'█'*filled}{DIM}{'·'*empty}{RESET}"
    pct = f"{ratio*100:6.2f}%"
    return f"{bar} {pct}"

# ---------- 숫자/퍼센트 포맷 ----------
def fmt_pct(v):
    return f"{v:.6f}" if abs(v) < 1 else f"{v:.4f}"

# ---------- 한 줄 상태 렌더 ----------
def render_metric_line(name, acc_value, cur, total, bar_w):
    # 예: "inst_seg acc :  100.0 %   [progress]"
    left = f"{BOLD}{name:<14}{RESET}: {FG['yellow']}{fmt_pct(acc_value)} %{RESET}"
    right = progress_bar(cur, total, width=bar_w, color="cyan")
    # 터미널 폭에 맞춰 자르기/패딩
    w = term_width()
    msg = f"{left}   {right}"
    if len(strip_ansi(msg)) > w:
        # progress 바 폭 줄이기
        shrink = max(10, bar_w - (len(strip_ansi(msg)) - w))
        right = progress_bar(cur, total, width=shrink, color="cyan")
        msg = f"{left}   {right}"
    return msg

def strip_ansi(s: str) -> str:
    # 간단한 ANSI 제거(폭 계산용)
    import re
    return re.sub(r"\x1b\[[0-9;]*m", "", s)

# ---------- 요약 라인 ----------
def render_total_line(name, acc_value):
    return f"{BOLD}{name:<20}{RESET}: {FG['green']}{fmt_pct(acc_value)} %{RESET}"

# ---------- 데모용 처리(여기만 실제 로직으로 교체) ----------
def fake_compute_step():
    # 실제 평가 로직으로 교체하는 부분: 처리 후 정확도 갱신값 리턴
    # 예시로 ±0.05% 랜덤 변동
    return (random.random() - 0.5) * 0.1

def main():
    # 총 샘플 수(진행바 기준). 실제 처리 개수로 바꿔줘.
    total = 120

    # 실시간 누적 정확도(예시 초기값). 실제 초기값/계산으로 대체.
    acc = {
        "inst_seg acc":  100.0,
        "bbox acc":      100.0,
        "grasp acc":      93.3494858385709,
        "scene_meta acc":100.0,
    }

    # 첫 렌더(4줄)
    bar_w = 32
    lines = [
        render_metric_line("inst_seg acc",  acc["inst_seg acc"], 0, total, bar_w),
        render_metric_line("bbox acc",      acc["bbox acc"],     0, total, bar_w),
        render_metric_line("grasp acc",     acc["grasp acc"],    0, total, bar_w),
        render_metric_line("scene_meta acc",acc["scene_meta acc"],0, total, bar_w),
    ]
    for ln in lines:
        print(ln)

    # 진행 루프(데모용). 실제 처리 루프에서 각 스텝마다 갱신하면 됨.
    for i in range(1, total+1):
        # --- 여기서 실제 평가 로직을 수행하고, acc[...] 값을 갱신 ---
        # 예시: grasp만 살짝 변동, 나머지는 고정
        acc["grasp acc"] = max(0.0, min(100.0, acc["grasp acc"] + fake_compute_step()))

        # 커서를 위로 4줄 올려서 갱신
        move_up(4)
        print(render_metric_line("inst_seg acc",  acc["inst_seg acc"],  i, total, bar_w)); clr_line()
        print(render_metric_line("bbox acc",      acc["bbox acc"],      i, total, bar_w)); clr_line()
        print(render_metric_line("grasp acc",     acc["grasp acc"],     i, total, bar_w)); clr_line()
        print(render_metric_line("scene_meta acc",acc["scene_meta acc"],i, total, bar_w)); clr_line()
        sys.stdout.flush()
        time.sleep(0.02)  # 데모용 딜레이

    # 최종 집계(예시 total 값). 실제 최종 산식으로 대체.
    total_acc = {
        "inst_seg acc total":   100.0,
        "bbox acc total":       100.0,
        "grasp acc total":       98.0376266544723,
        "scene_meta acc total": 100.0,
    }

    # 공백 라인
    print()
    # 요약 출력
    print(f"{BOLD}{FG['blue']}== Final Summary =={RESET}")
    print(render_total_line("inst_seg acc total",   total_acc["inst_seg acc total"]))
    print(render_total_line("bbox acc total",       total_acc["bbox acc total"]))
    print(render_total_line("grasp acc total",      total_acc["grasp acc total"]))
    print(render_total_line("scene_meta acc total", total_acc["scene_meta acc total"]))

if __name__ == "__main__":
    main()
