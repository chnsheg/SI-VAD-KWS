from __future__ import annotations

import math
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


W, H = 1800, 5100
M = 28
BLUE = "#0B3A8D"
BLUE2 = "#1E5BB8"
LIGHT_BLUE = "#EAF3FF"
GREEN = "#16834A"
LIGHT_GREEN = "#ECFFF3"
ORANGE = "#E5791A"
LIGHT_ORANGE = "#FFF5E8"
PURPLE = "#6C43B8"
LIGHT_PURPLE = "#F4EFFF"
RED = "#B3261E"
GRAY = "#56616F"
LIGHT_GRAY = "#F7F9FC"
INK = "#102033"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        r"C:\Windows\Fonts\NotoSansSC-VF.ttf",
        r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


F_TITLE = font(52, True)
F_H1 = font(32, True)
F_H2 = font(26, True)
F_BODY = font(22)
F_SMALL = font(18)
F_TINY = font(15)
F_MONO = font(18)


def text_width(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont) -> float:
    return draw.textlength(text, font=fnt)


def wrap_text(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont, max_width: int) -> list[str]:
    lines: list[str] = []
    for raw_line in str(text).split("\n"):
        line = ""
        for ch in raw_line:
            if ch == " " and not line:
                continue
            trial = line + ch
            if text_width(draw, trial, fnt) <= max_width or not line:
                line = trial
            else:
                lines.append(line.rstrip())
                line = ch.lstrip()
        if line:
            lines.append(line.rstrip())
    return lines or [""]


def rounded(draw: ImageDraw.ImageDraw, box, fill, outline=BLUE, width=3, radius=14):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def label(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, fill=INK, fnt=F_BODY, max_width: int | None = None, line_gap: int = 5):
    if max_width is None:
        draw.text((x, y), text, fill=fill, font=fnt)
        return y + fnt.size + line_gap
    cur = y
    for line in wrap_text(draw, text, fnt, max_width):
        draw.text((x, cur), line, fill=fill, font=fnt)
        cur += fnt.size + line_gap
    return cur


def section(draw: ImageDraw.ImageDraw, y: int, title: str, height: int) -> tuple[int, int, int, int]:
    box = (M, y, W - M, y + height)
    rounded(draw, box, "white", BLUE, 4, 16)
    draw.rounded_rectangle((M + 10, y + 10, M + 560, y + 54), radius=12, fill=BLUE, outline=BLUE)
    draw.text((M + 28, y + 16), title, fill="white", font=F_H2)
    return box


def arrow(draw: ImageDraw.ImageDraw, start, end, color=BLUE, width=5):
    x1, y1 = start
    x2, y2 = end
    draw.line((x1, y1, x2, y2), fill=color, width=width)
    ang = math.atan2(y2 - y1, x2 - x1)
    size = 16
    pts = [
        (x2, y2),
        (x2 - size * math.cos(ang - math.pi / 6), y2 - size * math.sin(ang - math.pi / 6)),
        (x2 - size * math.cos(ang + math.pi / 6), y2 - size * math.sin(ang + math.pi / 6)),
    ]
    draw.polygon(pts, fill=color)


def card(draw: ImageDraw.ImageDraw, x, y, w, h, title, body="", fill=LIGHT_BLUE, outline=BLUE, title_fill=BLUE):
    rounded(draw, (x, y, x + w, y + h), fill, outline, 3, 14)
    draw.text((x + 18, y + 14), title, fill=title_fill, font=F_H2)
    if body:
        label(draw, x + 18, y + 58, body, fnt=F_SMALL, max_width=w - 36, line_gap=5)


def mini_wave(draw: ImageDraw.ImageDraw, x, y, w, h, color=BLUE2):
    mid = y + h // 2
    pts = []
    for i in range(w):
        t = i / max(1, w - 1)
        amp = math.sin(t * math.pi * 12) * (0.4 + 0.6 * math.sin(t * math.pi) ** 2)
        pts.append((x + i, mid - int(amp * h * 0.38)))
    draw.line(pts, fill=color, width=3)
    draw.line((x, mid, x + w, mid), fill="#AAB7CF", width=1)


def grid(draw: ImageDraw.ImageDraw, x, y, cols, rows, cw, ch, fill="#B8D5FF", outline="#4879C8"):
    for r in range(rows):
        for c in range(cols):
            shade = (r * 11 + c * 7) % 45
            color = fill
            if fill.startswith("#"):
                base = tuple(int(fill[i : i + 2], 16) for i in (1, 3, 5))
                color = tuple(max(0, min(255, v - shade)) for v in base)
            draw.rectangle((x + c * cw, y + r * ch, x + (c + 1) * cw, y + (r + 1) * ch), fill=color, outline=outline, width=1)


def cube(draw: ImageDraw.ImageDraw, x, y, w, h, d, fill="#C7DAFF", outline=BLUE):
    front = [(x, y + d), (x + w, y + d), (x + w, y + h + d), (x, y + h + d)]
    top = [(x, y + d), (x + d, y), (x + w + d, y), (x + w, y + d)]
    side = [(x + w, y + d), (x + w + d, y), (x + w + d, y + h), (x + w, y + h + d)]
    draw.polygon(top, fill="#E7F0FF", outline=outline)
    draw.polygon(side, fill="#AFC8F6", outline=outline)
    draw.polygon(front, fill=fill, outline=outline)


def param_counts():
    # Default model: C24x5_H64, input feature=10.
    block1 = 1 * 15 + 1 * 24 + 2 * 1 + 2 * 24
    block = 24 * 15 + 24 * 24 + 2 * 24 + 2 * 24
    cnn = block1 + 4 * block
    gru = 3 * 64 * 24 + 3 * 64 * 64 + 2 * 3 * 64
    fc = 64 * 2 + 2
    total = cnn + gru + fc
    mac_window = 1_751_360
    mac_frame = mac_window / 32
    return block1, block, cnn, gru, fc, total, mac_window, mac_frame


def draw_diagram(out_path: Path):
    img = Image.new("RGB", (W, H), "#FFFFFF")
    draw = ImageDraw.Draw(img)

    # Title
    draw.rounded_rectangle((M, 16, W - M, 92), radius=18, fill="#F2F7FF", outline=BLUE, width=4)
    draw.text((W // 2, 50), "当前推理流程：Streaming MFCC 前端 + CRNN 主干结构详解", fill=BLUE, font=F_TITLE, anchor="mm")
    draw.text((W - 480, 68), "默认配置：C24x5_H64，16 kHz，10×32 MFCC", fill=GRAY, font=F_SMALL)

    # 1. Input and streaming schedule
    y = 112
    section(draw, y, "1. 输入音频与流式调度", 470)
    xs = [55, 370, 685, 1000, 1315]
    w = 260
    h = 280
    card(draw, xs[0], y + 90, w, h, "原始音频 / waveform", "Mobvoi 二分类\npositive / negative\n输入: [B, 1, 16000]\n1 second", LIGHT_BLUE)
    mini_wave(draw, xs[0] + 35, y + 210, 190, 80)
    card(draw, xs[1], y + 90, w, h, "采样与分块", "sample_rate = 16000\nwindow = 512 samples\nhop = 512 samples\n每 32 ms 到来一块", LIGHT_BLUE)
    grid(draw, xs[1] + 40, y + 220, 8, 2, 22, 18, "#BBD6FF")
    card(draw, xs[2], y + 90, w, h, "pre-emphasis", "y[t] = x[t] - 0.97 x[t-1]\n硬件流式需要保存\nprev_sample 状态", LIGHT_GREEN, GREEN, GREEN)
    mini_wave(draw, xs[2] + 38, y + 220, 185, 70, GREEN)
    card(draw, xs[3], y + 90, w, h, "StreamingMFCC 状态", "history = win - hop = 0\npending: 非整 hop 残留\nflush_tail=True\n1 秒输出 ceil(16000/512)=32 帧", LIGHT_ORANGE, ORANGE, ORANGE)
    grid(draw, xs[3] + 40, y + 230, 8, 3, 20, 16, "#FFD6AD", "#C76B1E")
    card(draw, xs[4], y + 90, w, h, "输出节拍", "每个 hop 产生 1 帧 MFCC\nframe rate = 31.25 Hz\nT = 32 frames / 1s", LIGHT_PURPLE, PURPLE, PURPLE)
    grid(draw, xs[4] + 45, y + 215, 8, 4, 20, 16, "#D9C8FF", "#7355B8")
    for i in range(4):
        arrow(draw, (xs[i] + w + 12, y + 230), (xs[i + 1] - 12, y + 230))

    # 2. Streaming MFCC frontend
    y = 610
    section(draw, y, "2. StreamingMFCC 前端：逐 hop 输出 MFCC 帧", 900)
    card(draw, 55, y + 90, 290, 250, "A. causal framing", "输入新 chunk: [B, 512]\n与 pending 拼接\n每满 hop_length=512\n形成一个 frame\nwin_length=512", LIGHT_GREEN, GREEN, GREEN)
    grid(draw, 100, y + 230, 8, 2, 22, 18, "#BFF0D1", "#3C925B")
    card(draw, 390, y + 90, 270, 250, "B. Hann window", "window = Hann(512)\n逐点乘法\n不使用 center=True\n不需要未来采样点", LIGHT_BLUE)
    mini_wave(draw, 430, y + 230, 190, 65)
    card(draw, 705, y + 90, 270, 250, "C. FFT / 功率谱", "rFFT n_fft=512\n频点数 = 257\npower = real² + imag²\n形状: [B, 257]", LIGHT_ORANGE, ORANGE, ORANGE)
    grid(draw, 745, y + 225, 9, 5, 18, 14, "#FFD4A3", "#C66C1E")
    card(draw, 1020, y + 90, 270, 250, "D. Mel filterbank", "40 triangular filters\nf_min = 20 Hz\nf_max = 8000 Hz\n输出: [B, 40]", LIGHT_GREEN, GREEN, GREEN)
    for i in range(7):
        x0 = 1060 + i * 27
        draw.line((x0, y + 275, x0 + 24, y + 210, x0 + 48, y + 275), fill=GREEN, width=2)
    card(draw, 1335, y + 90, 380, 250, "E. log / DCT-II", "log: exact 或 PWL\nDCT-II: n_mfcc=40, norm=ortho\n输出单帧: [B, 40]\n累计 1 秒: [B, 40, 32]", LIGHT_PURPLE, PURPLE, PURPLE)
    grid(draw, 1435, y + 235, 8, 5, 19, 15, "#D4C4FF", "#7355B8")
    for x1, x2 in [(345, 390), (660, 705), (975, 1020), (1290, 1335)]:
        arrow(draw, (x1, y + 215), (x2 - 10, y + 215))

    # Frontend shape prep
    card(draw, 70, y + 420, 300, 300, "MFCC 原始输出", "StreamingMFCC.forward\n输出: [B, 40, 32]\n40 个 cepstral 系数\n32 个时间帧", LIGHT_BLUE)
    grid(draw, 135, y + 555, 8, 5, 22, 18, "#BBD6FF")
    card(draw, 430, y + 420, 300, 300, "系数选择", "features[:, :10, :]\n只取前 10 个 MFCC\n输出: [B, 10, 32]", LIGHT_BLUE)
    grid(draw, 505, y + 555, 8, 3, 22, 18, "#A9CBFF")
    card(draw, 790, y + 420, 300, 300, "维度转换", "transpose / permute\n[B, 10, 32]\n→ [B, 32, 10]\n时间维作为序列", LIGHT_BLUE)
    grid(draw, 865, y + 540, 5, 8, 22, 15, "#A9CBFF")
    card(draw, 1150, y + 420, 500, 300, "训练路径与在线路径一致", "训练: 整段 waveform → 32 帧特征 → CRNN\n在线: 每 32 ms 新 chunk → 1 帧 MFCC → CRNN step\n关键: 前端不等待完整 1 秒即可产出当前帧", "#FFF9E8", ORANGE, ORANGE)
    for x1, x2 in [(370, 430), (730, 790), (1090, 1150)]:
        arrow(draw, (x1, y + 570), (x2 - 10, y + 570))

    # 3. CRNN input and states
    y = 1540
    section(draw, y, "3. CRNN 输入与流式状态", 520)
    card(draw, 55, y + 90, 320, 300, "输入特征序列", "离线张量: [B, T=32, F=10]\n在线单帧: [B, 10]\n每 32 ms 更新一次", LIGHT_BLUE)
    grid(draw, 115, y + 235, 5, 8, 24, 16, "#BBD6FF")
    card(draw, 450, y + 90, 360, 300, "CNN cache", "每个 causal DSConv 保存\nkernel_time - 1 = 4 帧\nBlock1: [B,1,4,10]\nBlock2-5: [B,24,4,10]", LIGHT_GREEN, GREEN, GREEN)
    cube(draw, 520, y + 235, 100, 70, 35, "#BFF0D1", GREEN)
    cube(draw, 640, y + 235, 100, 70, 35, "#BFF0D1", GREEN)
    card(draw, 880, y + 90, 360, 300, "GRU hidden", "1 层 GRU\nhidden = 64\n状态形状: [1, B, 64]\nh_t = GRU(x_t, h_{t-1})", LIGHT_PURPLE, PURPLE, PURPLE)
    grid(draw, 950, y + 245, 12, 2, 18, 22, "#D4C4FF", "#7355B8")
    card(draw, 1310, y + 90, 390, 300, "输出节拍", "每帧得到 logits [B,2]\n可做 argmax 分类\n或 softmax positive 分数\n配合阈值与 cooldown 触发", "#FFF8F8", RED, RED)
    arrow(draw, (375, y + 240), (450, y + 240))
    arrow(draw, (810, y + 240), (880, y + 240))
    arrow(draw, (1240, y + 240), (1310, y + 240))

    # 4. Backbone blocks
    y = 2090
    section(draw, y, "4. CRNN 主干网络：5 层 causal depthwise-separable CNN + GRU", 1550)
    block1, block, cnn, gru, fc, total, mac_window, mac_frame = param_counts()
    card(draw, 55, y + 90, 230, 300, "输入到 CNN", "reshape\n[B,32,10]\n→ [B,1,32,10]\n时间=32\nfreq=10", LIGHT_BLUE)
    cube(draw, 100, y + 225, 82, 70, 35)
    x = 330
    for i in range(5):
        if i == 0:
            title = "Block 1"
            body = (
                "Causal DSConv2d\n"
                "in=1, out=24\n"
                "depthwise k=5×3, groups=1\n"
                "time left pad=4, freq pad=1\n"
                "pointwise 1×1: 1→24\n"
                "BN + ReLU\n"
                f"params={block1}\n"
                "输出: [B,24,32,10]"
            )
        else:
            title = f"Block {i + 1}"
            body = (
                "Depthwise-Separable\n"
                "in=24, out=24\n"
                "depthwise k=5×3, groups=24\n"
                "time left pad=4, freq pad=1\n"
                "pointwise 1×1: 24→24\n"
                "BN + ReLU\n"
                f"params={block}\n"
                "输出: [B,24,32,10]"
            )
        card(draw, x, y + 90, 260, 500, title, body, "white", BLUE)
        grid(draw, x + 65, y + 435, 6, 4, 18, 16, "#C5DCFF")
        if i == 0:
            arrow(draw, (285, y + 250), (x - 12, y + 250))
        else:
            arrow(draw, (x - 55, y + 250), (x - 12, y + 250))
        x += 285
    draw.rounded_rectangle((75, y + 650, W - 75, y + 780), radius=16, fill="#F8FBFF", outline=BLUE2, width=2)
    label(
        draw,
        105,
        y + 675,
        "说明：CNN 全部使用 stride=1，时间维不下采样；causal padding 只看当前帧和历史 4 帧，因此在线推理可用 cache 增量更新。",
        fnt=F_BODY,
        max_width=W - 210,
    )

    card(draw, 95, y + 860, 350, 420, "频率池化", "CNN 输出 [B,24,32,10]\n对 freq 维求均值\nmean over frequency\n得到序列 [B,32,24]", LIGHT_BLUE)
    grid(draw, 160, y + 1035, 6, 4, 20, 16, "#BBD6FF")
    arrow(draw, (445, y + 1070), (535, y + 1070))
    card(draw, 535, y + 860, 430, 420, "GRU 时序建模", "GRU input_size=24\nhidden_size=64\nnum_layers=1\nweight_ih: 3H×I = 192×24\nweight_hh: 3H×H = 192×64\nbias: 2×192\nparams=17280\n输出序列: [B,32,64]", LIGHT_PURPLE, PURPLE, PURPLE)
    grid(draw, 615, y + 1140, 12, 2, 18, 22, "#D4C4FF", "#7355B8")
    arrow(draw, (965, y + 1070), (1055, y + 1070))
    card(draw, 1055, y + 860, 300, 420, "取最后时间步", "last = output[:, -1, :]\n形状: [B,64]\n代表当前 1 秒上下文\n在线时每帧都可输出", "#F8FBFF", BLUE)
    grid(draw, 1120, y + 1135, 12, 2, 16, 22, "#BBD6FF")
    arrow(draw, (1355, y + 1070), (1440, y + 1070))
    card(draw, 1440, y + 860, 260, 420, "FC 分类", "Linear 64→2\nparams=130\nlogits [B,2]\n类别:\n0 negative\n1 positive", "#FFF8F8", RED, RED)
    draw.ellipse((1510, y + 1120, 1534, y + 1144), fill="#BBD6FF", outline=BLUE)
    draw.ellipse((1510, y + 1160, 1534, y + 1184), fill="#5B8DEF", outline=BLUE)

    # 5. Params and compute
    y = 3675
    section(draw, y, "5. 参数量、计算量与硬件状态汇总", 850)
    card(
        draw,
        60,
        y + 90,
        500,
        340,
        "默认网络参数量",
        f"CNN block1: {block1}\nCNN block2-5: {block} × 4\nCNN total: {cnn}\nGRU total: {gru}\nFC: {fc}\n总参数量: {total}\n约低于 L5_C64 DSCNN(22530)",
        LIGHT_BLUE,
    )
    card(
        draw,
        650,
        y + 90,
        500,
        340,
        "计算量估算",
        f"完整 1 秒: {mac_window:,} MAC\n流式每帧: {mac_frame:,.0f} MAC\n每 32 ms 更新一次\n相比滑窗整秒重算，后端计算显著减少",
        LIGHT_GREEN,
        GREEN,
        GREEN,
    )
    card(
        draw,
        1240,
        y + 90,
        450,
        340,
        "在线状态寄存器",
        "pre-emphasis: prev_sample\nMFCC: pending/history\nCNN cache: 每层 4 帧\nGRU hidden: 64 维\n触发: score/threshold/cooldown",
        LIGHT_ORANGE,
        ORANGE,
        ORANGE,
    )
    draw.rounded_rectangle((70, y + 500, W - 70, y + 720), radius=16, fill="#F8FBFF", outline=BLUE, width=3)
    label(draw, 105, y + 530, "端到端流式执行顺序", fill=BLUE, fnt=F_H2)
    steps = [
        "new audio chunk [B,512]",
        "stateful pre-emphasis",
        "StreamingMFCC → one frame [B,10]",
        "CNN cache update",
        "GRU hidden update",
        "logits / score / trigger",
    ]
    sx = 105
    for i, s in enumerate(steps):
        card(draw, sx, y + 585, 245, 85, f"{i + 1}", s, "white", BLUE)
        if i < len(steps) - 1:
            arrow(draw, (sx + 245, y + 628), (sx + 285, y + 628), BLUE, 4)
        sx += 285

    # 6. Notes
    y = 4550
    section(draw, y, "6. 关键注意事项", 500)
    notes = [
        "当前训练脚本默认 frontend=mfcc + streaming_mfcc=True，特征提取使用 StreamingMFCC。",
        "window_size_ms = window_stride_ms = 32，因此 StreamingMFCC 的 history_length = 0；若未来窗口大于 hop，需要保存历史音频。",
        "CRNN 的 CNN 部分不做时间下采样，便于每帧增量计算；GRU hidden 保存长期时序上下文。",
        "pre-emphasis 若在硬件中逐 chunk 执行，需要保存上一采样点 prev_sample，避免 chunk 边界误差。",
        "普通分类可用 argmax；实际唤醒建议使用 positive 概率阈值 + 连续帧确认 + cooldown。",
    ]
    ny = y + 100
    for note in notes:
        draw.ellipse((85, ny + 8, 103, ny + 26), fill=BLUE)
        ny = label(draw, 120, ny, note, fnt=F_BODY, max_width=W - 220, line_gap=8) + 6

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, quality=95)


def main() -> None:
    out = Path("dscnn_kws/streaming/figures/streaming_mfcc_crnn_inference_structure_17x6.png")
    draw_diagram(out)
    print(f"[INFO] saved: {out.resolve()}")


if __name__ == "__main__":
    main()

