import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = [("Qwen3-8B", 38.8, 58.5), ("Llama-3.1-8B", 81.5, 4.2), ("gemma-2-9b", 87.7, 3.6)]
COLS = {"Latin": "#2166ac", "Han": "#b2182b", "other": "#bdbdbd"}
INK = {"Latin": "white", "Han": "white", "other": "#222222"}
FS, FLOOR = 7.0, 5.0

fig, axes = plt.subplots(1, 3, figsize=(5.5, 0.8))
segs = []
for ax, (model, lat, han) in zip(axes, DATA):
    x = 0.0
    for part, val in [("Latin", lat), ("Han", han), ("other", round(100.0 - lat - han, 1))]:
        patch = ax.barh(0, val, left=x, height=1.0, color=COLS[part], edgecolor="white", linewidth=0.5).patches[0]
        t = ax.text(
            x + val / 2,
            0,
            f"{part}\n{val:.0f}%",
            ha="center",
            va="center",
            fontsize=FS,
            color=INK[part],
            linespacing=1.05,
        )
        segs.append((ax, patch, t, x + val / 2))
        x += val
    ax.set_title(model, fontsize=7.5, pad=3, loc="left")
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 0.5)
    ax.axis("off")
fig.subplots_adjust(left=0.004, right=0.996, top=0.66, bottom=0.02, wspace=0.07)

r = fig.canvas.get_renderer()
m = 0.5 * fig.dpi / 72
report, above = {}, {}
for ax, patch, t, xc in segs:
    while True:
        fig.canvas.draw()
        tb, pb = t.get_window_extent(r), patch.get_window_extent(r)
        fits = tb.x0 >= pb.x0 + m and tb.x1 <= pb.x1 - m and tb.y0 >= pb.y0 and tb.y1 <= pb.y1
        if fits or t.get_fontsize() - 0.25 < FLOOR:
            break
        t.set_fontsize(t.get_fontsize() - 0.25)
    if not fits:
        t.set_text(t.get_text().replace("\n", " "))
        t.set_fontsize(FLOOR + 0.5)
        t.set_color("#222222")
        t.set_va("bottom")
        t.set_ha("right")
        above.setdefault(ax, []).append((xc, t))
        report[t.get_text()] = "above"
    else:
        report[t.get_text() + f"({patch.get_x():.0f})"] = t.get_fontsize()

GAP = 2.0
for ax, items in above.items():
    right = 100.0
    for xc, t in sorted(items, key=lambda it: -it[0]):
        t.set_position((0, 0.62))
        fig.canvas.draw()
        bb = t.get_window_extent(r).transformed(ax.transData.inverted())
        w = bb.x1 - bb.x0
        xr = min(right, xc + w / 2)
        t.set_position((xr, 0.62))
        ax.plot([xc, xr - w / 2], [0.5, 0.61], color="#222222", lw=0.5, clip_on=False)
        right = xr - w - GAP
fig.savefig("latent-language.pdf", bbox_inches="tight", pad_inches=0.01)
print(report)
