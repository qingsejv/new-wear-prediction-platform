import matplotlib.pyplot as plt
from matplotlib import font_manager
import os

# Windows 中文字体路径
font_path = r"C:\Windows\Fonts\msyh.ttc"   # 微软雅黑

if os.path.exists(font_path):
    font_manager.fontManager.addfont(font_path)
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams["font.family"] = font_name
else:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]

plt.rcParams["axes.unicode_minus"] = False


from pathlib import Path
import io
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
from matplotlib import font_manager

# Altair 图表在浏览器端渲染文字，能直接使用浏览器/系统中文字体，
# 比 Matplotlib 服务端出图更不容易出现中文方块乱码。
try:
    import altair as alt
except Exception:
    alt = None

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "friction_wear_model.joblib"

# ---------- Matplotlib 中文字体 ----------
def get_chinese_font():
    """优先加载 Windows/项目目录/Linux 中常见的中文字体，避免图表中文乱码。"""
    font_paths = [
        BASE_DIR / "fonts" / "NotoSansCJKsc-Regular.otf",
        BASE_DIR / "fonts" / "SourceHanSansSC-Regular.otf",
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
    ]

    for font_path in font_paths:
        if font_path.exists():
            try:
                font_manager.fontManager.addfont(str(font_path))
            except Exception:
                pass
            font_prop = font_manager.FontProperties(fname=str(font_path))
            try:
                plt.rcParams["font.sans-serif"] = [font_prop.get_name()]
            except Exception:
                pass
            plt.rcParams["axes.unicode_minus"] = False
            return font_prop

    # 如果没有找到字体文件，再从系统已安装字体名称中查找
    installed_names = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Zen Hei",
        "Arial Unicode MS",
    ]:
        if font_name in installed_names:
            plt.rcParams["font.sans-serif"] = [font_name]
            plt.rcParams["axes.unicode_minus"] = False
            return font_manager.FontProperties(family=font_name)

    # 未找到中文字体时不让程序报错；部署端可在 fonts 文件夹放入中文字体
    plt.rcParams["axes.unicode_minus"] = False
    return None


CHINESE_FONT = get_chinese_font()

# 图表中使用更适合展示的中文坐标名称，模型特征名本身保持不变
FEATURE_DISPLAY_NAMES = {
    "温度 ℃": "温度（℃）",
    "磨损载荷 N": "铸轧力（N）",
    "磨损载荷 kN": "铸轧力（kN）",
    "镀层实际厚度 μm": "镀层实际厚度（μm）",
    "粗糙度Rz μm": "粗糙度（μm）",
}


def display_feature_name(col):
    """仅修改界面显示名称，内部模型特征名保持不变，避免预测时报特征名不匹配。"""
    return FEATURE_DISPLAY_NAMES.get(
        col,
        col.replace("磨损载荷", "铸轧力").replace("粗糙度Rz", "粗糙度")
    )

PLATFORM_NAME = "双辊薄带连铸镀层磨损预测"
SUBTITLE = "AI + 材料小样本预测平台 · 面向双辊薄带轧制镀铬层摩擦行为的智能评估"

st.set_page_config(
    page_title=PLATFORM_NAME,
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model_package():
    return joblib.load(MODEL_PATH)

pkg = load_model_package()
friction_model = pkg["friction_model"]
wear_model = pkg.get("wear_model", None)
feature_cols = pkg.get("feature_cols", ["温度 ℃", "磨损载荷 N", "镀层实际厚度 μm", "粗糙度Rz μm"])

# ---------- UI CSS ----------
st.markdown("""
<style>
:root{
    --navy:#071a3d;
    --blue:#1f6fff;
    --cyan:#11c5ff;
    --purple:#7657ff;
    --card:#ffffff;
    --muted:#64748b;
    --line:#e8eef7;
}
.stApp {
    background:
      radial-gradient(circle at 20% 0%, rgba(31,111,255,0.12), transparent 32%),
      radial-gradient(circle at 80% 10%, rgba(118,87,255,0.12), transparent 30%),
      linear-gradient(135deg, #f7fbff 0%, #eef5ff 45%, #f8fbff 100%);
    color:#0f172a;
}
[data-testid="stSidebar"] {
    background:
      linear-gradient(180deg, #061631 0%, #09234f 55%, #071a3d 100%);
    border-right:1px solid rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] * { color: #eef6ff !important; }
.block-container {
    padding-top: 1.35rem;
    padding-bottom: 2rem;
    max-width: 1480px;
}
.hero {
    position: relative;
    overflow: hidden;
    padding: 30px 34px;
    border-radius: 24px;
    background:
      linear-gradient(120deg, rgba(255,255,255,0.94), rgba(248,252,255,0.83)),
      radial-gradient(circle at 88% 20%, rgba(17,197,255,0.18), transparent 28%);
    border:1px solid rgba(210,224,245,0.9);
    box-shadow: 0 20px 60px rgba(15, 56, 118, 0.10);
}
.hero:after{
    content:"";
    position:absolute;
    right:-120px;
    top:-120px;
    width:360px;
    height:360px;
    background:
      linear-gradient(60deg, rgba(31,111,255,0.12), rgba(118,87,255,0.08));
    border-radius:50%;
    filter: blur(2px);
}
.hero h1 {
    margin:0 0 8px 0;
    font-size: 2.35rem;
    letter-spacing: -0.04em;
    font-weight: 900;
    background: linear-gradient(90deg, #145cff, #6b4cff 65%, #00b7ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero p {
    margin:0;
    color:#52627a;
    font-size: 1.05rem;
}
.top-pills {
    display:flex;
    flex-wrap:wrap;
    gap:10px;
    margin-top:18px;
}
.pill {
    display:inline-flex;
    gap:8px;
    align-items:center;
    padding:8px 12px;
    border:1px solid #d8e6fb;
    border-radius: 999px;
    background: rgba(255,255,255,0.7);
    color:#27405f;
    font-size:0.88rem;
    font-weight:700;
}
.pill-dot{
    width:8px;
    height:8px;
    border-radius:99px;
    background:#11c5ff;
    box-shadow:0 0 0 5px rgba(17,197,255,0.12);
}
.card {
    border-radius: 22px;
    padding: 22px;
    background: rgba(255,255,255,0.86);
    border:1px solid rgba(211,225,246,0.9);
    box-shadow: 0 16px 38px rgba(15, 56, 118, 0.08);
    backdrop-filter: blur(12px);
    height:100%;
    margin-bottom: 14px;
}
.section-card {
    /* 顶部三个白色边框标题卡片：增大高度和内边距，保证放大后的黑字不挤出边框 */
    min-height: 112px;
    padding: 24px 28px;
    display:flex;
    flex-direction:column;
    justify-content:center;
}
.section-card .card-title {
    /* 只放大这些标题卡片里的黑色标题，例如“参数输入 / 预测结果 / 模型信息” */
    font-size: 1.34rem;
    line-height: 1.25;
    font-weight: 900;
    margin-bottom: 8px;
    color:#0b1f3a;
}
.section-card .section-caption {
    /* 放大标题下面的说明文字，并控制行高，让文字仍然落在边框内部 */
    font-size: 1.02rem;
    line-height: 1.55;
    margin-top: 0;
    margin-bottom: 0;
    color:#475569;
}
.card-title {
    font-size: 1.05rem;
    font-weight: 850;
    color:#10213d;
    margin-bottom: 16px;
    display:flex;
    align-items:center;
    gap:8px;
}
.metric-card {
    border-radius: 22px;
    padding: 22px;
    background:
      linear-gradient(135deg, rgba(31,111,255,0.12), rgba(255,255,255,0.86));
    border:1px solid #d7e7ff;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.9);
}
.metric-card.purple {
    background: linear-gradient(135deg, rgba(118,87,255,0.14), rgba(255,255,255,0.86));
    border-color:#e2ddff;
}
.metric-card.highlight {
    position: relative;
    overflow: hidden;
    padding: 24px 24px 22px 24px;
    border-width: 1.5px;
    box-shadow: 0 18px 42px rgba(31,111,255,0.13), inset 0 1px 0 rgba(255,255,255,0.95);
}
.metric-card.highlight:after {
    content:"";
    position:absolute;
    right:-42px;
    top:-42px;
    width:118px;
    height:118px;
    border-radius:999px;
    background:rgba(255,255,255,0.45);
}
.metric-card.wear {
    background: linear-gradient(135deg, rgba(255,77,109,0.18), rgba(255,255,255,0.90));
    border-color:#ffd2dc;
}
.metric-card.thickness {
    background: linear-gradient(135deg, rgba(17,197,255,0.18), rgba(255,255,255,0.90));
    border-color:#c9f3ff;
}
.metric-card.friction {
    background: linear-gradient(135deg, rgba(31,111,255,0.10), rgba(255,255,255,0.88));
    border-color:#d7e7ff;
}
.metric-card.highlight .metric-value {
    font-size:3.05rem;
}
.metric-card.wear .metric-value { color:#ff3d67; }
.metric-card.thickness .metric-value { color:#0aa7d8; }
.metric-label {
    color:#52627a;
    font-weight:800;
    margin-bottom: 4px;
}
.metric-value {
    font-size:2.6rem;
    line-height:1.05;
    font-weight:900;
    letter-spacing:-0.04em;
    color:#1768ff;
}
.metric-card.purple .metric-value { color:#704cff; }
.metric-sub {
    color:#6b7a90;
    font-size:0.88rem;
    margin-top: 8px;
}
.section-caption {
    color:#64748b;
    font-size:0.92rem;
    margin-top:-10px;
    margin-bottom:14px;
}
button[kind="primary"]{
    background: linear-gradient(90deg, #1f6fff, #7657ff) !important;
    border:none !important;
    box-shadow: 0 12px 28px rgba(31,111,255,0.24) !important;
}
.stSlider [data-baseweb="slider"] > div {
    color:#1f6fff;
}
input {
    border-radius:12px !important;
}
.summary-row {
    display:flex;
    justify-content:space-between;
    padding: 9px 0;
    border-bottom:1px solid #eef2f7;
    color:#334155;
}
.summary-row b {
    color:#1f6fff;
}
.footer {
    text-align:center;
    color:#7b8aa0;
    font-size:0.84rem;
    margin-top: 28px;
}
.nav-mini {
    display:grid;
    gap:12px;
    margin-top: 14px;
}
.nav-item {
    padding:12px 14px;
    border-radius:16px;
    background:rgba(255,255,255,0.08);
    border:1px solid rgba(255,255,255,0.08);
    font-weight:700;
}
.nav-item.active {
    background:linear-gradient(90deg, rgba(31,111,255,0.9), rgba(118,87,255,0.92));
}
.small-note {
    font-size:0.86rem;
    color:#64748b;
    line-height:1.65;
}
</style>

""", unsafe_allow_html=True)

# ---------- Section card helper ----------
def section_card(title, caption=None):
    """在 Streamlit 中把栏目标题直接放进白色圆角框。"""
    caption_html = f'<div class="section-caption">{caption}</div>' if caption else ''
    st.markdown(f"""
    <div class="card section-card">
        <div class="card-title">{title}</div>
        {caption_html}
    </div>
    """, unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### ⚙️ AI材料预测平台")
    st.caption("AI + Materials · Smart Prediction")
    st.markdown("""
    <div class="nav-mini">
      <div class="nav-item active">🏠 总览</div>
      <div class="nav-item">📈 摩擦系数预测</div>
      <div class="nav-item">🧬 小样本模型</div>
      <div class="nav-item">🖼️ 形貌生成</div>
      <div class="nav-item">📄 结果报告</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.success("模型状态：Online")
    st.write("模型文件：`friction_wear_model.joblib`")
    st.write("输入维度：4")
    st.write("输出指标：已磨损深度 / 当前镀层厚度 / 平均摩擦系数")

# ---------- Header ----------
st.markdown(f"""
<div class="hero">
  <h1>{PLATFORM_NAME}</h1>
  <p>{SUBTITLE}</p>
  <div class="top-pills">
    <span class="pill"><span class="pill-dot"></span> Model Online</span>
    <span class="pill">双模型预测</span>
    <span class="pill">小样本学习</span>
    <span class="pill">AI + 材料摩擦学</span>
  </div>
</div>
""", unsafe_allow_html=True)

st.write("")

# ---------- Feature names and defaults ----------
# Make robust mapping for possible names
def default_range(name):
    if "温度" in name:
        return 25.0, 400.0, 200.0, 5.0
    if "载荷" in name or "铸轧力" in name:
        return 1.0, 10.0, 3.0, 0.1
    if "厚度" in name:
        return 50.0, 150.0, 100.0, 1.0
    if "粗糙" in name or "Rz" in name:
        return 1.0, 25.0, 8.0, 0.1
    return 0.0, 100.0, 10.0, 1.0


# ---------- 同步滑条和右侧数值框 ----------
def widget_keys(col):
    """为每个输入变量生成稳定且唯一的 slider / number_input key。"""
    return f"slider_{col}", f"num_{col}"


def clamp_and_snap(value, vmin, vmax, step):
    """把数值限制在实验范围内，并尽量对齐到滑条步长。"""
    value = float(value)
    value = max(float(vmin), min(float(vmax), value))
    step = float(step)
    if step > 0:
        value = float(vmin) + round((value - float(vmin)) / step) * step
    return round(value, 10)


def sync_num_from_slider(col):
    """拖动左侧滑条后，同步右侧数字框。"""
    slider_key, num_key = widget_keys(col)
    st.session_state[num_key] = st.session_state[slider_key]


def sync_slider_from_num(col):
    """修改右侧数字框后，同步左侧滑条。"""
    slider_key, num_key = widget_keys(col)
    vmin, vmax, _, step = default_range(col)
    synced_value = clamp_and_snap(st.session_state[num_key], vmin, vmax, step)
    st.session_state[num_key] = synced_value
    st.session_state[slider_key] = synced_value


def reset_inputs():
    """点击重置按钮后，同时重置滑条和数字框。"""
    for col in feature_cols:
        _, _, vdef, _ = default_range(col)
        slider_key, num_key = widget_keys(col)
        st.session_state[slider_key] = float(vdef)
        st.session_state[num_key] = float(vdef)


# Input and result layout
left, mid, right = st.columns([1.12, 0.88, 1.0], gap="large")

with left:
    section_card("🎚️ 参数输入", "输入双辊薄带轧制镀铬层工况参数，建议控制在实验范围内。")

    input_values = {}

    for col in feature_cols:
        vmin, vmax, vdef, step = default_range(col)
        slider_key, num_key = widget_keys(col)

        # 首次打开页面时，初始化两个控件的状态
        if slider_key not in st.session_state:
            st.session_state[slider_key] = float(vdef)
        if num_key not in st.session_state:
            st.session_state[num_key] = float(vdef)

        c_a, c_b = st.columns([0.66, 0.34])

        with c_a:
            st.slider(
                display_feature_name(col),
                min_value=float(vmin),
                max_value=float(vmax),
                step=float(step),
                key=slider_key,
                on_change=sync_num_from_slider,
                args=(col,),
            )

        with c_b:
            st.number_input(
                "数值",
                min_value=float(vmin),
                max_value=float(vmax),
                step=float(step),
                key=num_key,
                label_visibility="collapsed",
                on_change=sync_slider_from_num,
                args=(col,),
                format="%.2f",
            )

        # 两个控件已经同步，取任意一个都可以；这里以右侧数字框为准
        input_values[col] = float(st.session_state[num_key])

    st.write("")
    btn_col1, btn_col2 = st.columns([0.62, 0.38])
    with btn_col1:
        predict_btn = st.button("▶ 开始预测", type="primary", use_container_width=True)
    with btn_col2:
        reset_hint = st.button("↺ 重置输入", use_container_width=True, on_click=reset_inputs)

input_df = pd.DataFrame({c: [input_values[c]] for c in feature_cols})

# Predict by default as well, so UI has values
try:
    pred_friction = float(np.ravel(friction_model.predict(input_df))[0])
except Exception as e:
    pred_friction = None
    predict_error = e

pred_wear = None
if wear_model is not None:
    try:
        pred_wear = float(np.ravel(wear_model.predict(input_df))[0])
    except Exception:
        pred_wear = None

# 将 wear_model 的输出作为“已磨损深度”，并由输入的初始厚度计算当前镀层厚度。
thickness_col = next((c for c in feature_cols if "厚度" in c), None)
initial_thickness = None
current_thickness = None
if thickness_col is not None:
    initial_thickness = float(input_df[thickness_col].iloc[0])

if pred_wear is not None and initial_thickness is not None:
    current_thickness = initial_thickness - pred_wear

with mid:
    section_card("📊 预测结果")

    if pred_wear is not None:
        st.markdown(f"""
        <div class="metric-card highlight wear">
            <div class="metric-label">已磨损深度</div>
            <div class="metric-value">{pred_wear:.4f}</div>
            <div class="metric-sub">单位：μm，由 wear_model 预测输出</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("未检测到 wear_model，无法输出已磨损深度。")

    st.write("")
    if current_thickness is not None:
        st.markdown(f"""
        <div class="metric-card highlight thickness">
            <div class="metric-label">当前镀层厚度</div>
            <div class="metric-value">{current_thickness:.4f}</div>
            <div class="metric-sub">单位：μm；当前厚度 = 初始厚度 {initial_thickness:.2f} μm - 已磨损深度</div>
        </div>
        """, unsafe_allow_html=True)
        if current_thickness < 0:
            st.warning("当前镀层厚度小于 0，请检查输入厚度或磨损深度模型输出是否在合理范围内。")
    else:
        st.warning("未找到厚度输入列，无法计算当前镀层厚度。")

    st.write("")
    if pred_friction is not None:
        st.markdown(f"""
        <div class="metric-card friction">
            <div class="metric-label">平均摩擦系数</div>
            <div class="metric-value">{pred_friction:.4f}</div>
            <div class="metric-sub">基于小样本 SVR 模型输出</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error(f"预测失败：{predict_error}")

with right:
    section_card("🧠 模型信息")
    st.markdown("""
    <div class="summary-row"><span>平台名称</span><b>双辊薄带轧制镀铬层</b></div>
    <div class="summary-row"><span>预测目标</span><b>已磨损深度 / 当前镀层厚度 / 平均摩擦系数</b></div>
    <div class="summary-row"><span>算法类型</span><b>小样本机器学习</b></div>
    <div class="summary-row"><span>模型状态</span><b>Online</b></div>
    """, unsafe_allow_html=True)
    st.write("")
    st.markdown('<div class="small-note">该平台采用轻量化、工程化界面设计，适合论文展示、答辩演示和后续部署。当前输入越接近实验数据范围，预测结果越可靠。</div>', unsafe_allow_html=True)

st.write("")

# ---------- Visualization ----------
chart_col1, chart_col2 = st.columns([1.25, 0.75], gap="large")

with chart_col1:
    section_card("📈 工况参数响应趋势")

    if pred_friction is not None:
        # Generate one-factor trend for first feature
        selected_feature = st.selectbox(
            "选择单因素趋势变量",
            feature_cols,
            index=0,
            format_func=display_feature_name
        )
        vmin, vmax, _, _ = default_range(selected_feature)
        grid = np.linspace(vmin, vmax, 80)
        trend_df = pd.concat([input_df] * len(grid), ignore_index=True)
        trend_df[selected_feature] = grid
        try:
            preds = np.ravel(friction_model.predict(trend_df))
            x_label = display_feature_name(selected_feature)
            y_label = "平均摩擦系数"

            # 优先使用 Altair：文字由浏览器渲染，中文通常不会乱码。
            # 若部署环境没有 altair，则自动回退到 Matplotlib。
            if alt is not None:
                trend_plot_df = pd.DataFrame({
                    x_label: grid,
                    y_label: preds,
                })
                current_point_df = pd.DataFrame({
                    x_label: [float(input_df[selected_feature].iloc[0])],
                    y_label: [float(pred_friction)],
                    "标记": ["当前输入值"],
                })

                font_family = "Microsoft YaHei, SimHei, Noto Sans CJK SC, Source Han Sans SC, WenQuanYi Zen Hei, sans-serif"

                line = alt.Chart(trend_plot_df).mark_line(strokeWidth=3).encode(
                    x=alt.X(f"{x_label}:Q", title=x_label),
                    y=alt.Y(f"{y_label}:Q", title=y_label),
                    tooltip=[
                        alt.Tooltip(f"{x_label}:Q", title=x_label, format=".2f"),
                        alt.Tooltip(f"{y_label}:Q", title=y_label, format=".4f"),
                    ],
                )

                point = alt.Chart(current_point_df).mark_circle(size=95).encode(
                    x=alt.X(f"{x_label}:Q", title=x_label),
                    y=alt.Y(f"{y_label}:Q", title=y_label),
                    tooltip=[
                        alt.Tooltip(f"{x_label}:Q", title=x_label, format=".2f"),
                        alt.Tooltip(f"{y_label}:Q", title=y_label, format=".4f"),
                        alt.Tooltip("标记:N", title="标记"),
                    ],
                )

                chart = (line + point).properties(
                    title="单因素响应趋势",
                    height=330,
                ).configure_axis(
                    labelFont=font_family,
                    titleFont=font_family,
                    labelFontSize=12,
                    titleFontSize=13,
                ).configure_title(
                    font=font_family,
                    fontSize=16,
                    anchor="middle",
                )

                st.altair_chart(chart, use_container_width=True)
            else:
                fig, ax = plt.subplots(figsize=(8.4, 3.6))

                ax.plot(
                    grid,
                    preds,
                    linewidth=2.5,
                    label="预测平均摩擦系数"
                )
                ax.scatter(
                    [input_df[selected_feature].iloc[0]],
                    [pred_friction],
                    s=70,
                    zorder=5,
                    label="当前输入值"
                )

                text_kwargs = {"fontproperties": CHINESE_FONT} if CHINESE_FONT else {}
                ax.set_xlabel(x_label, **text_kwargs)
                ax.set_ylabel(y_label, **text_kwargs)
                ax.set_title("单因素响应趋势", **text_kwargs)
                ax.grid(alpha=0.25)

                if CHINESE_FONT:
                    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
                        tick_label.set_fontproperties(CHINESE_FONT)
                    ax.legend(frameon=False, prop=CHINESE_FONT)
                else:
                    ax.legend(frameon=False)
                    st.info("提示：当前部署环境未找到中文字体。建议使用 Altair 图表，或在项目 fonts 文件夹放入中文字体文件。")

                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
        except Exception as e:
            st.warning(f"趋势图生成失败：{e}")

with chart_col2:
    section_card("📋 本次预测摘要")
    for c in feature_cols:
        st.markdown(f'<div class="summary-row"><span>{display_feature_name(c)}</span><b>{float(input_df[c].iloc[0]):.3g}</b></div>', unsafe_allow_html=True)
    if pred_wear is not None:
        st.markdown(f'<div class="summary-row"><span>已磨损深度 μm</span><b>{pred_wear:.4f}</b></div>', unsafe_allow_html=True)
    if current_thickness is not None:
        st.markdown(f'<div class="summary-row"><span>当前镀层厚度 μm</span><b>{current_thickness:.4f}</b></div>', unsafe_allow_html=True)
    if pred_friction is not None:
        st.markdown(f'<div class="summary-row"><span>平均摩擦系数</span><b>{pred_friction:.4f}</b></div>', unsafe_allow_html=True)

    result_df = input_df.copy()
    result_df = result_df.rename(columns={c: display_feature_name(c) for c in result_df.columns})
    if pred_wear is not None:
        result_df["预测已磨损深度 μm"] = pred_wear
    if current_thickness is not None:
        result_df["预测当前镀层厚度 μm"] = current_thickness
    if pred_friction is not None:
        result_df["预测平均摩擦系数"] = pred_friction

    csv = result_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button("下载预测结果 CSV", csv, file_name="prediction_result.csv", mime="text/csv", use_container_width=True)

st.markdown('<div class="footer">© 2025 双辊薄带轧制镀铬层的摩擦系数预测平台 · AI驱动材料摩擦学分析</div>', unsafe_allow_html=True)
