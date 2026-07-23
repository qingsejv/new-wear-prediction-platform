
from pathlib import Path
import base64
import urllib.request

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import font_manager
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "friction_wear_model.joblib"

HERO_BG_PATH = BASE_DIR / "assets" / "twin_roll_ai_background.png"


def load_background_data_uri(path):
    """读取本地背景图片并转换为CSS可使用的数据地址。"""
    try:
        if not path.exists():
            return ""
        encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"
    except Exception:
        return ""


HERO_BG_URI = load_background_data_uri(HERO_BG_PATH)

PLATFORM_NAME = "AI驱动双辊薄带连铸镀层智能预测系统"
SUBTITLE = "面向镀铬层摩擦磨损行为的小样本推理、低碳工况分析与节能减排优化"

st.set_page_config(
    page_title=PLATFORM_NAME,
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------------- Matplotlib font ----------------
def get_chinese_font():
    def use_font(font_path):
        try:
            font_manager.fontManager.addfont(str(font_path))
        except Exception:
            pass
        prop = font_manager.FontProperties(fname=str(font_path))
        plt.rcParams["font.family"] = prop.get_name()
        plt.rcParams["font.sans-serif"] = [prop.get_name()]
        plt.rcParams["axes.unicode_minus"] = False
        return prop

    candidates = [
        BASE_DIR / "fonts" / "NotoSansCJKsc-Regular.otf",
        BASE_DIR / "fonts" / "NotoSansSC-Regular.otf",
        BASE_DIR / "fonts" / "SourceHanSansSC-Regular.otf",
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
    ]
    for path in candidates:
        if path.exists():
            return use_font(path)

    try:
        runtime_font = Path("/tmp/NotoSansCJKsc-Regular.otf")
        if not runtime_font.exists() or runtime_font.stat().st_size < 1024 * 1024:
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf",
                runtime_font,
            )
        if runtime_font.exists():
            return use_font(runtime_font)
    except Exception:
        pass

    plt.rcParams["axes.unicode_minus"] = False
    return None


CHINESE_FONT = get_chinese_font()


# ---------------- Model ----------------
@st.cache_resource
def load_model_package():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"未找到模型文件：{MODEL_PATH}\n"
            "请保持 models/friction_wear_model.joblib 与本文件的相对路径不变。"
        )
    return joblib.load(MODEL_PATH)


try:
    pkg = load_model_package()
except Exception as exc:
    st.error("模型加载失败")
    st.code(str(exc))
    st.stop()

friction_model = pkg["friction_model"]
wear_model = pkg.get("wear_model")
feature_cols = pkg.get(
    "feature_cols",
    ["温度 ℃", "磨损载荷 N", "镀层实际厚度 μm", "粗糙度Rz μm"],
)


def get_model_display_name(model):
    if model is None:
        return "未加载"
    try:
        if hasattr(model, "named_steps") and model.named_steps:
            model = list(model.named_steps.values())[-1]
    except Exception:
        pass

    names = {
        "RandomForestRegressor": "Random Forest",
        "ExtraTreesRegressor": "Extra Trees",
        "GradientBoostingRegressor": "GBDT",
        "HistGradientBoostingRegressor": "Hist-GBDT",
        "XGBRegressor": "XGBoost",
        "LGBMRegressor": "LightGBM",
        "CatBoostRegressor": "CatBoost",
        "SVR": "SVR",
        "KNeighborsRegressor": "KNN Regressor",
        "LinearRegression": "Linear Regression",
        "Ridge": "Ridge",
        "Lasso": "Lasso",
        "ElasticNet": "Elastic Net",
        "PLSRegression": "PLS",
        "MLPRegressor": "MLP",
    }
    return names.get(type(model).__name__, type(model).__name__)


FRICTION_MODEL_NAME = get_model_display_name(friction_model)
WEAR_MODEL_NAME = get_model_display_name(wear_model)

FEATURE_DISPLAY_NAMES = {
    "温度 ℃": "温度（℃）",
    "磨损载荷 N": "铸轧力（N）",
    "磨损载荷 kN": "铸轧力（kN）",
    "镀层实际厚度 μm": "镀层实际厚度（μm）",
    "粗糙度Rz μm": "粗糙度（μm）",
}

FEATURE_CHART_NAMES = {
    "温度 ℃": "温度（℃）",
    "磨损载荷 N": "铸轧力（N）",
    "磨损载荷 kN": "铸轧力（kN）",
    "镀层实际厚度 μm": "初始镀层厚度（μm）",
    "粗糙度Rz μm": "表面粗糙度（μm）",
}


def display_feature_name(col):
    return FEATURE_DISPLAY_NAMES.get(
        col,
        col.replace("磨损载荷", "铸轧力").replace("粗糙度Rz", "粗糙度"),
    )


def chart_feature_name(col):
    return FEATURE_CHART_NAMES.get(col, col)


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


def widget_keys(col):
    return f"slider_{col}", f"num_{col}"


def clamp_and_snap(value, vmin, vmax, step):
    value = max(float(vmin), min(float(vmax), float(value)))
    if step > 0:
        value = float(vmin) + round((value - float(vmin)) / float(step)) * float(step)
    return round(value, 10)


def sync_num_from_slider(col):
    slider_key, num_key = widget_keys(col)
    st.session_state[num_key] = st.session_state[slider_key]


def sync_slider_from_num(col):
    slider_key, num_key = widget_keys(col)
    vmin, vmax, _, step = default_range(col)
    value = clamp_and_snap(st.session_state[num_key], vmin, vmax, step)
    st.session_state[num_key] = value
    st.session_state[slider_key] = value


def reset_inputs():
    for col in feature_cols:
        _, _, default, _ = default_range(col)
        slider_key, num_key = widget_keys(col)
        st.session_state[slider_key] = float(default)
        st.session_state[num_key] = float(default)


def tight_y_limits(values, current_value=None, target_name=""):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if current_value is not None and np.isfinite(current_value):
        finite = np.append(finite, float(current_value))
    if finite.size == 0:
        return 0.0, 1.0

    data_min = float(np.min(finite))
    data_max = float(np.max(finite))
    center = float(np.mean(finite))
    span = data_max - data_min
    if "摩擦系数" in target_name:
        minimum_span = max(abs(center) * 0.0025, 0.003)
    else:
        minimum_span = max(abs(center) * 0.012, 0.02)
    visible_span = max(span, minimum_span)
    padding = visible_span * 0.10
    return data_min - padding, data_max + padding


# ---------------- Theme ----------------

theme_css = """
<style>
:root{
  --primary:#1769ff;
  --primary-2:#5b7cff;
  --violet:#7868ff;
  --cyan:#1ab7ff;
  --ink:#171b24;
  --muted:#6d7584;
  --line:#e8edf5;
  --soft:#f5f8ff;
  --card:#ffffff;
  --success:#00a870;
}

html{scroll-behavior:smooth;}
html,body,.stApp,[class*="css"]{
  font-family:"Microsoft YaHei","PingFang SC","Noto Sans CJK SC","Source Han Sans SC","Hiragino Sans GB",Arial,sans-serif;
  font-synthesis:none;
  -webkit-font-smoothing:antialiased;
  text-rendering:optimizeLegibility;
}
.stApp{
  background:
    radial-gradient(circle at 86% 2%, rgba(42,119,255,.14), transparent 30rem),
    radial-gradient(circle at 9% 38%, rgba(91,102,220,.08), transparent 26rem),
    linear-gradient(180deg,#f4f7fc 0%,#eef4fb 100%);
  color:var(--ink);
}
[data-testid="stHeader"]{background:transparent;height:0;}
[data-testid="stToolbar"]{display:none;}
#MainMenu, footer{visibility:hidden;}
.block-container{max-width:1560px;padding:0 2.2rem 3rem;}
[data-testid="collapsedControl"]{display:none;}

.topbar{
  position:sticky;top:0;z-index:999;
  margin:0 -2.2rem;
  padding:0 2.2rem;
  height:74px;
  display:flex;align-items:center;gap:32px;
  background:rgba(255,255,255,.93);
  border-bottom:1px solid rgba(223,230,241,.88);
  backdrop-filter:blur(18px);
}
.brand{display:flex;align-items:center;gap:12px;min-width:330px;}
.brand-mark{
  width:34px;height:34px;border-radius:10px;
  display:grid;place-items:center;color:white;font-weight:900;
  background:linear-gradient(135deg,var(--violet),var(--cyan));
  box-shadow:0 10px 25px rgba(53,117,255,.26);
}
.brand-cn{font-weight:800;font-size:1.05rem;letter-spacing:0;}
.brand-en{font-size:.67rem;color:#8a93a3;letter-spacing:.09em;margin-top:2px;}
.topnav{display:flex;gap:26px;align-items:center;flex:1;}
.topnav a{color:#434a57;text-decoration:none;font-size:.92rem;font-weight:700;}
.topnav a:hover{color:var(--primary);}
.nav-actions{display:flex;gap:9px;align-items:center;margin-left:auto;}
.nav-chip{
  padding:10px 15px;border-radius:8px;font-size:.82rem;font-weight:800;
  background:#f6f8fc;color:#596170;border:1px solid #edf0f6;
}
.nav-login{
  padding:12px 24px;border-radius:0;background:linear-gradient(135deg,#2f82f6,#1769ff);
  color:white;font-size:.86rem;font-weight:900;box-shadow:0 9px 23px rgba(23,105,255,.24);
}

.hero-shell{
  position:relative;
  overflow:hidden;
  margin:0 -2.2rem;
  padding:42px 2.2rem 0;
  background:
    linear-gradient(
      90deg,
      rgba(238,245,255,.97) 0%,
      rgba(235,243,255,.93) 30%,
      rgba(226,238,255,.74) 52%,
      rgba(11,48,107,.24) 73%,
      rgba(3,29,75,.12) 100%
    ),
    url("__HERO_BG_URI__") center right / cover no-repeat;
  box-shadow:inset 0 -1px 0 rgba(219,228,242,.90);
}
.hero-shell:before{
  content:"";
  position:absolute;
  inset:0;
  pointer-events:none;
  background:
    linear-gradient(180deg,rgba(255,255,255,.12),rgba(224,237,255,.08)),
    radial-gradient(circle at 18% 48%,rgba(255,255,255,.72),transparent 31rem);
}
.hero-shell > *{
  position:relative;
  z-index:1;
}
.hero{
  position:relative;overflow:hidden;min-height:360px;
  border-radius:0;
  display:grid;grid-template-columns:1.16fr .84fr;gap:38px;
  padding:44px 52px 34px;
}
.hero:before{
  content:"";
  position:absolute;
  right:-120px;
  top:-170px;
  width:620px;
  height:620px;
  border-radius:50%;
  background:radial-gradient(circle,rgba(39,132,255,.18),rgba(53,104,215,.05) 48%,transparent 72%);
  pointer-events:none;
}
.hero:after{
  content:"";
  position:absolute;
  inset:0;
  pointer-events:none;
  opacity:.20;
  background-image:
    linear-gradient(120deg,transparent 0 61%,rgba(157,216,255,.34) 64%,transparent 67%),
    radial-gradient(circle at 80% 20%,rgba(130,212,255,.92) 0 1.5px,transparent 3px);
  background-size:auto,92px 92px;
}
.hero-copy{
  position:relative;
  z-index:2;
  align-self:center;
  padding:20px 22px 22px 0;
  border-radius:22px;
  text-shadow:0 1px 0 rgba(255,255,255,.65);
}
.eyebrow{
  display:inline-flex;align-items:center;gap:9px;padding:7px 12px;border-radius:999px;
  background:rgba(234,241,255,.88);color:#285fd1;border:1px solid rgba(210,226,255,.92);backdrop-filter:blur(8px);
  font-size:.74rem;font-weight:900;letter-spacing:.08em;
}
.eyebrow-dot{width:8px;height:8px;border-radius:50%;background:#487bff;box-shadow:0 0 0 5px rgba(72,123,255,.13);}
.hero-title{
  margin:18px 0 15px;
  max-width:920px;
  font-size:clamp(2.25rem,2.65vw,2.75rem);
  line-height:1.22;
  letter-spacing:0;
  font-weight:800;
  color:#191d27;
  word-break:keep-all;
  overflow-wrap:normal;
  text-shadow:none;
}
.hero p{max-width:800px;color:#70798a;font-size:1rem;line-height:1.85;margin:0;}
.hero-actions{display:flex;gap:14px;margin-top:28px;}
.hero-btn{
  display:inline-flex;align-items:center;justify-content:center;height:52px;padding:0 27px;border-radius:9px;
  text-decoration:none;font-weight:900;font-size:.92rem;
}
.hero-btn.primary{color:#fff;background:linear-gradient(135deg,#6b5eff,#218cff);box-shadow:0 16px 34px rgba(45,112,255,.26);}
.hero-btn.secondary{color:#273043;background:#fff;border:1px solid #e1e7f1;box-shadow:0 9px 26px rgba(37,59,95,.07);}
.hero-visual{position:relative;z-index:2;display:grid;place-items:center;min-height:280px;}
.visual-panel{
  width:100%;max-width:520px;min-height:260px;border-radius:26px;padding:24px;
  background:rgba(255,255,255,.72);border:1px solid rgba(255,255,255,.88);
  box-shadow:0 28px 72px rgba(4,35,91,.27);backdrop-filter:blur(16px) saturate(115%);
  transform:perspective(900px) rotateY(-5deg) rotateX(2deg);
}
.visual-top{display:flex;justify-content:space-between;align-items:center;margin-bottom:20px;}
.visual-title{font-size:.78rem;color:#7d8797;font-weight:850;letter-spacing:.08em;}
.live{font-size:.72rem;color:#00a870;font-weight:900;background:#e9fbf4;border:1px solid #cef4e6;padding:6px 9px;border-radius:999px;}
.visual-chart{height:108px;display:flex;align-items:flex-end;gap:10px;padding:15px 10px;border-radius:16px;background:linear-gradient(180deg,#f3f7ff,#fff);}
.bar{flex:1;border-radius:8px 8px 3px 3px;background:linear-gradient(180deg,#6c72ff,#2aa8ff);box-shadow:0 8px 18px rgba(45,104,255,.16);}
.visual-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:14px;}
.visual-kpi{padding:14px;border-radius:14px;background:#fff;border:1px solid #edf1f7;}
.visual-kpi span{display:block;color:#929aaa;font-size:.69rem;}
.visual-kpi b{display:block;color:#222936;font-size:1.05rem;margin-top:6px;}

.promo-row{
  display:grid;grid-template-columns:repeat(4,1fr);
  background:rgba(255,255,255,.95);border-top:1px solid rgba(230,236,246,.92);border-bottom:1px solid rgba(230,236,246,.95);backdrop-filter:blur(12px);
}
.promo-item{padding:23px 26px;display:flex;gap:14px;min-height:112px;align-items:flex-start;}
.promo-item + .promo-item{border-left:1px solid #edf1f7;}
.promo-icon{
  flex:0 0 34px;width:34px;height:34px;border-radius:10px;display:grid;place-items:center;
  color:#276df3;background:#edf3ff;font-weight:950;border:1px solid #dfe9ff;
}
.promo-title{font-size:.93rem;font-weight:900;color:#232833;}
.promo-desc{font-size:.75rem;color:#8a92a1;line-height:1.55;margin-top:6px;}

.main-intro{
  margin:58px 0 28px;
  display:grid;grid-template-columns:1.05fr .95fr;gap:34px;align-items:end;
}
.main-intro h2{font-size:2.45rem;line-height:1.32;letter-spacing:0;margin:0;color:#191d25;font-weight:800;}
.main-intro p{margin:0;color:#7e8797;line-height:1.75;font-size:.94rem;}

.section-anchor{scroll-margin-top:90px;}
.section-head{
  display:flex;align-items:flex-start;justify-content:space-between;gap:20px;margin:14px 0 15px;
}
.section-head h3{margin:0;font-size:1.22rem;color:#1d222c;}
.section-head p{margin:6px 0 0;color:#8a93a2;font-size:.82rem;line-height:1.6;}
.section-tag{padding:7px 11px;border-radius:999px;background:#edf3ff;color:#386ee8;font-weight:850;font-size:.7rem;}

.white-card{
  background:#fff;border:1px solid #e8edf5;border-radius:18px;padding:24px;
  box-shadow:0 12px 36px rgba(39,62,98,.075);
}

/* Streamlit 原生带边框容器：内容会真实位于白框内部，不再产生空白卡片 */
div[data-testid="stVerticalBlockBorderWrapper"]{
  background:#ffffff !important;
  border:1px solid #e8edf5 !important;
  border-radius:18px !important;
  box-shadow:0 12px 36px rgba(39,62,98,.075) !important;
  padding:1.15rem 1.15rem .95rem !important;
}
div[data-testid="stVerticalBlockBorderWrapper"] > div{
  background:transparent !important;
}
div[data-testid="stVerticalBlockBorderWrapper"] h4{
  color:#1d222c;
  font-weight:800;
  letter-spacing:0;
}

.soft-card{
  background:linear-gradient(180deg,#f9fbff,#fff);border:1px solid #e6edf8;border-radius:17px;padding:20px;
}
.metric-card{
  position:relative;overflow:hidden;border-radius:18px;padding:21px 22px;
  background:linear-gradient(145deg,#f7faff,#fff);border:1px solid #e5ebf5;
}
.metric-card:after{
  content:"";position:absolute;right:-40px;top:-42px;width:120px;height:120px;border-radius:50%;
  background:radial-gradient(circle,rgba(63,132,255,.13),transparent 70%);
}
.metric-label{color:#7f8898;font-size:.8rem;font-weight:800;}
.metric-value{font-size:2.4rem;color:#1e66ef;font-weight:950;line-height:1.08;margin-top:8px;letter-spacing:-.045em;}
.metric-sub{color:#929aaa;font-size:.74rem;line-height:1.5;margin-top:8px;}
.metric-card.wear .metric-value{color:#7057ed;}
.metric-card.thickness .metric-value{color:#0c9bc3;}
.metric-card.friction .metric-value{color:#1769ff;}

.summary-row{
  display:flex;justify-content:space-between;gap:16px;padding:12px 0;
  border-bottom:1px solid #edf1f6;color:#7e8797;font-size:.82rem;
}
.summary-row b{color:#242a35;text-align:right;}
.model-status{display:inline-flex;align-items:center;gap:7px;color:#00a870;font-weight:900;}
.status-dot{width:7px;height:7px;border-radius:50%;background:#00b579;box-shadow:0 0 0 5px rgba(0,181,121,.10);}

.info-strip{
  display:flex;align-items:center;justify-content:space-between;gap:15px;margin:18px 0 24px;
  border-radius:15px;padding:16px 19px;background:linear-gradient(90deg,#eef5ff,#f6f4ff);
  border:1px solid #dfe9fb;
}
.info-title{font-weight:900;color:#26334a;}
.info-desc{font-size:.76rem;color:#7c8798;margin-top:4px;}
.info-badges{display:flex;gap:8px;flex-wrap:wrap;justify-content:flex-end;}
.info-badge{padding:6px 9px;border-radius:999px;background:#fff;border:1px solid #e0e7f3;color:#5f6d82;font-size:.69rem;font-weight:850;}

div[data-testid="stSlider"]{padding-top:.15rem;}
div[data-testid="stNumberInput"] input{border-radius:10px;background:#fbfcff;border:1px solid #dfe6f1;color:#2a303b;}
div[data-baseweb="select"]>div{border-radius:10px;background:#fbfcff;border-color:#dfe6f1;}
button[kind="primary"]{
  border:0!important;border-radius:10px!important;background:linear-gradient(135deg,#6c5cff,#218cff)!important;
  box-shadow:0 12px 28px rgba(51,103,255,.22)!important;font-weight:900!important;
}
button[kind="secondary"]{border-radius:10px!important;background:#fff!important;border-color:#dfe5ef!important;color:#465066!important;}
.stDownloadButton button{border-radius:10px!important;background:#f4f7fd!important;border-color:#dde5f1!important;color:#354057!important;font-weight:850!important;}
.stAlert{border-radius:13px;}
.small-note{font-size:.78rem;color:#858e9d;line-height:1.7;}
.chart-note{padding:12px 15px;border-radius:12px;background:#f4f7fd;border:1px solid #e4eaf4;color:#748094;font-size:.76rem;line-height:1.55;margin-bottom:14px;}
.chart-note b{color:#326fe8;}
.footer{text-align:center;color:#a0a7b2;font-size:.72rem;margin-top:32px;padding:22px 0 10px;border-top:1px solid #e7ebf2;}

@media(max-width:1120px){
  .brand{min-width:250px}.promo-row{grid-template-columns:repeat(2,1fr)}
  .promo-item:nth-child(3){border-left:0;border-top:1px solid #edf1f7}.promo-item:nth-child(4){border-top:1px solid #edf1f7}
}
@media(max-width:900px){
  .topnav,.nav-chip{display:none}.topbar{justify-content:space-between}
  .hero{grid-template-columns:1fr;padding:34px 24px}.hero-title{font-size:2.25rem}.hero-visual{display:none}
  .main-intro{grid-template-columns:1fr}.main-intro h2{font-size:2rem}
}
@media(max-width:640px){
  .block-container{padding:0 1rem 2rem}.topbar,.hero-shell{margin-left:-1rem;margin-right:-1rem}
  .topbar{padding:0 1rem}.brand-en{display:none}.nav-login{padding:10px 14px}
  .promo-row{grid-template-columns:1fr}.promo-item + .promo-item{border-left:0;border-top:1px solid #edf1f7}
  .hero-actions{flex-direction:column}.hero-btn{width:100%}
}
</style>
"""
if HERO_BG_URI:
    theme_css = theme_css.replace("__HERO_BG_URI__", HERO_BG_URI)
else:
    theme_css = theme_css.replace(
        'url("__HERO_BG_URI__") center right / cover no-repeat',
        'linear-gradient(135deg,#eaf3ff 0%,#d8e9ff 58%,#3d72c7 100%)'
    )

st.markdown(theme_css, unsafe_allow_html=True)

# ---------------- Header ----------------
st.markdown(
    f"""
<div class="topbar">
  <div class="brand">
    <div class="brand-mark">AI</div>
    <div>
      <div class="brand-cn">材料智能预测平台</div>
      <div class="brand-en">材料智能云平台</div>
    </div>
  </div>
  <div class="topnav">
    <a href="#overview">平台首页</a>
    <a href="#ai-workbench">智能预测</a>
    <a href="#trend-analysis">趋势分析</a>
    <a href="#result-export">结果导出</a>
  </div>
  <div class="nav-actions">
    <span class="nav-chip">文档</span>
    <span class="nav-chip">控制台</span>
    <span class="nav-login">模型在线</span>
  </div>
</div>
<div id="overview" class="hero-shell section-anchor">
  <div class="hero">
    <div class="hero-copy">
      <div class="eyebrow"><span class="eyebrow-dot"></span> AI 材料智能 · 绿色制造</div>
      <div class="hero-title">{PLATFORM_NAME}</div>
      <p>{SUBTITLE}</p>
      <div class="hero-actions">
        <a class="hero-btn primary" href="#ai-workbench">立即开始预测　→</a>
        <a class="hero-btn secondary" href="#trend-analysis">查看响应趋势</a>
      </div>
    </div>
    <div class="hero-visual">
      <div class="visual-panel">
        <div class="visual-top">
          <span class="visual-title">AI 实时推理</span>
          <span class="live">● 在线</span>
        </div>
        <div class="visual-chart">
          <div class="bar" style="height:39%"></div><div class="bar" style="height:58%"></div>
          <div class="bar" style="height:47%"></div><div class="bar" style="height:76%"></div>
          <div class="bar" style="height:66%"></div><div class="bar" style="height:92%"></div>
          <div class="bar" style="height:82%"></div>
        </div>
        <div class="visual-grid">
          <div class="visual-kpi"><span>特征向量</span><b>{len(feature_cols)}D</b></div>
          <div class="visual-kpi"><span>推理模式</span><b>实时</b></div>
          <div class="visual-kpi"><span>模型状态</span><b>在线</b></div>
        </div>
      </div>
    </div>
  </div>
  <div class="promo-row">
    <div class="promo-item"><div class="promo-icon">01</div><div><div class="promo-title">小样本智能推理</div><div class="promo-desc">面向有限实验数据，快速生成材料性能预测结果</div></div></div>
    <div class="promo-item"><div class="promo-icon">02</div><div><div class="promo-title">多目标同步输出</div><div class="promo-desc">同时预测磨损深度、当前厚度与平均摩擦系数</div></div></div>
    <div class="promo-item"><div class="promo-icon">03</div><div><div class="promo-title">绿色工况分析</div><div class="promo-desc">辅助减少重复试验、材料浪费与非必要能耗</div></div></div>
    <div class="promo-item"><div class="promo-icon">04</div><div><div class="promo-title">实时模型中枢</div><div class="promo-desc">模型状态、特征空间与推理结果集中展示</div></div></div>
  </div>
</div>

<div class="main-intro">
  <h2>更强模型、更准预测、<br>绿色制造智能体验</h2>
  <p>保留原有业务主题、模型推理、参数输入、趋势分析与结果导出功能，仅将整体视觉重构为更轻盈、更官方的云平台风格。</p>
</div>

<div class="info-strip">
  <div>
    <div class="info-title">AI + 绿色制造</div>
    <div class="info-desc">以模型预测辅助低碳工况筛选、节能减排分析与实验决策。</div>
  </div>
  <div class="info-badges">
    <span class="info-badge">碳排辅助优化</span>
    <span class="info-badge">节能分析</span>
    <span class="info-badge">材料利用率</span>
    <span class="info-badge">绿色工艺</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------- Input and prediction ----------------
st.markdown('<div id="ai-workbench" class="section-anchor"></div>', unsafe_allow_html=True)
st.markdown(
    """
<div class="section-head">
  <div><h3>AI 智能预测工作台</h3><p>配置材料工况参数，构建特征向量并由模型实时生成多目标预测。</p></div>
  <span class="section-tag">实时推理</span>
</div>
""",
    unsafe_allow_html=True,
)

left, middle, right = st.columns([1.12, 0.84, 0.92], gap="large")

with left:
    with st.container(border=True):
        st.markdown("#### 工况特征输入")
        st.caption("温度、铸轧力、镀层厚度与粗糙度共同构成当前模型输入。")

        input_values = {}
        for col in feature_cols:
            vmin, vmax, vdef, step = default_range(col)
            slider_key, num_key = widget_keys(col)

            if slider_key not in st.session_state:
                st.session_state[slider_key] = float(vdef)
            if num_key not in st.session_state:
                st.session_state[num_key] = float(vdef)

            slider_col, num_col = st.columns([0.72, 0.28])
            with slider_col:
                st.slider(
                    display_feature_name(col),
                    min_value=float(vmin),
                    max_value=float(vmax),
                    step=float(step),
                    key=slider_key,
                    on_change=sync_num_from_slider,
                    args=(col,),
                )
            with num_col:
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
            input_values[col] = float(st.session_state[num_key])

        button_a, button_b = st.columns([0.64, 0.36])
        with button_a:
            st.button("启动 AI 推理", type="primary", use_container_width=True)
        with button_b:
            st.button("重置特征", use_container_width=True, on_click=reset_inputs)

input_df = pd.DataFrame({c: [input_values[c]] for c in feature_cols})

pred_friction = None
pred_wear = None
predict_error = None

try:
    pred_friction = float(np.ravel(friction_model.predict(input_df))[0])
except Exception as exc:
    predict_error = exc

if wear_model is not None:
    try:
        pred_wear = float(np.ravel(wear_model.predict(input_df))[0])
    except Exception:
        pred_wear = None

thickness_col = next((c for c in feature_cols if "厚度" in c), None)
initial_thickness = float(input_df[thickness_col].iloc[0]) if thickness_col else None
current_thickness = (
    initial_thickness - pred_wear
    if pred_wear is not None and initial_thickness is not None
    else None
)

with middle:
    with st.container(border=True):
        st.markdown("#### 实时推理结果")
        st.caption("模型随当前特征向量自动刷新预测。")

        if pred_wear is not None:
            st.markdown(
                f"""
<div class="metric-card wear">
  <div class="metric-label">已磨损深度</div>
  <div class="metric-value">{pred_wear:.4f}</div>
  <div class="metric-sub">单位：μm · 磨损预测模型实时输出</div>
</div>
""",
                unsafe_allow_html=True,
            )
        else:
            st.warning("未检测到可用的磨损预测模型。")

        st.write("")
        if current_thickness is not None:
            st.markdown(
                f"""
<div class="metric-card thickness">
  <div class="metric-label">当前镀层厚度</div>
  <div class="metric-value">{current_thickness:.4f}</div>
  <div class="metric-sub">初始厚度 {initial_thickness:.2f} μm − 已磨损深度</div>
</div>
""",
                unsafe_allow_html=True,
            )
            if current_thickness < 0:
                st.warning("当前镀层厚度小于 0，请检查输入厚度或模型输出范围。")
        else:
            st.warning("无法计算当前镀层厚度。")

        st.write("")
        if pred_friction is not None:
            st.markdown(
                f"""
<div class="metric-card friction">
  <div class="metric-label">平均摩擦系数</div>
  <div class="metric-value">{pred_friction:.4f}</div>
  <div class="metric-sub">由摩擦系数预测模型实时推理</div>
</div>
""",
                unsafe_allow_html=True,
            )
        else:
            st.error("预测失败，请检查模型和输入特征是否匹配。")
            if predict_error:
                st.caption(str(predict_error))

with right:
    with st.container(border=True):
        st.markdown("#### AI 模型中枢")
        st.caption("展示当前模型结构、特征空间与运行状态。")
        st.markdown(
            f"""
<div class="summary-row"><span>预测任务</span><b>多目标材料性能预测</b></div>
<div class="summary-row"><span>摩擦系数模型</span><b>{FRICTION_MODEL_NAME}</b></div>
<div class="summary-row"><span>磨损深度模型</span><b>{WEAR_MODEL_NAME}</b></div>
<div class="summary-row"><span>特征空间</span><b>{len(feature_cols)} 维工况向量</b></div>
<div class="summary-row"><span>推理模式</span><b>实时推理</b></div>
<div class="summary-row"><span>引擎状态</span><b class="model-status"><span class="status-dot"></span>在线</b></div>
""",
            unsafe_allow_html=True,
        )
        st.write("")
        st.markdown(
            '<div class="small-note">模型输出属于已有实验数据分布内的小样本机器学习推理结果，不等同于新的实验测量值。用于论文展示、工况筛选和节能分析时，建议结合代表性实验进行验证。</div>',
            unsafe_allow_html=True,
        )

# ---------------- Trend chart ----------------
st.write("")
st.markdown('<div id="trend-analysis" class="section-anchor"></div>', unsafe_allow_html=True)
st.markdown(
    """
<div class="section-head">
  <div><h3>单因素响应趋势分析</h3><p>保持其他变量不变，扫描单个工况参数对模型输出的影响。</p></div>
  <span class="section-tag">纵轴局部放大</span>
</div>
""",
    unsafe_allow_html=True,
)

chart_col, snapshot_col = st.columns([1.28, 0.72], gap="large")

with chart_col:
    with st.container(border=True):
        st.markdown(
            '<div class="chart-note"><b>显示说明：</b>纵轴会根据当前预测区间自动缩放，仅放大观察趋势，不改变模型预测数据。</div>',
            unsafe_allow_html=True,
        )

        control_a, control_b = st.columns([0.58, 0.42])
        with control_a:
            selected_feature = st.selectbox(
                "选择扫描特征",
                feature_cols,
                format_func=display_feature_name,
            )
        with control_b:
            target_options = ["平均摩擦系数"]
            if wear_model is not None:
                target_options.append("已磨损深度")
            selected_target = st.selectbox("选择预测目标", target_options)

        vmin, vmax, _, _ = default_range(selected_feature)
        grid = np.linspace(vmin, vmax, 80)
        trend_df = pd.concat([input_df] * len(grid), ignore_index=True)
        trend_df[selected_feature] = grid

        try:
            if selected_target == "已磨损深度" and wear_model is not None:
                trend_model = wear_model
                current_prediction = pred_wear
                y_label = "已磨损深度（μm）"
                curve_label = "预测已磨损深度"
            else:
                trend_model = friction_model
                current_prediction = pred_friction
                y_label = "平均摩擦系数"
                curve_label = "预测平均摩擦系数"

            preds = np.ravel(trend_model.predict(trend_df)).astype(float)
            y_lower, y_upper = tight_y_limits(preds, current_prediction, selected_target)

            fig, ax = plt.subplots(figsize=(9.2, 4.5))
            fig.patch.set_facecolor("#ffffff")
            ax.set_facecolor("#ffffff")

            ax.plot(grid, preds, linewidth=8, alpha=.08, color="#4f71ff")
            ax.plot(grid, preds, linewidth=2.7, color="#3f6ff5", label=curve_label)
            ax.fill_between(grid, y_lower, preds, color="#6d78ff", alpha=.08)

            marker_indices = np.linspace(0, len(grid) - 1, 10, dtype=int)
            ax.scatter(
                grid[marker_indices],
                preds[marker_indices],
                s=20,
                color="#38a4f5",
                edgecolors="#ffffff",
                linewidths=.6,
                zorder=4,
            )

            if current_prediction is not None:
                ax.scatter(
                    [input_df[selected_feature].iloc[0]],
                    [current_prediction],
                    s=90,
                    color="#7b5cf4",
                    edgecolors="#ffffff",
                    linewidths=1.2,
                    zorder=5,
                    label="当前工况点",
                )

            ax.set_ylim(y_lower, y_upper)
            ax.margins(x=.025)
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
            ax.yaxis.set_major_formatter(
                FormatStrFormatter("%.4f" if selected_target == "平均摩擦系数" else "%.3f")
            )

            font_kwargs = {"fontproperties": CHINESE_FONT} if CHINESE_FONT else {}
            ax.set_xlabel(chart_feature_name(selected_feature), color="#566174", labelpad=10, **font_kwargs)
            ax.set_ylabel(y_label, color="#566174", labelpad=10, **font_kwargs)
            ax.set_title(
                f"单因素响应趋势 · {selected_target}",
                color="#212633",
                pad=14,
                fontweight="bold",
                **font_kwargs,
            )

            ax.grid(axis="y", alpha=.55, color="#e4eaf3", linestyle="--", linewidth=.8)
            ax.grid(axis="x", alpha=.3, color="#eef2f7", linestyle=":", linewidth=.7)
            ax.tick_params(colors="#778296", labelsize=9)
            for spine in ax.spines.values():
                spine.set_color("#dfe6ef")

            if CHINESE_FONT:
                for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
                    tick_label.set_fontproperties(CHINESE_FONT)
                legend = ax.legend(
                    frameon=False,
                    loc="lower center",
                    bbox_to_anchor=(.5, -.30),
                    ncol=2,
                    prop=CHINESE_FONT,
                )
            else:
                legend = ax.legend(
                    frameon=False,
                    loc="lower center",
                    bbox_to_anchor=(.5, -.30),
                    ncol=2,
                )
            for legend_text in legend.get_texts():
                legend_text.set_color("#4f596b")

            fig.tight_layout(rect=(0, .05, 1, 1))
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        except Exception as exc:
            st.warning("响应曲线生成失败，请检查模型与输入特征是否匹配。")
            st.caption(str(exc))

with snapshot_col:
    st.markdown('<div id="result-export" class="section-anchor"></div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("#### AI 推理快照")
        st.caption("记录本次输入和预测结果，便于导出及实验对照。")

        for col in feature_cols:
            st.markdown(
                f'<div class="summary-row"><span>{display_feature_name(col)}</span><b>{float(input_df[col].iloc[0]):.3g}</b></div>',
                unsafe_allow_html=True,
            )
        if pred_wear is not None:
            st.markdown(
                f'<div class="summary-row"><span>已磨损深度（μm）</span><b>{pred_wear:.4f}</b></div>',
                unsafe_allow_html=True,
            )
        if current_thickness is not None:
            st.markdown(
                f'<div class="summary-row"><span>当前镀层厚度（μm）</span><b>{current_thickness:.4f}</b></div>',
                unsafe_allow_html=True,
            )
        if pred_friction is not None:
            st.markdown(
                f'<div class="summary-row"><span>平均摩擦系数</span><b>{pred_friction:.4f}</b></div>',
                unsafe_allow_html=True,
            )

        result_df = input_df.rename(columns={c: display_feature_name(c) for c in input_df.columns}).copy()
        if pred_wear is not None:
            result_df["预测已磨损深度 μm"] = pred_wear
        if current_thickness is not None:
            result_df["预测当前镀层厚度 μm"] = current_thickness
        if pred_friction is not None:
            result_df["预测平均摩擦系数"] = pred_friction

        csv = result_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.write("")
        st.download_button(
            "下载 AI 推理结果 CSV",
            csv,
            file_name="ai_prediction_result.csv",
            mime="text/csv",
            use_container_width=True,
        )

st.markdown(
    '<div class="footer">AI 材料智能预测 · 绿色工艺分析 · 节能减排辅助决策 · 2026</div>',
    unsafe_allow_html=True,
)
