# 双辊薄带轧制镀铬层的摩擦系数预测平台

## 文件说明
- `app.py`：Streamlit 预测平台主程序
- `models/friction_wear_model.joblib`：你提供的预测模型文件
- `网页版预览.html`：静态网页版界面预览，可直接双击打开看设计风格
- `requirements.txt`：运行依赖
- `启动平台.bat`：Windows 一键启动脚本

## 平台名称
双辊薄带轧制镀铬层的摩擦系数预测

## 运行方式
在本文件夹打开命令行，执行：

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

或双击：

```text
启动平台.bat
```

## 注意
网页版预览.html 只是设计预览，不连接模型。
真正可预测的平台是 app.py，需要用 Streamlit 打开。
