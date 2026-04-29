# 手写数字识别实验

基于CNN的MNIST手写数字识别系统。

## 项目结构

```
├── app.py                      # Flask Web应用
├── cnn_model.pth               # 训练好的CNN模型
├── CNN手写数字识别实验报告.md # 实验报告
├── optimized_train.py          # 训练脚本
├── requirements.txt            # 依赖列表
├── templates/
│   └── index.html             # 前端页面
└── digit-recognizer/
    ├── sample_submission.csv  # Kaggle提交文件
    └── *.png                  # 图表和截图
```

## 运行Web应用

```bash
python app.py
```

访问: http://127.0.0.1:8080

## 模型准确率

- **Kaggle Score**: 99.2%
- **模型**: AdvancedCNN with BatchNorm + Dropout
