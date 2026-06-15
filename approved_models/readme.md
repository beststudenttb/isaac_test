# Approved Models

这里保存已经确认值得保留的模型及其说明文件。`models/` 是训练现场输出目录，不入库；只有从 `models/` 中挑选出来的结果才放到这里。

每个模型单独一个文件夹，命名为 `yyyy_mm_dd`。若同一天需要保留多个模型，可以在日期后加简短后缀。

推荐结构：

```text
approved_models/
  readme.md
  yyyy_mm_dd/
    readme.md
    config.py
    model.pt
    policy.zip
    trajectory.csv
```

每个模型文件夹中的 `readme.md` 用来说明模型用途、训练数据、主要指标和人工判断；`config.py` 用来保存当次训练参数；`.pt` / `.zip` 保存模型权重；CSV 保存轨迹或评估结果。

