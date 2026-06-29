# 2026_06_29_cv_old

`old` 视觉预训练模型，用于视觉 student 的 `old + xd/shared/feature` 输入。

## 文件

- `best.pt`: 从 `models/vision/cv_old/best.pt` 复制，是当前 `old` CV checkpoint。
- `log.csv`: CV 训练日志。

## 训练入口

原训练入口是：

```bash
python scripts/train_cv_all.py --train --updates 1000 --batch-size 128 --device cuda
```

实际 RL 训练读取路径固定为：

```text
models/vision/cv_old/best.pt
```

## 服务器恢复

服务器 clone 后执行：

```bash
mkdir -p models/vision/cv_old
cp approved_models/2026_06_29_cv_old/best.pt models/vision/cv_old/best.pt
```

然后可以直接运行视觉 student：

```bash
./IsaacLab/isaaclab.sh -p train/student_vision.py --cv-model old --state shared --student
```

## 说明

这个目录只保存视觉预训练依赖，不保存 RL policy。周末 RL 对照结果在 `approved_models/2026_06_27_weekend_students/`。
