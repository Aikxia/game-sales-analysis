# models/

训练脚本生成的随机森林模型保存在这里：

- `random_forest_log_target.joblib`

该二进制文件由 `py -m src.main` 生成，并已被 `.gitignore` 排除，不进入版本库。
使用 `py -m src.predict` 预测前，请先运行 `py -m src.main` 重新生成模型。
