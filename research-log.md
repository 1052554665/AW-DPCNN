>add ResNet18 to the model registry and create the config

### Model specs

| Model | Params | FLOPs |
|---|---|---|
| ResNet18 | 11.71M | ~1.8G |
| VGG16 | 15.3M | ~15.4G |
| MSCA-VGG16 | 26.8M | ~15.9G |

### To train

```bash
python scripts/train.py \
    --config configs/default.yaml \
    --exp-config experiments/exp1/resnet18.yaml
```

After training, regenerate tables:
```bash
python scripts/analyze_results.py --trial-seeds 42,123,456
```

Made changes.