## Week 4: Model Packaging - ONNX

ONNX를 이용하면 Model Dependency와 관계없이 Model을 사용할 수 있다. ONNX를 사용해서 모델을 배포할 줄 알아야 나중에 Production Level에서 실제 내 모델을 세상에 배포할 수 있을 것 같다.

### Convert model to ONNX Format

```bash
python convert_model_to_onnx.py
```

### Inference with ONNX

```bash
python infer.py
```