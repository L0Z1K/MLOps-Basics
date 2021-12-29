## Week 0: PyTorch Lightning

### Requirements

```bash
conda create --name project-setup python=3.8
conda activate project-setup
pip install -r requirements.txt
```

### Train & Infer

```bash
python train.py
```

```bash
python infer.py
Input the sentence for inference: A girl is eating the candy.
[{'label': 'unacceptable', 'score': 0.30912938714027405}, {'label': 'acceptable', 'score': 0.6908706426620483}]
```