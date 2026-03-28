# deep_learning_project
Comparer la performance en classification d’images des CNNs (Krizhevsky et al., 2012) et des Vision Transformers (Lu et al., 2022). Vous devez concevoir des jeux de donnees qui mettent en evidence les differences de performance. 

projet_deep/
│
├── notebooks/
│   ├── 01_baseline_cnn.ipynb   # CNN baseline experiments
│   ├── 02_vit.ipynb            # Vision Transformer experiments
│   └── 03_robustness.ipynb     # Robustness & evaluation analysis
│
├── src/
│   ├── models.py               # Model architectures (CNN & ViT)
│   ├── train.py                # Training pipeline
│   ├── eval.py                 # Evaluation & metrics
│   └── data.py                 # Data loading & preprocessing
│
├── configs/
│   ├── cnn.yaml                # CNN training configuration
│   └── vit.yaml                # ViT training configuration
│
├── outputs/
│   ├── models/                 # Saved trained models
│   └── logs/                   # Training logs & results
│
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── .gitignore                  # Ignored files