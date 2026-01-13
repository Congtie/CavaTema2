# Scooby-Doo Face Detection and Recognition System

## Descriere

Acest proiect implementează un sistem de detectare și recunoaștere facială pentru personajele din serialul Scooby-Doo, folosind o abordare clasică de Computer Vision:

- **Task 1**: Detectarea tuturor fețelor din imagini
- **Task 2**: Recunoașterea celor 4 personaje principale: Fred, Daphne, Shaggy, Velma

## Pipeline-ul Implementat

1. **Sliding Window** - Parcurgerea imaginii cu ferestre de dimensiuni diferite
2. **HOG Feature Extraction** - Extragerea trăsăturilor Histogram of Oriented Gradients
3. **SVM Classifier** - Clasificator Linear SVM pentru detectare și recunoaștere
4. **Non-Maximum Suppression** - Eliminarea detecțiilor suprapuse

## Cerințe

```
numpy>=1.19.0
opencv-python>=4.5.0
scikit-image>=0.18.0
scikit-learn>=0.24.0
tqdm>=4.60.0
matplotlib>=3.3.0
Pillow>=8.0.0
```

## Instalare

```bash
pip install -r requirements.txt
```

## Utilizare

### Rularea completă (antrenare + inferență)

```bash
python RunProject.py
```

### Rularea pe un folder specific de imagini

```bash
python RunProject.py --input path/to/images --output path/to/output
```

### Moduri de rulare disponibile

```bash
# Doar antrenare
python src/run_project.py --mode train

# Rulare pe setul de validare
python src/run_project.py --mode validate

# Rulare pe setul de test
python src/run_project.py --mode test --input testare

# Vizualizare detecții pe o imagine
python src/run_project.py --visualize path/to/image.jpg
```

## Structura Proiectului

```
├── src/
│   ├── config.py           # Configurări și parametri
│   ├── data_loader.py      # Încărcarea și pregătirea datelor
│   ├── feature_extraction.py # Extragerea trăsăturilor HOG
│   ├── classifier.py       # Clasificatoarele SVM
│   ├── sliding_window.py   # Detecție multi-scală
│   ├── nms.py              # Non-Maximum Suppression
│   └── run_project.py      # Pipeline-ul principal
├── models/                 # Modele antrenate (generate automat)
├── RunProject.py           # Script principal de rulare
├── requirements.txt        # Dependențe Python
└── README.txt              # Acest fișier
```

## Output

Rezultatele sunt salvate în format `.npy` compatibil cu scriptul de evaluare:

### Task 1 (Detectarea fețelor)
- `detections_all_faces.npy` - Coordonatele bounding box-urilor [xmin, ymin, xmax, ymax]
- `scores_all_faces.npy` - Scorurile de încredere pentru fiecare detecție
- `file_names_all_faces.npy` - Numele fișierelor imagine

### Task 2 (Recunoaștere personaje)
Pentru fiecare personaj (fred, daphne, shaggy, velma):
- `detections_{character}.npy`
- `scores_{character}.npy`
- `file_names_{character}.npy`

## Parametri Configurabili

În fișierul `src/config.py` se pot modifica:

- `WINDOW_SIZE` - Dimensiunea ferestrei pentru patch-uri (default: 64x64)
- `HOG_ORIENTATIONS` - Numărul de orientări pentru HOG (default: 9)
- `HOG_PIXELS_PER_CELL` - Pixeli per celulă HOG (default: 8x8)
- `SCALE_FACTOR` - Factorul de scalare pentru piramida de imagini (default: 1.2)
- `STEP_SIZE` - Pasul pentru sliding window (default: 8)
- `NMS_THRESHOLD` - Pragul IoU pentru NMS (default: 0.3)
- `SVM_C` - Parametrul de regularizare pentru SVM (default: 1.0)

## Autor

Student - Proiect CAVA 2025
