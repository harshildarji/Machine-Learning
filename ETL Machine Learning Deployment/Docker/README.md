#### Breast Cancer Prediction
**Note:** I have not built a web interface for this, so use **Postman** to request APIs.

**Docker:**
```bash
docker build --tag breast-cancer-predict:latest .
docker run --name breast-cancer -d -p 5000:5000 breast-cancer-predict  
```
**API:**
1. `[GET]` localhost:5000
2. `[POST]` localhost:5000/predict

**Example JSON request:**
```JSON
{
    "data": [
        {
            "mean radius": 20.6,
            "mean texture": 29.33,
            "mean perimeter": 140.1,
            "mean area": 1265.0,
            "mean smoothness": 0.1178,
            "mean compactness": 0.277,
            "mean concavity": 0.3514,
            "mean concave points": 0.152,
            "mean symmetry": 0.2397,
            "mean fractal dimension": 0.07016,
            "radius error": 0.726,
            "texture error": 1.595,
            "perimeter error": 5.772,
            "area error": 86.22,
            "smoothness error": 0.006522,
            "compactness error": 0.06158,
            "concavity error": 0.07117,
            "concave points error": 0.01664,
            "symmetry error": 0.02324,
            "fractal dimension error": 0.006185,
            "worst radius": 25.74,
            "worst texture": 39.42,
            "worst perimeter": 184.6,
            "worst area": 1821.0,
            "worst smoothness": 0.165,
            "worst compactness": 0.8681,
            "worst concavity": 0.9387,
            "worst concave points": 0.265,
            "worst symmetry": 0.4087,
            "worst fractal dimension": 0.124
        }
    ]
}
```
