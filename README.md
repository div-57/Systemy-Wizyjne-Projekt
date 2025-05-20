# Systemy-Wizyjne-Projekt

## Opis projektu
Projekt służy do automatycznego rozpoznawania znaków tablic rejestracyjnych na zdjęciach pojazdów. Wykorzystuje bibliotekę OpenCV do przetwarzania obrazu oraz klasyfikator SVM (Support Vector Machine) ze scikit-learn do rozpoznawania wydzielonych znaków.

## Wymagania
- Python 3.8 lub nowszy  
- OpenCV (`cv2`)  
- NumPy  
- scikit-learn  
- Pliki z danymi treningowymi w katalogu `train` (obrazki pojedynczych znaków `.jpg`)  

Instalacja zależności:
```bash
pip install -r requirements.txt
```

## Struktura repozytorium
```graphql
.
├── img/                   # Przykładowe obrazy tablic do przetwarzania
├── processing/
│   └── utils.py           # Funkcje do wczytywania danych, trenowania i przetwarzania obrazu
├── train/                 # Katalog z danymi treningowymi (obrazy pojedynczych znaków)
├── Wozniak_Dawid.py       # Główny skrypt uruchomieniowy
├── requirements.txt       # Lista zależności
└── results.json           # Przykładowy plik wynikowy
```

## Uruchomienie

W katalogu głównym projektu:
```bash
python Wozniak_Dawid.py <ścieżka_do_katalogu_z_obrazami> <ścieżka_do_pliku_wyników.json>
```
Przykład:
```bash
python Wozniak_Dawid.py img/ output/results.json
```

## Opis działania

#### 1. Wczytanie danych treningowych
- ```load_training_data('train')``` — wczytuje obrazy pojedynczych znaków, przeskalowuje je do rozmiaru 70×100 px i spłaszcza do wektora cech.

#### 2. Trenowanie klasyfikatora
- ```train_model()``` — trenuje klasyfikator SVM na przygotowanych wektorach cech.

#### 3. Przetwarzanie obrazów wejściowych
- Skalowanie — zmniejszenie obrazu do 35% oryginału,
- Konwersja na skalę szarości i rozmycie Gaussa,
- Binaryzacja (```cv2.threshold```) i detekcja krawędzi (```cv2.Canny```),
- Detekcja konturów — wybór 5 największych konturów, aproksymacja wielokąta (```cv2.approxPolyDP```) w celu znalezienia narożników tablicy,
- Transformacja perspektywiczna (```cv2.getPerspectiveTransform``` + ```cv2.warpPerspective```) — wyrównanie obrazu tablicy,
- Segmentacja znaków — dylatacja, ponowne znajdowanie konturów, filtrowanie według wymiarów i pozycji,
- Rozpoznawanie znaków — każdy wycięty fragment jest przeskalowywany do 70×100 px, spłaszczany i klasyfikowany przez SVM (```svc.predict```).

## Wyniki
Program zapisuje w pliku JSON mapowanie nazw plików na rozpoznane napisy, np.:
```bash
{
  "nazwa_pliku1.jpg": "AB123CD",
  "nazwa_pliku2.jpg": "XYZ7890",
  …
}
```

*Autor: Woźniak Dawid*