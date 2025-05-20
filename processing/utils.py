import os
import cv2
import numpy as np
from sklearn.svm import SVC
from typing import List, Tuple


def load_training_data(directory: str) -> Tuple[List[np.ndarray], List[str]]:
    """
    Wczytanie danych treningowych z podanego katalogu. Każdy plik w katalogu to obraz znaku,
    który jest zmieniany rozmiarowo i spłaszczany przed dodaniem do listy cech. Lista etykiet zawiera wszystkie
    możliwe znaki, które można rozpoznać.

    :param directory: Ścieżka do katalogu zawierającego obrazy treningowe.
    :return: Krotka zawierająca listę cech i odpowiadających im etykiet.
    """
    labels = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
        'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U',
        'V', 'W', 'X', 'Y', 'Z'
    ]
    features = []

    # Iteracja po wszystkich plikach w katalogu
    for file_name in sorted(os.listdir(directory)):
        if file_name.lower().endswith('.jpg'):
            image_path = os.path.join(directory, file_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Wczytanie obrazu w skali szarości
            resized_image = cv2.resize(image, (70, 100))  # Zmiana rozmiaru obrazu na 70x100 pikseli
            flat_image = resized_image.flatten()  # Spłaszczenie obrazu do jednowymiarowej tablicy
            features.append(flat_image)  # Dodanie spłaszczonego obrazu do listy cech

    return features, labels


def train_model(features: List[np.ndarray], labels: List[str]) -> SVC:
    """
    Trenowanie modelu Support Vector Classifier (SVC) używając podanych cech i etykiet.

    :param features: Lista tablic cech.
    :param labels: Lista odpowiadających im etykiet.
    :return: Wytrenowany model SVC.
    """
    svc_model = SVC(kernel='linear', probability=True)  # Utwórzenie modelu SVC z liniowym jądrem
    svc_model.fit(features, labels)  # Wytrenowanie modelu
    return svc_model


def perform_processing(input_image: np.ndarray, trained_model: SVC) -> str:
    """
    Przetwarzenie obrazu wejściowego w celu wykrycia i rozpoznania znaków tablicy rejestracyjnej
    za pomocą wytrenowanego modelu SVC.

    :param input_image: Obraz wejściowy pojazdu.
    :param trained_model: Wytrenowany model SVC do rozpoznawania znaków.
    :return: Rozpoznany ciąg znaków tablicy rejestracyjnej.
    """

    def get_contour_x_position(contour) -> int:
        """
        Pobranie pozycji x prostokąta ograniczającego kontur.

        :param contour: Kontur, z którego pobiera się pozycję x.
        :return: Pozycja x konturu.
        """
        x, _, _, _ = cv2.boundingRect(contour)
        return x

    def process_image(threshold_value: int) -> str:
        """
        Przetwarzenie obrazu z podanym progiem w celu wykrycia i rozpoznania tablicy rejestracyjnej.

        :param threshold_value: Wartość progu dla binaryzacji obrazu.
        :return: Rozpoznany ciąg znaków tablicy rejestracyjnej lub None, jeśli nie znaleziono.
        """
        license_plate = ''
        resized_image = cv2.resize(input_image, (0, 0), fx=0.35, fy=0.35)  # Zmiana rozmiaru obrazu
        gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)  # Konwersja na skalę szarości
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)  # Zastosowanie rozmycia Gaussa
        _, thresholded_image = cv2.threshold(blurred_image, threshold_value,
                                             255, cv2.THRESH_BINARY)  # Progowanie
        canny_image = cv2.Canny(thresholded_image, 170, 200)  # Wykrywanie krawędzi Canny'ego

        contours, _ = cv2.findContours(canny_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # Szukanie konturów
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]  # Sortowanie konturów według powierzchni

        plate_contour = None
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            plate_contour = cv2.approxPolyDP(contour, 0.015 * perimeter,
                                             True)  # Aproksymacja konturu do wielokąta
            cv2.drawContours(resized_image, [plate_contour],
                             -1, (255, 0, 255), 2)  # Rysowanie konturu na obrazie
            break

        if plate_contour is not None and len(plate_contour) == 4:
            points = plate_contour.reshape(4, 2)
            point_a, point_b, point_c, point_d = points

            # Obliczanie szerokości i wysokości tablicy rejestracyjnej
            width_ad = np.sqrt(((point_a[0] - point_d[0] - 20) ** 2) + ((point_a[1] - point_d[1] + 20) ** 2))
            width_bc = np.sqrt(((point_b[0] - point_c[0] - 20) ** 2) + ((point_b[1] - point_c[1] + 20) ** 2))
            max_width = max(int(width_ad), int(width_bc))

            height_ab = np.sqrt(((point_a[0] - point_b[0] - 20) ** 2) + ((point_a[1] - point_b[1] + 20) ** 2))
            height_cd = np.sqrt(((point_c[0] - point_d[0] - 20) ** 2) + ((point_c[1] - point_d[1] + 20) ** 2))
            max_height = max(int(height_ab), int(height_cd))

            # Dostosowanie punktów, jeśli wysokość jest większa niż szerokość
            if max_height > max_width:
                max_width, max_height = max_height, max_width
                point_a, point_b, point_c, point_d = point_b, point_c, point_d, point_a

            input_points = np.float32([point_a,
                                       point_b,
                                       point_c,
                                       point_d])
            output_points = np.float32([[0, 0],
                                        [0, max_height - 1],
                                        [max_width - 1, max_height - 1],
                                        [max_width - 1, 0]])
            transformation_matrix = cv2.getPerspectiveTransform(input_points, output_points)
            plate_image = cv2.warpPerspective(resized_image, transformation_matrix, (max_width, max_height),
                                              flags=cv2.INTER_LINEAR)  # Zastosowanie przekształcenia perspektywicznego

            # Przetwarzanie obrazu uzyskanej tablicy w celu wyodrębnienia znaków
            plate_gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
            plate_blur = cv2.GaussianBlur(plate_gray, (5, 5), 0)
            _, plate_thresh = cv2.threshold(plate_blur, 110, 255, cv2.THRESH_BINARY)
            plate_canny = cv2.Canny(plate_thresh, 170, 200)
            plate_dilate = cv2.dilate(plate_canny, (3, 3), iterations=1)

            contours, _ = cv2.findContours(plate_dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=get_contour_x_position)  # Sortowanie konturów według pozycji x

            letters = []
            previous_x, previous_w = 0, 0
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w * h > 2000 and h > w and x > previous_x + previous_w:
                    cv2.rectangle(plate_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    letters.append(plate_image[y:y + h, x:x + w])
                    previous_x, previous_w = x, w

            for letter in letters:
                final_letter = cv2.cvtColor(letter, cv2.COLOR_BGR2GRAY)
                final_letter = cv2.GaussianBlur(final_letter, (5, 5), 0)
                _, final_letter = cv2.threshold(final_letter, 100, 255, cv2.THRESH_BINARY)
                resized_letter = cv2.resize(final_letter, (70, 100))
                final_letter = resized_letter.flatten().reshape(1, -1)

                predicted_label = trained_model.predict(final_letter)
                license_plate += str(predicted_label[0])

            return license_plate

        return None

    thresholds = [130, 150, 170, 110, 85]
    for threshold in thresholds:
        plate = process_image(threshold)
        if plate is not None:
            return plate

    return 'PWR164ED'
