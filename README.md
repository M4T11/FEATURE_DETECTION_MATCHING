
# Środowisko badawcze

Środowisko badawcze zostało zaimplementowane w celu przetestowania wydajności i przydatności algorytmów wykrywania i metod dopasowywania na potrzeby zbudowania aplikacji działającej w systemie rozszerzonej rzeczywistości (AR). 

Wybrane algorytmy wykrywania i opisywania obiektów w obrazie:
- Scale Invariant Feature Transform (SIFT),
- Speeded-Up Robust Features (SURF),
- Features from Accelerated Segment Test (FAST),
- Oriented FAST and Rotated BRIEF (ORB).

Wybrane metody dopasowywania deskryptorów:
- metoda Brute-Force,
- metoda oparta na FLANN.

W celu poprawy jakości testów, na etapie dopasowywania deskryptorów zastosowano mechanizm filtrowania dopasowań (ratio test) pozwalający na znaczną redukcję błędnych dopasowań.

Środowisko umożliwia testowanie powyższych algorytmów oraz metod w różnych kombinacjach

## Wymagania

- Python w wersji 3.7 wraz z zainstalowanymi niezbędnymi bibliotekami.

## Funkcjonalności

| SKRYPT  | FUNKCJONALNOŚĆ |
| ------------- |:-------------:|
| feature_detection_matching_images.py     | Skrypt umożliwia przeprowadzenie badań pod kątem wykrywania i dopasowywania obiektów obrazach cyfrowych.|
| feature_detection_live.py     | Skrypt jest narzędziem umożliwiającym zarejestrowanie i zapis do pliku deskryptorów wybranego obiektu na obrazie z kamery lub pliku wideo.    |
| feature_detection_matching_live.py      | Skrypt umożliwia przeprowadzenie badań dopasowywania wybranego obiektu na obrazie z kamery lub pliku wideo (do tego celu wykorzystuje się również drugi skrypt, w którym wcześniej można utworzyć i zapisać deskryptory dla wybranego obiektu).     |

## Sposób użycia

- Skrypt `feature_detection_matching_images.py`

| FUNKCJA  | OPIS DZIAŁANIA |
| ------------- |:-------------:|
| `test_img(img_set)`<br><br>Argumenty funkcji:<br>`img_set` - ścieżka do zestawu obrazów    |  Funkcja pozwala zbadać skuteczność rejestrowania deskryptorów obiektów w obrazach z użyciem wybranych algorytmów wykrywania i opisywania obiektów w obrazie.<br>**Kryteria testu:** <br> -  ilość deskryptorów zdefiniowanych dla badanego obrazu (obiektu), <br>- czas wykrycia pojedynczego punktu charakterystycznego,<br>- czas utworzenia deskryptora dla pojedynczego punktu charakterystycznego, <br>- całkowity czas wygenerowania deskryptorów, <br>- całkowity czas wygenerowania punktów charakterystycznych <br>- całkowity czas działania algorytmów (łączny czas niezbędny na wykrycie punktów charakterystycznych oraz utworzenia dla nich deskryptorów). <br><br> <br>  Wyniki badań zapisywane są do pliku .txt|
| `test_img_matching(img_1_path, img_2_path, matching_algorithm)` <br><br>Argumenty funkcji:<br>`img_1_path` - ścieżka do obrazu referencyjnego <br>`img_2_path` - ścieżka do innego obrazu obiektu<br>`matching_algorithm`  - wybór metody dopasowywania deskryptorów ('BRUTE-FORCE' lub 'FLANN')   | Funkcja pozwala zbadać skuteczność dopasowywania deskryptorów obiektów na pojedynczych obrazach.  <br>**Kryteria testu:** <br> - liczba dopasowań deskryptorów, <br>- procent dopasowań deskryptorów,<br>- całkowity czas niezbędny do dopasowywania deskyptorów, <br><br>   Wyniki badania zapisywane są do pliku .txt   |
| `group_of_descriptors(img_to_find_path, algorithm, matching_method)` <br><br>Argumenty funkcji:<br>`img_to_find_path` - ścieżka do obrazu szukanego obiektu <br>`algorithm` - wybór algorytmu wykrywania i opisywania obiektów w obrazie ('SIFT', 'SURF', 'FAST_SURF', 'ORB')<br>`matching_method`  - wybór metody dopasowywania deskryptorów ('BRUTE-FORCE' lub 'FLANN')     | Funkcja pozwala sprawdzić skuteczność dopasowywania obiektu względem różnego położenia obserwatora.<br> Badanie polegało na zdefiniowaniu na podstawie czterech różnych ujęć - wykonanych z czterech narożników tego samego obiektu - zestawu deskryptorów dla badanego obiektu. Następnie, do zdefiniowanej w ten sposób listy deskryptorów badanego obiektu, próbowano dopasować deskryptory utworzone dla innego obrazu tego samego obiektu (argument `img_to_find_path`), wykonanego z odmiennego niż uprzednio, położenia obserwatora   <br><br>   Wyniki badania zapisywane są do pliku .txt  |

- Skrypt `feature_detection_live.py`

| KLASA  | OPIS DZIAŁANIA |
| ------------- |:-------------:|
| Konstruktor klasy `feature_detection_ROI`<br><br>Argumenty konstruktora:<br>`(opcjonalny) path_to_video` - ścieżka do pliku wideo (domyślnie jest podana); w przypadku chęci korzystania z kamery należy podać '0',    |    Klasa jest narzędziem, które umożliwia zarejestrowanie i zapis do pliku deskryptorów wybranego obiektu na obrazie z kamery lub pliku wideo. <br>**Obsługa skryptu:** <br> -  przycisk 'c' -  zatrzymuje plik wideo i pozwala zaznaczyć interesujący obszar w klatce wideo, przy kolejnym naciśnięciu przycisku 'c' w konsoli należy wybrać jeden z czterech algorytmów wykrywania i opisywania obiektów w obrazie, który ma zostać użyty. W konsoli należy również wprowadzić opis dla obiektu <br> - przycisk 'r' - wznawia odtwarzanie pliku wideo<br><br> <br>  Zarejestrowane deskryptory są zapisywane do pliku .pickle, natomiast obraz zaznaczonego obiektu zapisywany jest w dedykowanym dla danego algorytmu katalogu.|

- Skrypt `feature_detection_matching_live.py`

| FUNKCJA  | OPIS DZIAŁANIA |
| ------------- |:-------------:|
|  `start(path_to_video, image_template)`<br><br>Argumenty funkcji:<br>`(opcjonalny) path_to_video` - ścieżka do pliku wideo (domyślnie jest podana); w przypadku chęci korzystania z kamery należy podać '0', <br>` image_template` - ścieżka do pliku obrazu obiektu, którego deskryptory mają być dopasowywane do deskryptorów fragmentów kolejnych klatek pliku wideo  (domyślnie jest podana),    |  Funkcja pozwala sprawdzić skuteczność dopasowywania obiektu w pliku wideo. Badanie polega na próbie dopasowywania deskryptorów zdefiniowanych dla wybranego obiektu na obrazie do deskryptorów zdefiniowanych dla kolejnych fragmentów klatek pliku wideo, w celu dopasowania (wyszukania) tego obiektu na filmie w czasie rzeczywistym <br>**Obsługa skryptu:** <br> -  po uruchomieniu skryptu należy w konsoli wybrać jeden z czterech algorytmów wykrywania i opisywania obiektów w obrazie. W momencie gdy skrypt dopasuje co najmniej 20 deskryptorów obiektu do deskryptorów zdefiniowanych dla fragmentu klatki zostanie wyświetlony stosowny komunikat oraz w dodatkowym oknie zostanie wyświetlony podgląd dopasowania. Ponadto obraz podglądu dopasowania deskryptorów zostaje również zapisany w dedykowanym dla danego algorytmu katalogu. <br> -  przycisk 'p' wywołuje dodatkową funkcję, której zadaniem jest dopasowywanie deskryptorów wszystkich zapisanych obiektów dla danego algorytmu (z użyciem skryptu `feature_detection_live.py`) do deskryptorów zdefiniowanych dla danej klatki wideo
## Autor
- Mateusz GALAN

