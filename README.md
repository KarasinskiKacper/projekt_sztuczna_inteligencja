# Projekt rozpoznawania tekstu na zdjęciach

<h2>Autorzy projektu:</h2>
- Kacper Karasiński<br>
- Paweł Herzyk<br>
- Paweł Dyczek<br>
- Mikołaj Całus<br>

<h2>O projekcie</h2>
W projekcie została stworzona aplikacja webowa wykorzystująca bibliotekę Flask, w której zaimplementowano modele konwolucyjnych sieci neuronowych do rozpoznawania tekstu na obrazach. Po przesłaniu pliku ze zdjęciem można wybrać 1 z 4 przygotowanych modeli do rozpoznania tekstu. Następnie po kliknięciu "Wyślij", aplikacja wykona analizę zdjęcia wybranym modelem i wypisze na ekranie rozpoznany tekst. W celu trenowania modeli, została wygenerowana baza danych zawierająca 5mln zdjęć z polskimi słowami (skrypt do wygenerowania bazy danych - ./generate_db/generate_data.py).

Zdjęcia ze słowami były generowane z wykorzystaniem repozytorium: [text_renderer](https://github.com/oh-my-ocr/text_renderer).

W projekcie został wykorzystany gotowy model służący do znajdywania położenia tekstu na zdjęciu: [CRAFT-pytorch](https://github.com/clovaai/CRAFT-pytorch).

<h2>Wymagania</h2>
<h3>Aplikacja webowa</h3>
- Python 3.12<br>
- Biblioteki wymienione w `requirements.txt`<br>

<h3>Generowanie bazy danych</h3>
- Python 3.8<br>
- Biblioteki wymienione w `./generate_db/requirements.txt`<br>

<h2>Uruchomienie aplikacji webowej</h2>

```
git clone https://github.com/KarasinskiKacper/projekt_sztuczna_inteligencja.git [nazwa folderu]
cd [nazwa folderu]
pip install -r requirements.txt
python main.py
```

serwer aplikacji webowej wystartuje pod adresem http://127.0.0.1:5000

<h2>Uruchomienie skryptu do generowania bazy danych</h2>
<b>Zainstalować pythona 3.8</b>

```
git clone https://github.com/KarasinskiKacper/projekt_sztuczna_inteligencja.git [nazwa folderu]
cd [nazwa folderu]
cd generate_db
git clone https://github.com/oh-my-ocr/text_renderer.git text_renderer
cd text_renderer
pip install -e .
cd ..
pip install -r requirements.txt
python generate_data.py
```

<b>Uwaga baza danych zajmuje 64GB!</b>
