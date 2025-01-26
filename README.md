# Projekt rozpoznawania tekstu na zdjęciach

<h2>Autorzy projektu:</h2>
- Kacper Karasiński
- Paweł Herzyk
- Paweł Dyczek
- Mikołaj Całus

<h2>O projekcie</h2>
W projekcie została stworzona aplikacja webowa wykorzystująca bibliotekę Flask, w której zaimplementowano modele konwolucyjnych sieci neuronowych do rozpoznawania tekstu na obrazach. Po przesłaniu pliku ze zdjęciem można wybrać 1 z 4 przygotowanych modeli do rozpoznania tekstu. Następnie po kliknięciu "Wyślij", aplikacja wykona analizę zdjęcia wybranym modelem i wypisze na ekranie rozpoznany tekst. W celu trenowania modeli, została wygenerowana baza danych zawierająca 5mln zdjęć z polskimi słowami (skrypt do wygenerowania bazy danych - ./generate_db/generate_data.py).

W projekcie został wykorzystany gotowy model służący do znajdywania położenia tekstu na zdjęciu: [CRAFT-pytorch](https://github.com/clovaai/CRAFT-pytorch)

<h2>Wymagania</h2>
<h3>Aplikacja webowa</h3>
- Python 3.12<br>
- Biblioteki wymienione w `requirements.txt`<br>

<h3>Generowanie bazy danych</h3>
- Python 3.8<br>
- Biblioteki wymienione w `./generate_db/requirements.txt`<br>

<h2>Uruchomienie aplikacji webowej</h2>
git clone https://github.com/KarasinskiKacper/projekt_sztuczna_inteligencja.git [nazwa folderu]<br>
cd [nazwa folderu]<br>
pip install -r requirements.txt<br>
python main.py<br>
serwer aplikacji webowej wystartuje pod adresem http://127.0.0.1:5000<br>

<h2>Uruchomienie skryptu do generowania bazy danych</h2>
***Zainstalować pythona 3.8***<br>
git clone https://github.com/KarasinskiKacper/projekt_sztuczna_inteligencja.git [nazwa folderu]<br>
cd [nazwa folderu]<br>
cd generate_db<br>
pip install -r requirements.txt<br>
python generate_data.py<br>
***Uwaga baza danych zajmuje 64GB!***<br>
