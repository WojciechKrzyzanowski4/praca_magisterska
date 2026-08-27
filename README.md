# Porównanie metod gradientowych i ewolucyjnych

Repozytorium zawiera implementację oraz komplet zapisanych artefaktów
eksperymentów wykonanych na potrzeby pracy magisterskiej. Badania obejmują
regresję syntetyczną, klasyfikację MNIST oraz neuroewolucyjne sterowanie
agentami w grach Snake i Asteroids.

## Zawartość

- `networks/` — własna implementacja warstw sieci, funkcji strat i optymalizatorów;
- `mixed/neuro_grad/` — porównanie spadku gradientowego z metodą Monte Carlo;
- `mixed/minst/` — eksperymenty klasyfikacji MNIST metodami gradientowymi,
  ewolucyjnymi i hybrydowymi;
- `evolution/snake/` — trening i ewaluacja agenta Snake;
- `evolution/asteroids/` — trójfazowy trening curriculum agenta Asteroids;
- `media/` — nagrania prezentujące wytrenowanych agentów;
- katalogi `output/` — zapisane genomy, checkpointy, tabele, wykresy i manifesty
  końcowych eksperymentów.

Nazwa pakietu `mixed/minst` jest zachowana zgodnie ze strukturą wykorzystaną
w eksperymentach.

## Wymagania

Kod był uruchamiany w Pythonie 3.13. Zależności można zainstalować poleceniami:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Polecenia należy wykonywać z katalogu głównego repozytorium. Zbiór MNIST jest
dołączony w `mixed/minst/data/mnist.npz`; w przypadku jego braku aplikacja
pobierze go automatycznie z publicznego źródła Keras.

## Uruchamianie

Interaktywne aplikacje regresji i MNIST:

```bash
streamlit run mixed/neuro_grad/app.py
streamlit run mixed/minst/app.py
```

Gry i wizualizacja zapisanych agentów:

```bash
python -m evolution.snake play
python -m evolution.snake visualize
python -m evolution.asteroids play
python -m evolution.asteroids visualize --phase 3
```

Pełne eksperymenty od inicjalizacji modeli:

```bash
python -m mixed.neuro_grad.experiment
python -m mixed.minst.experiment
python -m evolution.snake.experiment
python -m evolution.asteroids.experiment
```

Pełne treningi są kosztowne obliczeniowo i nadpisują odpowiadające im pliki w
katalogach `output/`. Do samej prezentacji agentów służą dołączone genomy.

## Nagrania

- [Snake — długi przebieg agenta](media/snake_dlugi_agent.mp4)
- [Asteroids — ukończenie pięciu fal](media/asteroids_zwyciestwo.mp4)
