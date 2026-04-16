# Vizualizácia metód vytvárania rozhodovacích stromov

Interaktívna webová aplikácia a edukačná animácia vytvorené ako praktická časť bakalárskej práce na tému vizualizácie rozhodovacích stromov.

## O projekte

Projekt pozostáva z dvoch častí:

- **Interaktívna webová aplikácia** vytvorená v prostredí Streamlit, ktorá umožňuje experimentovať s parametrami rozhodovacieho stromu na dvoch datasetoch (vlastný dataset o úveroch a dataset Titanic).
- **Edukačná animácia** vytvorená v Adobe After Effects, ktorá krok za krokom vysvetľuje postup budovania rozhodovacieho stromu vrátane výpočtu Giniho indexu a binárneho rozdeľovania v algoritme CART.

## Použité technológie

- Python 3.14
- Streamlit
- scikit-learn
- pandas
- numpy
- matplotlib

## Spustenie lokálne

```bash
pip install -r requirements.txt
streamlit run main.py
```

## Online spustenie

Aplikácia je dostupná online: [https://rozhodovacie-stromy.streamlit.app](https://rozhodovacie-stromy.streamlit.app)

## Autor

Daniel Turančík  
Univerzita Konštantína Filozofa v Nitre  
Fakulta prírodných vied a informatiky  
2026
