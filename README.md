# CrowdAnalytix Propensity to Fund Mortgages
## Spis treści:
- [Dlaczego Deep Learning](#dlaczego-deep-learning)
- [Preprocessing](#preprocessing)
    - [Normalizacja](#normalizacja)
    - [Obsługa **nan**ów](#obuga-nanw)
    - [Kategoryzacja (wstęp do Embeddings)](#kategoryzacja)
- [Przygotowanie validacyjnego setu](#przygotowanie-valid-setu)
    - [Różnice w wynikach "lokalnie" i na portalu](#rnice-lokalnie-vs-serwis)
    - [Dlaczego mój model miał kiepskie wyniki](#dlaczego-kiepskie-wyniki)
    - [Dobre pomysły na przyszłość](#dobre-pomysy)
        - [Feature importance](#feature-importance)

- [Modele które zawiodły](#modele-ktre-zawiody)
    - CNN
    - Linear (150,BN,200,DPO,2)
- [Model który zadziałał](#model-ktry-zadziaa)
    - Linear w. Embedding
- [Co to Embedding?](#co-to-embedding)
- [Unsupervised Learning](#unsupervised-learning)
    - Realny wpływ na wynik


## Dlaczego Deep Learning
- Prawie zerowy Feature engineering (shannon entropia informacji)
- Chciałem sprawdzić czy w danych ustrukturyzowanych deep learning się sprawdzi.
## Preprocessing
### Normalizacja
- PCT Change
- X- mean / Deviation
### Obługa Nanów
![Fill Missing][fill_missing]
```python
from statistics import mean

train_data = {
    'Column_A': [1, 2, None, 3, 4, 5, None, None, 6, 7],
    'Column_B': [0.1, 0.4, 0.6, None, 0.77, None, None, 1, 0.9, 0]
}

valid_data = {
    'Column_A': [4, 3, None, 2, 1, 0],
    'Column_B': [0.1, 0.4, 0.6, None, None, None]
}


class FillMissing:
    def __init__(self, data_frame: dict):
        self.column_properties = {col: self.extract_properties(vals) for
                                  col, vals in data_frame.items()}

    def process_data(self, data_frame: dict) -> dict:
        data = {}
        for column in data_frame:
            if self.column_properties[column]['contains_nones']:
                data[f'is_{column}'] = [1 if l else 0 for l in
                                        data_frame[column]]
                data[column] = [self.column_properties[column]['average']
                                if not val else val for val in
                                data_frame[column]]
            else:
                data[column] = data_frame[column]
        return data

    @staticmethod
    def extract_properties(data: list) -> dict:
        length = len(data)
        number_of_nones = len([item for item in data if not item])
        values = [item for item in data if item]
        average = mean(values)
        return {'length': length, 'number_of_nones': number_of_nones,
                'average': average, 'contains_nones': number_of_nones > 0}

```
### Kategoryzacja
```python
import pandas as pd
col_a = 'cat_able_a'
col_b = 'cat_able_b'
embed_szs = []
df = pd.DataFrame()
df[col_a] = [1,2,3,2,2,2,1,3]
df[col_b] = ["dd","aa","dd","aa","dd","ee","dd","ee"]
for col in (col_a, col_b):
    df.loc[:, col] = df.loc[:, col].astype('category').cat.as_ordered()
    embed_szs.append((col, len(df[col].cat.categories))
# [('cat_able_a', 3), ('cat_able_b', 3)]
```
## Przygotowanie valid setu
### Różnice lokalnie vs serwis
![valid_kaggle]
### Dlaczego kiepskie wyniki
- kiepski valid set
### Dobre pomysły
#### [Feature Importance][feat_importance]
![feat_imp]


## Modele które zawiodły
- CNN - za mało danych
![koles_skacze]


- Linear (150,BN,200,DPO,2)


## Model który zadziałał
![model_arch]
(200, 200, 400, 800, 800, 1000, 1000, 600, 600, 400, 200, 200, 100, 2)
## Co to Embedding?
![embed_viz]

[LifeExample][embed_url]

## Unsupervised Learning
- Realny wpływ na wynik
58,05% --> 59.1%







[crowd_analytix]: https://www.crowdanalytix.com/contests/propensity-to-fund-mortgages
[embed_url]: https://embeddings.macheads101.com/
[feat_importance]: https://towardsdatascience.com/feature-selection-techniques-in-machine-learning-with-python-f24e7da3f36e
[fill_missing]: docs/fillmissing.png "Fill Missing Mechanism"
[embed_viz]: docs/embed_viz.png "Embedding visualization"
[model_arch]:docs/model_arch.png "Model Architecture"
[valid_kaggle]: docs/valid_kaggle.png
[feat_imp]:docs/feat_imp.png
[koles_skacze]:docs/action.jpg
