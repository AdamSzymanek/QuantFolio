import pandas as pd

# Ustaw prawidłową ścieżkę do pliku (np. 'src/data/all_stocks_5yr.csv')
input_path = 'data/all_stocks_5yr.csv' 
output_path = 'data/all_stocks_5yr_small.csv'

# 1. Wczytaj dane
df = pd.read_csv(input_path)

# 2. Zostaw tylko co 10-ty wiersz (slicing ::10)
df_small = df.iloc[::10]

# 3. Nadpisz plik lub zapisz jako nowy (tu zapisujemy jako nowy dla bezpieczeństwa)
df_small.to_csv(output_path, index=False)

print(f"Gotowe! Zmniejszono z {len(df)} do {len(df_small)} wierszy.")