# Drain-Brain
This tool helps you find the best countries based on your personal preferences, comparing key factors like jobs, safety, health, and more. Answer a few questions to get personalized recommendations and visualize the best destinations.

## Project Description
This work investigates the migration flows of tertiary-educated individuals, with a particular focus
on OECD countries. The analysis integrates statistical data and socio-economic indicators in order
to:
• perform data preparation and normalization;
• produce exploratory visualizations of migration flows;
• implement interactive tools for cross-country comparison;
• apply clustering methods and dimensionality reduction (PCA) to identify groups of similar
countries;
• develop a recommendation system capable of suggesting suitable destinations based on
user-defined preferences.

## Folder Structure
• Brain_drain.ipynb — the main notebook containing code and analysis.
• Datasets/ — folder with the CSV files required by the notebook.
For proper execution, the Datasets folder must be located in the following Google Drive path:
MyDrive/Data_Science_Lab/Datasets

## Execution Guidelines
1. Open Brain_drain.ipynb in Google Colab.
2. Mount Google Drive when prompted.
3. Ensure that the datasets are stored in the specified directory.
4. Execute the notebook cell by cell (rather than using Run all) to guarantee correct behavior of
interactive widgets.

## Requirements
The notebook is designed for Google Colab, which provides the required libraries by default:
pandas, numpy, matplotlib, seaborn, scikit-learn, plotly, networkx, ipywidgets.




# Web App

📁 **File inclusi nella cartella `Web_app`**

* `app.py` → codice principale dell’app Streamlit
* `dataset_final.csv` → dataset dei flussi migratori e indicatori OCSE
* `requirements.txt` → pacchetti Python necessari

🚀 **Come usare l'app**

1. Clona la repository o scarica i file:

   ```bash
   git clone <url-della-repo>
   cd Brain_drain/Web_app
   ```

2. Assicurati di avere **Python 3.9+** installato.

3. Installa i pacchetti richiesti:

   ```bash
   pip install -r requirements.txt
   ```

4. ⚠️ **Nota:** `pygraphviz` richiede Graphviz installato sul sistema. Su Ubuntu/Debian:

   ```bash
   sudo apt-get install graphviz graphviz-dev
   ```

5. Avvia l'app Streamlit:

   ```bash
   streamlit run app.py
   ```

