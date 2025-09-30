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
