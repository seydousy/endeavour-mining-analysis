# Endeavour Mining — Analytics Dashboard

Dashboard interactif d'analyse des données d'Endeavour Mining (2024-2025).

## Installation & Lancement (Ubuntu)

### 1. Ouvrir un terminal dans ce dossier
```bash
cd edv_dashboard
```

### 2. Créer un environnement virtuel (recommandé)
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 4. Lancer le dashboard
```bash
streamlit run app.py
```

### 5. Ouvrir dans le navigateur
Le terminal affiche : `Local URL: http://localhost:8501`
Ouvre Firefox ou Chrome → http://localhost:8501

---

## Contenu du Dashboard

| Section | Analyses |
|---|---|
| 🏠 Vue Générale | KPIs groupe, production 2024 vs 2025 |
| ⛏️ Opérationnel | Grade vs production, recovery rate, strip ratio, quarterly |
| 💰 Financier | AISC par mine, revenue, free cash flow trimestriel |
| 🪨 Réserves | Mine life, qualité gisements, risque épuisement |
| 🤖 Prédiction ML | Régression, forecast 2026, comparaison vs guidance |
| 💡 Capex & Stratégie | Allocation capex, exploration → réserves |

## Sources
- Endeavour Mining Annual Report 2025
- Q3-2025 et Q4-2025 Operations Reports
- Mineral Reserves & Resources Statement 2025
