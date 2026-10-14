# Online Purchase Prediction

Streamlit application for predicting whether a website visitor will purchase
an item (`purchased = 1`) from browsing behavior.

## Run locally

```powershell
pip install -r requirements.txt
streamlit run app.py
```

The application loads the trained model from
`artifacts/best_model.joblib`. Keep the `artifacts` folder in the repository.

## Deploy with Streamlit Community Cloud

1. Push `app.py`, `requirements.txt`, `README.md`, and the `artifacts` folder
   to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io).
3. Select the `Kwansiri12/Shopping` repository.
4. Set the main file to `app.py`.
5. Deploy the application.

## Input features

- `pages_viewed`: number of pages viewed
- `session_minutes`: session duration in minutes
- `items_viewed`: number of items viewed
- `discount_seen`: `Yes` or `No`
- `returning_visitor`: `Yes` or `No`
- `device`: `Desktop`, `Mobile`, or `Tablet`

The current best model is SVM RBF. Model metrics and the comparison table are
loaded from the JSON and CSV files in `artifacts`.
