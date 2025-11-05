"""
Dash application for analyzing customer opinions.

This app allows users to upload a CSV file containing customer reviews or opinions
about a product, company or place. Once the file is uploaded, the app performs
basic text pre‑processing (lowercasing, removing punctuation, removing Spanish and
English stop words and stemming) and then displays:

1. A word cloud image of the most frequent terms.
2. A bar chart of the top 10 most common words.
3. An additional chart showing the distribution of review lengths (histogram).
4. A sentiment classification for each review (positive, neutral or negative) using
   the VADER sentiment analyser. A table displays the original review and its
   assigned sentiment. A pie chart summarises the distribution of sentiment classes.
5. A text input where users can enter a new comment and receive immediate
   feedback on its predicted sentiment.

The app is built with Dash and uses Plotly for interactive charts. Text
processing relies on NLTK for tokenisation, stop word removal and stemming, and
on the ``vaderSentiment`` library for sentiment analysis. Word clouds are
generated with the ``wordcloud`` library and rendered as base64‑encoded images.
"""

import base64
import io
import os
from dataclasses import dataclass
from typing import List, Tuple

import dash
from dash import Dash, html, dcc, dash_table, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd

# Text processing imports
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud


# Ensure NLTK data is downloaded only once. NLTK will store these in ~/.cache
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


############################################################
# Data classes and helper functions
############################################################

@dataclass
class ProcessedText:
    """Container for processed text information."""
    tokens: List[str]
    frequencies: pd.Series


def preprocess_reviews(texts: List[str]) -> ProcessedText:
    """Preprocess a list of review strings into tokens and frequencies.

    Steps:
    - Lowercase
    - Remove punctuation and non‑alphabetic tokens
    - Remove English and Spanish stop words
    - Apply stemming (Spanish stemmer) to approximate lemmatisation

    Args:
        texts: List of review strings.

    Returns:
        ProcessedText with list of processed tokens and a frequency series.
    """
    # Create stop word lists for both English and Spanish
    stop_en = set(stopwords.words("english"))
    stop_es = set(stopwords.words("spanish"))
    stop_all = stop_en.union(stop_es)

    stemmer = SnowballStemmer("spanish")  # Spanish stemmer is a good proxy for lemmatisation

    tokens: List[str] = []
    for text in texts:
        # Lowercase and tokenize
        words = word_tokenize(str(text).lower())
        for w in words:
            # Keep alphabetic tokens only
            if not w.isalpha():
                continue
            # Remove stop words
            if w in stop_all:
                continue
            # Stem the word
            stemmed = stemmer.stem(w)
            tokens.append(stemmed)
    # Compute frequencies
    freq_series = pd.Series(tokens).value_counts().sort_values(ascending=False)
    return ProcessedText(tokens=tokens, frequencies=freq_series)


def generate_wordcloud(tokens: List[str]) -> Tuple[str, WordCloud]:
    """Generate a word cloud from tokens and return a base64 image string.

    Args:
        tokens: List of processed tokens.

    Returns:
        A tuple of (base64 encoded image string, WordCloud object).
    """
    text = " ".join(tokens)
    wc = WordCloud(width=600, height=400, background_color="white").generate(text)
    # Convert to image and encode as base64
    img_buffer = io.BytesIO()
    wc.to_image().save(img_buffer, format="PNG")
    img_buffer.seek(0)
    encoded = base64.b64encode(img_buffer.read()).decode()
    image_uri = f"data:image/png;base64,{encoded}"
    return image_uri, wc


def classify_sentiments(texts: List[str]) -> List[str]:
    """Classify sentiment for a list of texts using VADER.

    VADER returns a compound score between -1 and 1. We categorise as:
    positive if compound >= 0.05, negative if compound <= -0.05, neutral otherwise.

    Args:
        texts: List of review strings.

    Returns:
        List of sentiment labels ("Positivo", "Negativo", "Neutral").
    """
    analyser = SentimentIntensityAnalyzer()
    labels = []
    for t in texts:
        scores = analyser.polarity_scores(str(t))
        comp = scores["compound"]
        if comp >= 0.05:
            labels.append("Positivo")
        elif comp <= -0.05:
            labels.append("Negativo")
        else:
            labels.append("Neutral")
    return labels


############################################################
# Dash application configuration
############################################################

# External Bootstrap theme for basic styling
external_stylesheets = [dbc.themes.CERULEAN]

app = Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

# Helper layout components
upload_component = dcc.Upload(
    id="upload-data",
    children=html.Div([
        "Arrastra y suelta o ", html.A("selecciona un archivo CSV")
    ]),
    style={
        "width": "100%",
        "height": "60px",
        "lineHeight": "60px",
        "borderWidth": "1px",
        "borderStyle": "dashed",
        "borderRadius": "5px",
        "textAlign": "center",
        "margin": "10px"
    },
    multiple=False
)

# Main layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H3("Análisis de Opiniones de Clientes"), className="mb-4 mt-3 text-center text-primary")
    ]),
    dbc.Row([
        dbc.Col(html.P(
            "Sube un archivo CSV con una columna de opiniones (20 registros). El archivo debe tener al menos"
            " una columna llamada 'review' o 'opinion'."
        ), width=12)
    ]),
    dbc.Row([
        dbc.Col(upload_component, width=12)
    ]),
    # Store components to hold intermediate data across callbacks
    dcc.Store(id="stored-data"),  # Raw dataframe
    dcc.Store(id="stored-processed"),  # Processed text info (frequencies etc.)
    dcc.Store(id="stored-sentiments"),  # Sentiment labels

    dbc.Row([
        # Left column: Word cloud and bar chart
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Nube de Palabras"),
                dbc.CardBody([
                    html.Div(id="wordcloud-container", children=[html.P("Carga un archivo para ver la nube de palabras")])
                ])
            ], className="mb-4"),
            dbc.Card([
                dbc.CardHeader("Top 10 Palabras Más Frecuentes"),
                dbc.CardBody([
                    dcc.Graph(id="bar-chart", figure={})
                ])
            ]),
        ], md=6),
        # Right column: Histogram and sentiment distribution
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Distribución de la Longitud de Opiniones"),
                dbc.CardBody([
                    dcc.Graph(id="histogram-chart", figure={})
                ])
            ], className="mb-4"),
            dbc.Card([
                dbc.CardHeader("Distribución de Sentimientos"),
                dbc.CardBody([
                    dcc.Graph(id="pie-chart", figure={})
                ])
            ])
        ], md=6)
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Tabla de Opiniones Clasificadas"),
                dbc.CardBody([
                    dash_table.DataTable(
                        id="sentiment-table",
                        columns=[{"name": "Opinión", "id": "review"}, {"name": "Sentimiento", "id": "sentiment"}],
                        data=[],
                        page_size=10,
                        style_table={"overflowX": "auto"},
                        style_cell={"textAlign": "left", "fontSize": 12, "fontFamily": "Arial"}
                    )
                ])
            ])
        ])
    ], className="mb-4"),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Clasifica una Nueva Opinión"),
                dbc.CardBody([
                    dcc.Textarea(id="new-review", placeholder="Escribe aquí tu opinión...", style={"width": "100%", "height": "100px"}),
                    dbc.Button("Analizar", id="analyze-button", color="primary", className="mt-2"),
                    html.Div(id="new-review-result", className="mt-2")
                ])
            ])
        ], md=6)
    ])
], fluid=True)


############################################################
# Callbacks
############################################################


@callback(
    Output("stored-data", "data"),
    Output("stored-processed", "data"),
    Output("stored-sentiments", "data"),
    Output("wordcloud-container", "children"),
    Output("bar-chart", "figure"),
    Output("histogram-chart", "figure"),
    Output("sentiment-table", "data"),
    Output("pie-chart", "figure"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    prevent_initial_call=True
)
def process_uploaded_file(contents: str, filename: str):
    """Process the uploaded CSV file and update all downstream components.

    Returns multiple outputs to update stores and figures simultaneously.
    """
    if contents is None:
        # No file uploaded
        return None, None, None, [html.P("Carga un archivo para ver la nube de palabras")], {}, {}, [], {}
    # Decode the uploaded file contents (Dash uploads as base64 string)
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    try:
        # Read CSV into DataFrame
        df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
    except Exception:
        return None, None, None, [html.P("No se pudo leer el archivo. Asegúrate de que sea un CSV válido.")], {}, {}, [], {}
    # Identify the column containing reviews
    review_col_candidates = [col for col in df.columns if col.lower() in ["review", "opinion", "comentario", "comentarios"]]
    if not review_col_candidates:
        return None, None, None, [html.P("No se encontró una columna 'review' u 'opinion' en el archivo.")], {}, {}, [], {}
    review_col = review_col_candidates[0]
    reviews = df[review_col].dropna().astype(str).tolist()
    if len(reviews) == 0:
        return None, None, None, [html.P("No hay opiniones en el archivo.")], {}, {}, [], {}
    # Preprocess reviews and compute frequencies
    processed = preprocess_reviews(reviews)
    # Generate word cloud
    img_uri, _ = generate_wordcloud(processed.tokens)
    wordcloud_component = html.Img(src=img_uri, style={"maxWidth": "100%", "height": "auto"})
    # Bar chart of top 10 words
    top_words = processed.frequencies.head(10)
    bar_fig = px.bar(
        x=top_words.index,
        y=top_words.values,
        labels={"x": "Palabra", "y": "Frecuencia"},
        title="Top 10 palabras más frecuentes"
    )
    bar_fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    # Histogram of review lengths (number of processed tokens per review)
    review_lengths = []
    for text in reviews:
        tokens = preprocess_reviews([text]).tokens  # Preprocess individually
        review_lengths.append(len(tokens))
    hist_fig = px.histogram(
        x=review_lengths,
        nbins=10,
        labels={"x": "Número de palabras procesadas", "y": "Cantidad de opiniones"},
        title="Distribución de la longitud de opiniones"
    )
    hist_fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    # Sentiment classification
    sentiments = classify_sentiments(reviews)
    # Data for table
    table_data = pd.DataFrame({"review": reviews, "sentiment": sentiments}).to_dict("records")
    # Pie chart of sentiment distribution
    sent_counts = pd.Series(sentiments).value_counts()
    pie_fig = px.pie(
        values=sent_counts.values,
        names=sent_counts.index,
        title="Distribución de sentimientos"
    )
    pie_fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    # Return all outputs
    return (
        df.to_dict("records"),
        {"tokens": processed.tokens, "frequencies": processed.frequencies.to_dict()},
        sentiments,
        [wordcloud_component],
        bar_fig,
        hist_fig,
        table_data,
        pie_fig,
    )


@callback(
    Output("new-review-result", "children"),
    Input("analyze-button", "n_clicks"),
    State("new-review", "value"),
    prevent_initial_call=True
)
def analyze_new_review(n_clicks: int, value: str):
    """Analyse a new review entered by the user and display its sentiment."""
    if not value:
        return html.P("Por favor, escribe una opinión para analizar.")
    label = classify_sentiments([value])[0]
    color_map = {"Positivo": "success", "Negativo": "danger", "Neutral": "secondary"}
    alert = dbc.Alert(
        f"La opinión ingresada se clasifica como: {label}",
        color=color_map.get(label, "primary"),
        dismissable=True
    )
    return alert


if __name__ == "__main__":
    # Run local development server
    app.run(host="0.0.0.0", port=8050, debug=True)