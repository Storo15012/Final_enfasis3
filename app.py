"""
Dash application for analyzing customer opinions and performing sentiment analysis.

This application allows users to upload a CSV file containing opinions (one per row
under a column named ``opinion``). It then cleans and lemmatizes the text,
displays a word cloud, a bar chart of the most frequent words, and an additional
histogram of comment lengths. A sentiment classifier based on a multilingual
BERT model from HuggingFace categorizes each opinion as positive, negative or
neutral. The results are shown in a table along with a pie chart summarizing
the class distribution. Users can also enter a new comment in a text field to
see its predicted sentiment immediately.

Dependencies:
    - dash
    - dash_bootstrap_components
    - pandas
    - plotly
    - spacy (with Spanish model ``es_core_news_sm`` installed)
    - wordcloud
    - transformers and torch (for the sentiment model)

Run the app with ``python app.py``. If no file is uploaded, the app will
automatically load the sample ``comentarios.csv`` in the repository.
"""

import base64
import io
from collections import Counter

import dash
from dash import dcc, html, dash_table, Output, Input, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px

# Text processing
import spacy
from wordcloud import WordCloud

from transformers import pipeline


# Load Spanish language model for lemmatization and stopword detection
try:
    nlp = spacy.load("es_core_news_sm")
except OSError:
    # If the model isn't present, download it programmatically
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "es_core_news_sm"], check=True)
    nlp = spacy.load("es_core_news_sm")


def clean_and_lemmatize(text: str) -> list[str]:
    """Remove stopwords, non‑alphabetic tokens and lemmatize Spanish text.

    Args:
        text: Raw text string.

    Returns:
        A list of cleaned, lemmatized tokens.
    """
    doc = nlp(text.lower())
    tokens = []
    for token in doc:
        # Keep alphabetic tokens that are not stop words and longer than 2 characters
        if token.is_alpha and not token.is_stop and len(token) > 2:
            tokens.append(token.lemma_)
    return tokens


def generate_wordcloud(tokens: list[str]) -> bytes:
    """Generate a word cloud image from a list of tokens.

    Args:
        tokens: A list of tokens.

    Returns:
        The PNG image content encoded as bytes.
    """
    # Build a single string for the WordCloud generator
    text = " ".join(tokens)
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    image = wc.to_image()
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def top_n_words(tokens: list[str], n: int = 10) -> pd.DataFrame:
    """Return a DataFrame with the top n most common tokens.

    Args:
        tokens: A list of tokens.
        n: Number of top words to return.

    Returns:
        DataFrame with columns ``word`` and ``frequency`` sorted descending.
    """
    counts = Counter(tokens)
    common = counts.most_common(n)
    return pd.DataFrame(common, columns=["word", "frequency"])


def sentiment_pipeline():
    """Load a multilingual sentiment analysis pipeline.

    Returns:
        A HuggingFace pipeline object that outputs star ratings.
    """
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")


def map_rating_to_sentiment(label: str) -> str:
    """Map the star rating (``1 star``, ``2 stars``, etc.) to sentiment classes.

    Args:
        label: A label returned by the sentiment model.

    Returns:
        ``"negativo"``, ``"neutro"`` or ``"positivo"``.
    """
    # Extract the number of stars from the label (e.g. '3 stars' -> 3)
    stars = int(label.split()[0])
    if stars <= 2:
        return "negativo"
    elif stars == 3:
        return "neutro"
    return "positivo"


def classify_opinions(opinions: list[str], clf) -> pd.DataFrame:
    """Classify opinions into sentiment categories using the provided classifier.

    Args:
        opinions: List of raw opinion strings.
        clf: The HuggingFace pipeline for sentiment analysis.

    Returns:
        A DataFrame with columns ``opinion``, ``sentiment`` and ``label``.
    """
    results = clf(opinions)
    sentiments = [map_rating_to_sentiment(res["label"]) for res in results]
    labels = [res["label"] for res in results]
    return pd.DataFrame({"opinion": opinions, "sentiment": sentiments, "rating": labels})


def load_initial_data() -> pd.DataFrame:
    """Load the default sample data from ``comentarios.csv`` if available.

    Returns:
        DataFrame containing the sample opinions.
    """
    try:
        df = pd.read_csv("comentarios.csv")
    except FileNotFoundError:
        # Fallback: return empty DataFrame
        df = pd.DataFrame(columns=["opinion"])
    return df


def prepare_data(df: pd.DataFrame) -> tuple[list[str], list[str], pd.DataFrame, list[str]]:
    """Process the DataFrame of opinions for visualization and analysis.

    Args:
        df: DataFrame with an ``opinion`` column.

    Returns:
        - tokens: list of all cleaned tokens from all opinions.
        - lengths: list of comment lengths (number of tokens per opinion).
        - freq_df: DataFrame of top words with frequencies.
        - cleaned_opinions: list of cleaned opinions reconstructed from tokens for classification.
    """
    cleaned_tokens = []
    lengths = []
    cleaned_opinions = []
    for text in df["opinion"].astype(str).tolist():
        tokens = clean_and_lemmatize(text)
        cleaned_tokens.extend(tokens)
        lengths.append(len(tokens))
        cleaned_opinions.append(" ".join(tokens))
    freq_df = top_n_words(cleaned_tokens, n=10)
    return cleaned_tokens, lengths, freq_df, cleaned_opinions


# Initialize sentiment classifier once (expensive)
sentiment_clf = sentiment_pipeline()

# Create Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Expose the underlying Flask server for deployment (e.g., Render, Heroku)
server = app.server
app.title = "Análisis de opiniones y sentimientos"

# Layout definition
app.layout = dbc.Container([
    html.H1("Análisis de opiniones de clientes", className="text-center mt-4 mb-4"),
    dcc.Upload(
        id="upload-data",
        children=html.Div([
            "Arrastre y suelte o ", html.A("seleccione un archivo CSV")
        ]),
        style={
            "width": "100%",
            "height": "60px",
            "lineHeight": "60px",
            "borderWidth": "1px",
            "borderStyle": "dashed",
            "borderRadius": "5px",
            "textAlign": "center",
            "margin-bottom": "20px",
        },
        # Allow multiple files to be uploaded
        multiple=False
    ),
    dbc.Row([
        dbc.Col([
            html.H4("Nube de palabras"),
            html.Img(id="wordcloud", style={"width": "100%"}),
        ], md=6),
        dbc.Col([
            html.H4("Palabras más frecuentes"),
            dcc.Graph(id="freq-bar"),
        ], md=6),
    ]),
    dbc.Row([
        dbc.Col([
            html.H4("Distribución de longitudes de opiniones"),
            dcc.Graph(id="length-hist"),
        ], md=6),
        dbc.Col([
            html.H4("Distribución de sentimientos"),
            dcc.Graph(id="sentiment-pie"),
        ], md=6),
    ]),
    html.H4("Tabla de opiniones clasificadas"),
    dash_table.DataTable(
        id="sentiment-table",
        columns=[{"name": col, "id": col} for col in ["opinion", "sentiment", "rating"]],
        style_cell={'whiteSpace': 'pre-line', 'textAlign': 'left'},
        style_table={'overflowX': 'auto'},
        page_size=10
    ),
    html.Hr(),
    html.H4("Enviar un comentario nuevo"),
    dcc.Textarea(
        id="new-comment",
        placeholder="Escriba su comentario aquí...",
        style={'width': '100%', 'height': 100},
    ),
    html.Br(),
    dbc.Button("Clasificar comentario", id="classify-button", color="primary", className="mt-2"),
    html.Div(id="new-comment-output", className="mt-3"),
], fluid=True)


def parse_contents(contents: str, filename: str) -> pd.DataFrame:
    """Parse uploaded file contents into a DataFrame.

    Args:
        contents: Base64 encoded string from the upload component.
        filename: Name of the uploaded file.

    Returns:
        DataFrame with an ``opinion`` column.
    """
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if filename.lower().endswith('.csv'):
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        else:
            raise ValueError("Formato no soportado. Por favor suba un archivo CSV.")
    except Exception as e:
        print(e)
        return pd.DataFrame(columns=["opinion"])
    # Ensure column exists
    if 'opinion' not in df.columns:
        # If there is only one column, rename it
        if df.shape[1] == 1:
            df.columns = ['opinion']
        else:
            raise ValueError("El archivo CSV debe contener una columna llamada 'opinion'")
    return df[['opinion']]


@app.callback(
    [Output('wordcloud', 'src'),
     Output('freq-bar', 'figure'),
     Output('length-hist', 'figure'),
     Output('sentiment-table', 'data'),
     Output('sentiment-pie', 'figure')],
    [Input('upload-data', 'contents'), Input('upload-data', 'filename')]
)
def update_output(uploaded_contents, uploaded_filename):
    """Update all outputs when a file is uploaded. If no file, use the default data."""
    if uploaded_contents is not None and uploaded_filename is not None:
        df = parse_contents(uploaded_contents, uploaded_filename)
    else:
        df = load_initial_data()
    if df.empty:
        return None, px.bar(title="No hay datos"), px.histogram(title="No hay datos"), [], px.pie()

    tokens, lengths, freq_df, cleaned_opinions = prepare_data(df)
    # Wordcloud image encoded in base64
    img_bytes = generate_wordcloud(tokens)
    encoded_img = 'data:image/png;base64,' + base64.b64encode(img_bytes).decode()
    # Bar chart for top words
    bar_fig = px.bar(freq_df, x='word', y='frequency', title='Top 10 palabras', text='frequency')
    bar_fig.update_layout(xaxis_title="Palabra", yaxis_title="Frecuencia")
    # Histogram of lengths
    hist_fig = px.histogram(x=lengths, nbins=10, title="Longitud de opiniones (número de tokens)")
    hist_fig.update_layout(xaxis_title="Número de tokens", yaxis_title="Número de opiniones")
    # Sentiment classification
    sentiment_df = classify_opinions(cleaned_opinions, sentiment_clf)
    # Pie chart of sentiment distribution
    pie_fig = px.pie(sentiment_df, names='sentiment', title="Distribución de sentimientos")
    return encoded_img, bar_fig, hist_fig, sentiment_df.to_dict('records'), pie_fig


@app.callback(
    Output('new-comment-output', 'children'),
    [Input('classify-button', 'n_clicks')],
    [State('new-comment', 'value')]
)
def classify_new_comment(n_clicks, comment):
    """Classify a new comment entered by the user and display its sentiment."""
    if not n_clicks or not comment:
        return ''
    tokens = clean_and_lemmatize(comment)
    cleaned_text = " ".join(tokens)
    result = sentiment_clf(cleaned_text)[0]
    sentiment = map_rating_to_sentiment(result['label'])
    return html.Div([
        html.Strong("Sentimiento predicho: "), html.Span(sentiment.capitalize()), html.Br(),
        html.Small(f"Etiqueta del modelo: {result['label']} (confianza {result['score']:.2f})")
    ])


if __name__ == '__main__':
    # If running locally, load the default data first so the app has something to show
    app.run_server(debug=False, host='0.0.0.0', port=8050)