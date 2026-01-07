import matplotlib
matplotlib.use("Agg")  # Non-GUI backend (REQUIRED for Flask)

from flask import Flask, render_template, request
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import unicodedata
import os
import io
import base64

app = Flask(__name__)

# -----------------------------
# Load Tamil Word2Vec model
# -----------------------------
MODEL_PATH = os.path.join("model", "ta_small.kv")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Model file not found. Expected at: model/ta_small.kv"
    )

model = KeyedVectors.load(MODEL_PATH)
print("✅ Tamil Word2Vec model loaded")
print("📘 Vocabulary size:", len(model))

# -----------------------------
# Load Tamil font
# -----------------------------
FONT_PATH = os.path.join("Font", "Lohit-Tamil.ttf")
tamil_font = fm.FontProperties(fname=FONT_PATH) if os.path.exists(FONT_PATH) else None

# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    img_data = None

    if request.method == "POST":
        word = request.form.get("word", "").strip()
        word = unicodedata.normalize("NFC", word)

        try:
            topn = int(request.form.get("topn", 5))
        except ValueError:
            topn = 5

        if not word:
            result = "Please enter a word."
        elif word not in model:
            result = "Word not found in vocabulary."
        else:
            similar_words = model.most_similar(word, topn=topn)

            words = [word] + [w for w, _ in similar_words]
            normalized_words = [unicodedata.normalize("NFC", w) for w in words]
            vectors = [model[w] for w in words]

            # PCA reduction
            pca = PCA(n_components=2)
            reduced = pca.fit_transform(vectors)

            # Plot
            fig, ax = plt.subplots(figsize=(8, 6))
            for i, w in enumerate(normalized_words):
                color = "red" if i == 0 else "blue"
                ax.scatter(reduced[i, 0], reduced[i, 1], color=color)
                ax.annotate(
                    w,
                    (reduced[i, 0] + 0.01, reduced[i, 1] + 0.01),
                    fontproperties=tamil_font,
                    fontsize=12,
                    color=color
                )

            ax.set_title(
                f"Top {topn} Similar Words to '{word}'",
                fontproperties=tamil_font
            )
            ax.grid(True)

            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format="png")
            buf.seek(0)
            img_data = base64.b64encode(buf.read()).decode("utf-8")
            buf.close()
            plt.close()

            result = [(w, f"{s:.4f}") for w, s in similar_words]

    return render_template(
        "tamil_index.html",
        result=result,
        image=img_data
    )

# -----------------------------
# Main (Render + local compatible)
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5002))
    app.run(host="0.0.0.0", port=port, debug=True)
