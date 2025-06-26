import os
import urllib.request

MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
MODEL_PATH = "lid.176.ftz"

if not os.path.exists(MODEL_PATH):
    print("🔽 Downloading FastText language model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("✅ Download complete.")
else:
    print("✅ FastText model already exists.")
