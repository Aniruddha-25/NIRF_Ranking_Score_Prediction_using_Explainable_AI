import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from flask import Flask, render_template_string, request, jsonify
import webbrowser
import threading
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# =====================================================
# 1. LOAD DATA
# =====================================================

DATA_PATH = "./Dataset/Dataset.csv"

df = pd.read_csv(DATA_PATH)
df.columns = [c.strip() for c in df.columns]

required = [
    "Year",
    "Institute Id",
    "Institute Name",
    "City",
    "State",
    "Score",
    "Rank",
    "TLR",
    "RPC",
    "GO",
    "OI",
    "Perception",
]

for c in required:
    if c not in df.columns:
        raise ValueError(f"Missing column: {c}")

numeric_cols = ["Score", "Rank", "TLR", "RPC", "GO", "OI", "Perception"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["Rank", "Score"]).reset_index(drop=True)

# =====================================================
# 2. MOVEMENT LABEL
# =====================================================

p33 = df["Rank"].quantile(0.33)
p66 = df["Rank"].quantile(0.66)


def movement(rank):
    if rank <= p33:
        return "Improve"
    elif rank <= p66:
        return "Stable"
    return "Decline"


df["Movement"] = df["Rank"].apply(movement)

# =====================================================
# 3. SAFE ENCODING
# =====================================================

feature_cols = [
    "Institute Id",
    "Institute Name",
    "City",
    "State",
    "TLR",
    "RPC",
    "GO",
    "OI",
    "Perception",
]

data = df.copy()

cat_cols = ["Institute Id", "Institute Name", "City", "State"]
safe_map = {}

for col in cat_cols:
    vals = list(data[col].unique())
    mp = {v: i for i, v in enumerate(vals)}
    mp["__NEW__"] = len(mp)
    safe_map[col] = mp
    data[col] = data[col].map(mp)

# =====================================================
# 4. TRAIN ML MODEL
# =====================================================

X = data[feature_cols]
y = data["Movement"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "RF": RandomForestClassifier(),
    "GB": GradientBoostingClassifier(),
    "LR": LogisticRegression(max_iter=500),
}

best_model = None
best_acc = -1

for name, model in models.items():
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    if acc > best_acc:
        best_model = model
        best_acc = acc

# =====================================================
# FLASK ROUTES
# =====================================================

@app.route('/')
def index():
    return render_template_string(open('./templates/index.html').read())

@app.route('/get_institutes')
def get_institutes():
    institutes = df['Institute Name'].unique().tolist()
    return jsonify(sorted(institutes))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        institute_name = data.get('institute_name', '').strip()
        
        # Find institute
        sel = df[df["Institute Name"].str.lower() == institute_name.lower()]
        if len(sel) == 0:
            return jsonify({'error': 'Institute not found'}), 404
        
        row = sel.iloc[0]

        # Prepare input
        base = pd.DataFrame(columns=feature_cols)
        base.loc[0] = 0

        for col in ["TLR", "RPC", "GO", "OI", "Perception"]:
            base.at[0, col] = row[col]

        for col in ["Institute Id", "Institute Name", "City", "State"]:
            mp = safe_map[col]
            v = row[col]
            base.at[0, col] = mp[v] if v in mp else mp["__NEW__"]

        # Current year
        present_year = datetime.datetime.now().year
        curr_score = float(row["Score"])
        curr_rank = float(row["Rank"])

        scaled = scaler.transform(base)
        present_mv = best_model.predict(scaled)[0]

        # Predictions
        score_delta = {"Improve": 2, "Stable": 0, "Decline": -2}
        rank_factor = {"Improve": 0.98, "Stable": 1.00, "Decline": 1.02}

        future_rows = []
        base_copy = base.copy()

        for i in range(1, 6):
            yr = present_year + i
            scaled_future = scaler.transform(base_copy)
            mv = best_model.predict(scaled_future)[0]

            d = score_delta[mv]
            curr_score = max(0, curr_score + d)
            curr_rank = max(1, curr_rank * rank_factor[mv])

            for c in ["TLR", "RPC", "GO", "OI", "Perception"]:
                base_copy.at[0, c] += d

            future_rows.append({
                'year': yr,
                'score': round(curr_score, 2),
                'rank': int(round(curr_rank)),
                'movement': mv
            })

        # Conclusion
        mvts = [r['movement'] for r in future_rows]
        if mvts.count("Improve") >= 3:
            conclusion = "Institution will MOST LIKELY IMPROVE."
        elif mvts.count("Decline") >= 3:
            conclusion = "Institution will MOST LIKELY DECLINE."
        else:
            conclusion = "Institution will MOST LIKELY REMAIN STABLE."

        return jsonify({
            'institute_info': {
                'Institute Id': row['Institute Id'],
                'Institute Name': row['Institute Name'],
                'City': row['City'],
                'State': row['State']
            },
            'present': {
                'year': present_year,
                'score': round(row['Score'], 2),
                'rank': int(row['Rank']),
                'movement': present_mv
            },
            'forecast': future_rows,
            'conclusion': conclusion
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Open browser
    threading.Timer(1.0, lambda: webbrowser.open('http://127.0.0.1:5000')).start()
    app.run(debug=True, use_reloader=False)




