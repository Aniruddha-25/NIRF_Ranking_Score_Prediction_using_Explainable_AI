import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings("ignore")

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

print(f"\nBest Model Selected → Accuracy = {best_acc:.4f}\n")

# =====================================================
# 5. SELECT COLLEGE
# =====================================================

disp = df[["Institute Name", "City"]].reset_index().rename(columns={"index": "SrNo"})

print("\n============================")
print("      SELECT A COLLEGE")
print("============================\n")

print(disp.to_string(index=False))

choice = input("\nEnter SrNo OR Institute Name: ").strip()

if choice.isdigit():
    row = df.loc[int(choice)]
else:
    sel = df[df["Institute Name"].str.lower().str.contains(choice.lower())]
    if len(sel) == 0:
        raise ValueError("No such college found.")
    elif len(sel) > 1:
        print("\nMultiple Matches:")
        opt = sel.reset_index().rename(columns={"index": "SrNo"})
        print(opt.to_string(index=False))
        idx = int(input("\nChoose SrNo: "))
        row = df.loc[idx]
    else:
        row = sel.iloc[0]

# =====================================================
# 6. PRINT SELECTED INSTITUTE (TABLE STYLE)
# =====================================================

print("\n============================")
print("      SELECTED INSTITUTE")
print("============================\n")

selected_info = pd.DataFrame(
    {
        "Field": ["Institute Id", "Institute Name", "City", "State"],
        "Value": [
            row["Institute Id"],
            row["Institute Name"],
            row["City"],
            row["State"],
        ],
    }
)

print(selected_info.to_string(index=False))

# =====================================================
# 7. PREPARE INPUT FOR ML
# =====================================================

base = pd.DataFrame(columns=feature_cols)
base.loc[0] = 0

for col in ["TLR", "RPC", "GO", "OI", "Perception"]:
    base.at[0, col] = row[col]

for col in ["Institute Id", "Institute Name", "City", "State"]:
    mp = safe_map[col]
    v = row[col]
    base.at[0, col] = mp[v] if v in mp else mp["__NEW__"]

# =====================================================
# 8. PRESENT YEAR (SYSTEM YEAR)
# =====================================================

present_year = datetime.datetime.now().year
curr_score = float(row["Score"])
curr_rank = float(row["Rank"])

scaled = scaler.transform(base)
present_mv = best_model.predict(scaled)[0]

print("\n============================")
print("      PRESENT YEAR RESULT")
print("============================\n")

present_table = pd.DataFrame(
    {
        "Year": [present_year],
        "Score": [curr_score],
        "Rank": [int(curr_rank)],
        "Movement": [present_mv],
    }
)

print(present_table.to_string(index=False))

# =====================================================
# 9. NEXT 5 YEARS PREDICTION
# =====================================================

score_delta = {"Improve": 2, "Stable": 0, "Decline": -2}
rank_factor = {"Improve": 0.98, "Stable": 1.00, "Decline": 1.02}

future_rows = []

for i in range(1, 6):
    yr = present_year + i

    scaled_future = scaler.transform(base)
    mv = best_model.predict(scaled_future)[0]

    d = score_delta[mv]
    curr_score = max(0, curr_score + d)
    curr_rank = max(1, curr_rank * rank_factor[mv])

    for c in ["TLR", "RPC", "GO", "OI", "Perception"]:
        base.at[0, c] += d

    future_rows.append([yr, round(curr_score, 2), int(round(curr_rank)), mv])

future_df = pd.DataFrame(future_rows, columns=["Year", "Score", "Rank", "Movement"])

print("\n============================")
print("      NEXT 5 YEARS FORECAST")
print("============================\n")
print(future_df.to_string(index=False))

# =====================================================
# 10. FINAL CONCLUSION
# =====================================================

mvts = future_df["Movement"].tolist()

if mvts.count("Improve") >= 3:
    final = "Institution will MOST LIKELY IMPROVE."
elif mvts.count("Decline") >= 3:
    final = "Institution will MOST LIKELY DECLINE."
else:
    final = "Institution will MOST LIKELY REMAIN STABLE."

print("\n============================")
print("        FINAL CONCLUSION")
print("============================")
print(final)




