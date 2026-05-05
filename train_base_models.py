from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def train_base_models(X, y):
    rf = RandomForestClassifier(n_estimators=100)
    gb = GradientBoostingClassifier(n_estimators=100)
    rf.fit(X, y)
    gb.fit(X, y)
    return rf, gb