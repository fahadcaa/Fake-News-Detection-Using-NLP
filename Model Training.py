# Passive Aggressive Classifier (good for fake news detection)
model = PassiveAggressiveClassifier(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
