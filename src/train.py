from preprocess import load_data
from model import build_model
from sklearn.model_selection import train_test_split

X, y = load_data('data/TB_Chest_Radiography_Database')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = build_model()

model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=64
)

model.save("model.keras")
