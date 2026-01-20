import pickle

# Modelul VECHI (cel trimis in arhiva)
with open('temp_models/character_recognizer.pkl', 'rb') as f:
    old_model = pickle.load(f)
    
print("Modelul VECHI (din temp_models/):")
print(f"  StandardScaler asteapta: {old_model['scaler'].n_features_in_} features")

# Modelul NOU (reantrenat)
with open('models/character_recognizer.pkl', 'rb') as f:
    new_model = pickle.load(f)

print("\nModelul NOU (din models/):")
print(f"  StandardScaler asteapta: {new_model['scaler'].n_features_in_} features")

print(f"\nDiferenta: {old_model['scaler'].n_features_in_ - new_model['scaler'].n_features_in_} features")
