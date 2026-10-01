import os
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# ---------------------------------------------------------
# 1. Generate Dummy Data (If no CSV exists)
# ---------------------------------------------------------
def generate_dummy_data(filename='used_cars_data.csv', num_samples=1000):
    brands = ['Maruti', 'Hyundai', 'Honda', 'Tata', 'Toyota']
    fuel_types = ['Petrol', 'Diesel', 'CNG']
    transmissions = ['Manual', 'Automatic']
    
    data = []
    for _ in range(num_samples):
        brand = random.choice(brands)
        year = random.randint(2010, 2024)
        fuel = random.choice(fuel_types)
        driven_kms = random.randint(10000, 150000)
        transmission = random.choice(transmissions)
        
        # Base price logic for realistic-looking dummy data
        base_price = 800000 # 8 Lakhs base
        if brand in ['Toyota', 'Honda']: base_price += 200000
        if brand == 'Maruti': base_price -= 100000
        
        # Depreciation based on age (40k per year)
        age = 2024 - year
        depreciation = age * 40000
        
        # Mileage impact (15k drop per 10,000 kms)
        mileage_drop = (driven_kms / 10000) * 15000
        
        # Fuel & Transmission impact
        fuel_bonus = 50000 if fuel == 'Diesel' else (0 if fuel == 'Petrol' else -30000)
        trans_bonus = 100000 if transmission == 'Automatic' else 0
        
        # Calculate final price with some random noise
        price = base_price - depreciation - mileage_drop + fuel_bonus + trans_bonus
        price += random.randint(-50000, 50000) 
        price = max(50000, price) # Set a minimum base price of 50k
        
        data.append([brand, year, fuel, driven_kms, transmission, price])
        
    df = pd.DataFrame(data, columns=['Brand', 'Year', 'Fuel_Type', 'Driven_KMs', 'Transmission', 'Price'])
    df.to_csv(filename, index=False)
    print(f"[*] Generated dummy dataset with {num_samples} samples -> {filename}")
    return df

def main():
    data_file = 'used_cars_data.csv'
    model_file = 'car_price_model.pkl'
    
    print("==================================================")
    print("      ML Model Training Pipeline      ")
    print("==================================================")

    # ---------------------------------------------------------
    # 2. Data Loading
    # ---------------------------------------------------------
    if not os.path.exists(data_file):
        df = generate_dummy_data(data_file)
    else:
        print(f"[*] Loading existing dataset -> {data_file}")
        df = pd.read_csv(data_file)
        
    # ---------------------------------------------------------
    # 3. Data Preprocessing & Split
    # ---------------------------------------------------------
    # Features (X) and Target (y)
    X = df.drop('Price', axis=1)
    y = df['Price']
    
    # Split into 80% training and 20% testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # ---------------------------------------------------------
    # 4. Create ML Pipeline
    # ---------------------------------------------------------
    # We use a ColumnTransformer to apply One-Hot Encoding strictly to categorical columns.
    # Numerical columns ('Year', 'Driven_KMs') bypass the encoder via 'passthrough'.
    categorical_cols = ['Brand', 'Fuel_Type', 'Transmission']
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ],
        remainder='passthrough' 
    )
    
    # Using RandomForest as it models non-linear relationships well
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # The Pipeline automatically applies preprocessing before passing data to the model
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', model)])
                               
    # ---------------------------------------------------------
    # 5. Model Training
    # ---------------------------------------------------------
    print("[*] Training the Random Forest model... please wait.")
    pipeline.fit(X_train, y_train)
    
    # ---------------------------------------------------------
    # 6. Model Evaluation
    # ---------------------------------------------------------
    predictions = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print("\n--- Model Evaluation Metrics ---")
    print(f"Mean Absolute Error (MAE): INR {mae:,.2f}")
    print(f"R2 Score:                  {r2:.4f}")
    print("--------------------------------\n")
    
    # ---------------------------------------------------------
    # 7. Save Model
    # ---------------------------------------------------------
    joblib.dump(pipeline, model_file)
    print(f"[*] Success! Model pipeline saved to '{model_file}'")

if __name__ == "__main__":
    main()
