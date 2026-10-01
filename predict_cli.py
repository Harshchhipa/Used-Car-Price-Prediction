import joblib
import pandas as pd
import sys
import os

def clear_screen():
    # Helper to clear terminal depending on OS
    os.system('cls' if os.name == 'nt' else 'clear')

def get_valid_input(prompt, valid_options=None, input_type=str):
    """
    Continually prompts the user until a valid input is provided.
    """
    while True:
        try:
            user_input = input(prompt).strip()
            
            # Type casting
            user_input = input_type(user_input)
            
            # Validation logic
            if valid_options is not None:
                if isinstance(valid_options, (list, tuple)):
                    # For string categorical lists
                    if input_type == str:
                        user_input_title = user_input.title()
                        valid_options_title = [str(opt).title() for opt in valid_options]
                        if user_input_title not in valid_options_title:
                            print(f"  [!] Invalid choice. Please choose from: {', '.join(valid_options)}")
                            continue
                        # Standardize formatting to match training data exactly
                        user_input = user_input_title 
                
                elif isinstance(valid_options, range):
                    # For integer range checking
                    if user_input not in valid_options:
                        print(f"  [!] Invalid input. Range must be between {valid_options.start} and {valid_options.stop - 1}.")
                        continue
            
            return user_input
            
        except ValueError:
            print(f"  [!] Invalid format. Expected a {input_type.__name__}.")
        except EOFError:
             print("\n  [!] Input stream ended. Exiting.")
             sys.exit(1)

def main():
    model_file = 'car_price_model.pkl'
    
    # ---------------------------------------------------------
    # 1. Verification and Loading
    # ---------------------------------------------------------
    if not os.path.exists(model_file):
        print(f"[!] Error: Trained model '{model_file}' not found.")
        print("    Please run 'python train_model.py' first.")
        sys.exit(1)
        
    print("[*] Loading trained machine learning model...")
    pipeline = joblib.load(model_file)
    
    # Only clear screen if it's connected to an interactive terminal, 
    # to avoid issues during automated testing loops.
    if sys.stdout.isatty():
        clear_screen()
        
    print("="*60)
    print("     Welcome to the Used Car Price Predictor     ")
    print("="*60)
    print("Please enter the details of the car below:\n")
    
    # ---------------------------------------------------------
    # 2. Interactive User Input (with Validation)
    # ---------------------------------------------------------
    brands = ['Maruti', 'Hyundai', 'Honda', 'Tata', 'Toyota']
    brand = get_valid_input(f"1. Car Brand ({', '.join(brands)}): ", valid_options=brands)
    
    year = get_valid_input("2. Manufacturing Year (e.g., 2010 - 2026): ", valid_options=range(1990, 2027), input_type=int)
    
    fuel_types = ['Petrol', 'Diesel', 'CNG']
    fuel = get_valid_input(f"3. Fuel Type ({', '.join(fuel_types)}): ", valid_options=fuel_types)
    
    driven_kms = get_valid_input("4. Driven Kilometers (e.g., 45000): ", valid_options=range(0, 1000000), input_type=int)
    
    transmissions = ['Manual', 'Automatic']
    transmission = get_valid_input(f"5. Transmission ({', '.join(transmissions)}): ", valid_options=transmissions)
    
    # ---------------------------------------------------------
    # 3. Model Prediction
    # ---------------------------------------------------------
    # Convert inputs to a DataFrame (the format the Pipeline expects)
    input_data = pd.DataFrame({
        'Brand': [brand],
        'Year': [year],
        'Fuel_Type': [fuel],
        'Driven_KMs': [driven_kms],
        'Transmission': [transmission]
    })
    
    print("\n[*] Analyzing features and computing prediction...")
    
    try:
        # Perform prediction
        prediction = pipeline.predict(input_data)[0]
        
        # Format the output mathematically (Convert to Lakhs)
        price_lakhs = prediction / 100000
        
        print("="*60)
        print(f"  Estimated Selling Price: INR {prediction:,.2f} ({price_lakhs:.2f} Lakhs)")
        print("="*60)
        print("\nThank you for using the predictor!\n")
        
    except Exception as e:
        print(f"\n[!] An error occurred during prediction computation: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[*] Process interrupted by user. Goodbye!")
        sys.exit(0)
