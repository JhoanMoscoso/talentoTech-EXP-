import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#Definir los dataframes
df = pd.read_csv(os.getcwd()+'/df_limpio.csv')

def remove_extreme_outliers(df, column='salario', group_col='sexo', threshold=3):
    """
    Se deben eliminar datos anómalos extremos que causan un desequilibro entre las features y el target (salarios muy altos).
    En este caso, lo hacemos agrupándolos por sexo, eliminando datos anómalos de la columna 'salario'
    """
    df_clean = df.copy()
    outlier_indices = []
    
    for group in df[group_col].unique():
        group_data = df[df[group_col] == group][column]
        
        # Cuartiles por fuera de los cuales se eliminan
        Q1 = group_data.quantile(0.25)
        Q3 = group_data.quantile(0.75)
        IQR = Q3 - Q1
        
        # Fronteras de decisión
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        # Indice de los elementos fuera
        outliers = df[(df[group_col] == group) & 
                      ((df[column] < lower_bound) | (df[column] > upper_bound))].index
        
        outlier_indices.extend(outliers)
        
        print(f"Grupo {group} (n={len(group_data)}):")
        print(f"  - Fronteras de anómalos: {lower_bound:,.2f} to {upper_bound:,.2f}")
        print(f"  - Se identificaron {len(outliers)} datos anómalos extremos ({len(outliers)/len(group_data)*100:.2f}%)")
    
    # Create clean dataframe without outliers
    df_clean = df_clean.drop(outlier_indices)
    print(f"\n Se eliminaron {len(outlier_indices)} outliers")
    print(f"Tamaño inicial: {len(df)}, Nuevo tamaño: {len(df_clean)}")
    
    return df_clean

df_cleann = remove_extreme_outliers(df, column='salario', group_col='sexo', threshold=3)

df_cleann.columns

X = df_cleann.drop(['salario', "UID"], axis=1)
y = df_cleann['salario']

class SalaryPredictor:
    def __init__(self, model_type=RandomForestRegressor, model_params=None):
        """
        Initialize the Salary Predictor
        
        Parameters:
        model_type (sklearn estimator): Machine learning model to use
        model_params (dict): Predefined optimal parameters
        """
        # Scalers
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
        # Model
        self.model_type = model_type
        self.model_params = model_params or {
            'n_estimators': 200,
            'max_depth': 20,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42
        }
        self.model = None
        
        # Metadata
        self.feature_columns = None
        self.original_target_mean = None
        self.original_target_std = None

    def _to_array(self, data):
        """
        Convert input to NumPy array
        """
        if isinstance(data, pd.Series):
            return data.values
        elif isinstance(data, pd.DataFrame):
            return data.values
        elif isinstance(data, np.ndarray):
            return data
        else:
            return np.array(data)

    def prepare_data(self, X, y):
        """
        Prepare and scale the data
        
        Parameters:
        X (DataFrame): Input features
        y (Series): Target variable
        
        Returns:
        tuple: Scaled features and target
        """
        # Store feature column names
        self.feature_columns = X.columns.tolist() if isinstance(X, pd.DataFrame) else None
        
        # Convert to arrays
        X_array = self._to_array(X)
        y_array = self._to_array(y)
        
        # Store original target statistics
        self.original_target_mean = np.mean(y_array)
        self.original_target_std = np.std(y_array)
        
        # Scale features
        X_scaled = self.feature_scaler.fit_transform(X_array)
        
        # Scale target variable
        y_scaled = self.target_scaler.fit_transform(y_array.reshape(-1, 1)).ravel()
        
        return X_scaled, y_scaled

    def train_model(self, X, y):
        """
        Train the model with predefined parameters
        
        Parameters:
        X (DataFrame): Input features
        y (Series): Target variable
        
        Returns:
        dict: Training results
        """
        # Prepare scaled data
        X_scaled, y_scaled = self.prepare_data(X, y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=0.2, random_state=42
        )
        
        # Create and train model with predefined parameters
        self.model = self.model_type(**self.model_params)
        self.model.fit(X_train, y_train)
        
        # Predictions
        y_pred_scaled = self.model.predict(X_test)
        
        # Reverse scaling for evaluation
        y_test_original = (y_test * self.original_target_std) + self.original_target_mean
        y_pred_original = (y_pred_scaled * self.original_target_std) + self.original_target_mean
        
        # Evaluation metrics
        results = {
            'mse': mean_squared_error(y_test_original, y_pred_original),
            'mae': mean_absolute_error(y_test_original, y_pred_original),
            'r2': r2_score(y_test_original, y_pred_original),
            'model_params': self.model_params
        }
        
        return results

    def predict(self, X_new):
        """
        Make predictions
        
        Parameters:
        X_new (DataFrame): New input features
        
        Returns:
        numpy.ndarray: Predicted salaries
        """
        # Convert to array
        X_new_array = self._to_array(X_new)
        
        # Scale input features
        X_new_scaled = self.feature_scaler.transform(X_new_array)
        
        # Predict scaled salaries
        y_pred_scaled = self.model.predict(X_new_scaled)
        
        # Reverse scaling
        y_pred_original = (y_pred_scaled * self.original_target_std) + self.original_target_mean
        
        return y_pred_original

    def save_model(self, filepath=os.getcwd()+'/randomForest_salarios1.pkl'):
        """
        Save the entire model object
        """
        with open(filepath, 'wb') as file:
            pickle.dump(self, file)
        print(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath=os.getcwd()+'/randomForest_salarios1.pkl'):
        """
        Load the entire model object
        """
        with open(filepath, 'rb') as file:
            return pickle.load(file)




# Custom model parameters (if needed)
custom_params = {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}

# Create predictor with custom parameters
predictor = SalaryPredictor(model_params=custom_params)

# Entrenar el modelo
"""
training_results = predictor.train_model(X, y)
print("Training Results:")
for key, value in training_results.items():
    print(f"{key}: {value}")

# Save the model
predictor.save_model()
"""
# Load the model
loaded_predictor = SalaryPredictor.load_model()

# Make predictions
X_sample = X.sample(n=5)
print("\n Features de prueba:\n",X_sample)
predictions = loaded_predictor.predict(X_sample)

print("\nSample Predictions:")
print(predictions)