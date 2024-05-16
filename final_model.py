{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "50848443-1723-4c93-80cb-c6fd247c56c8",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "\n",
    "df = pd.read_csv('metaverse_transactions_dataset.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "28c4d0f1-7599-4b68-bac5-66e5e45036a3",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "X = df.drop(columns=['anomaly'])\n",
    "y = df['anomaly']\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "d2c14b8f-ab15-4e7d-b618-f49532e0968a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Null/Missing Values:\n",
      "timestamp            0\n",
      "hour_of_day          0\n",
      "sending_address      0\n",
      "receiving_address    0\n",
      "amount               0\n",
      "transaction_type     0\n",
      "location_region      0\n",
      "ip_prefix            0\n",
      "login_frequency      0\n",
      "session_duration     0\n",
      "purchase_pattern     0\n",
      "age_group            0\n",
      "risk_score           0\n",
      "dtype: int64\n"
     ]
    }
   ],
   "source": [
    "# Categorize data into categorical and numerical values\n",
    "categorical_cols = ['transaction_type', 'location_region', 'ip_prefix', 'purchase_pattern', 'age_group']\n",
    "numerical_cols = ['hour_of_day', 'amount', 'risk_score', 'login_frequency', 'session_duration']\n",
    "\n",
    "# Check for null and missing values\n",
    "print(\"Null/Missing Values:\")\n",
    "print(X_train.isnull().sum())\n",
    "\n",
    "# Data types correction\n",
    "X_train['timestamp'] = pd.to_datetime(X_train['timestamp'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "b6eecebd-77ce-4091-9ab2-9cb7bd3ffc50",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.impute import SimpleImputer\n",
    "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n",
    "from sklearn.compose import ColumnTransformer\n",
    "from sklearn.pipeline import Pipeline\n",
    "\n",
    "numeric_transformer = Pipeline(steps=[\n",
    "    ('imputer', SimpleImputer(strategy='median')),  # Replace missing values with median\n",
    "    ('scaler', StandardScaler())  # Standardize the features\n",
    "])\n",
    "\n",
    "categorical_transformer = Pipeline(steps=[\n",
    "    ('imputer', SimpleImputer(strategy='most_frequent')),  # Replace missing values with most frequent value\n",
    "    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categorical features\n",
    "])\n",
    "\n",
    "preprocessor = ColumnTransformer(\n",
    "    transformers=[\n",
    "        ('num', numeric_transformer, numerical_cols),\n",
    "        ('cat', categorical_transformer, categorical_cols)\n",
    "    ])\n",
    "\n",
    "X_train_processed = preprocessor.fit_transform(X_train)\n",
    "\n",
    "X_test_processed = preprocessor.transform(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "3bcd09e1-49aa-4882-a066-f0638b001208",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier\n",
    "\n",
    "model = RandomForestClassifier()\n",
    "model.fit(X_train_processed, y_train)\n",
    "\n",
    "y_pred = model.predict(X_test_processed)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "2d3c2a82-c239-4146-bd66-769b4ba3df3f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 1299,     0,     0],\n",
       "       [    0, 12699,     0],\n",
       "       [    0,     0,  1722]], dtype=int64)"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.metrics import confusion_matrix\n",
    "confusion_matrix(y_test, y_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "557ff5c7-fee6-48cd-b0e1-fe4376950de1",
   "metadata": {},
   "outputs": [],
   "source": [
    "from dill import dump, load\n",
    "with open('model.pkl','wb') as f:\n",
    "    dump(model,f)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "b4c9fa74-05c5-4e50-99ee-ec42d4d109b0",
   "metadata": {},
   "outputs": [],
   "source": [
    "with open('model.pkl','rb') as f:\n",
    "    reloaded_model = load(f)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "29ec16b3-4565-4c39-8ae1-f8ea3b9616cb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 1299,     0,     0],\n",
       "       [    0, 12699,     0],\n",
       "       [    0,     0,  1722]], dtype=int64)"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_pred = reloaded_model.predict(X_test_processed)\n",
    "confusion_matrix(y_test, y_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b8b6b373-5023-4db9-a1d2-03b66e215327",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
