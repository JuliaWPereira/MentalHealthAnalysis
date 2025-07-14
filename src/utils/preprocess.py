import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

def preprocess_data(input_path, output_path):
    """
    Lê um dataset do input_path, aplica one hot encoding nas variáveis categóricas e normaliza todas as variáveis.
    Salva o dataset processado no output_path.
    """
    df = pd.read_csv(input_path)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    encoder = OneHotEncoder(sparse_output=False, drop=None)
    encoded = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols), index=df.index)
    df_processed = pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)
    scaler = MinMaxScaler()
    df_processed = pd.DataFrame(scaler.fit_transform(df_processed), columns=df_processed.columns, index=df_processed.index)
    df_processed.to_csv(output_path, index=False)
    return df_processed 