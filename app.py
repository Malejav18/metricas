import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px
import numpy as np
import joblib

# Cargar y preparar datos
df = pd.read_csv("CO2 Emissions_Canada.csv")
df = df.dropna()

# Etiqueta de alta emisión (para clasificación binaria)
df['Alta Emision'] = (df['CO2 Emissions(g/km)'] > 200).astype(int)

# Cargar el modelo entrenado
modelo = joblib.load("modelo_ridge.joblib")

# Crear figuras estáticas
fig_hist = px.histogram(df, x='CO2 Emissions(g/km)', nbins=50,
                        title='Distribución de Emisiones de CO2',
                        labels={'CO2 Emissions(g/km)': 'CO2 Emisiones (g/km)'})

fig_scatter = px.scatter(df, x='Fuel Consumption Comb (L/100 km)', 
                         y='CO2 Emissions(g/km)', color='Fuel Type',
                         title='Consumo de Combustible vs Emisiones de CO2',
                         labels={'Fuel Consumption Comb (L/100 km)': 'Consumo (L/100 km)',
                                 'CO2 Emissions(g/km)': 'CO2 Emisiones'})

fig_box = px.box(df, x='Fuel Type', y='CO2 Emissions(g/km)',
                 title='Emisiones de CO2 por Tipo de Combustible',
                 labels={'Fuel Type': 'Tipo de Combustible',
                         'CO2 Emissions(g/km)': 'CO2 Emisiones'})

df_mean = df.groupby('Vehicle Class')['CO2 Emissions(g/km)'].mean().reset_index()
fig_bar = px.bar(df_mean, x='Vehicle Class', y='CO2 Emissions(g/km)',
                 title='Emisión Promedio de CO2 por Clase de Vehículo',
                 labels={'CO2 Emissions(g/km)': 'CO2 Emisiones'})

# Crear la app
app = dash.Dash(__name__)
server = app.server

# Layout del dashboard
app.layout = html.Div([
    html.H1("Dashboard de Emisiones de CO2", style={'textAlign': 'center'}),

    dcc.Graph(figure=fig_hist),
    dcc.Graph(figure=fig_scatter),
    dcc.Graph(figure=fig_box),
    dcc.Graph(figure=fig_bar),

    html.H2("Emisiones según el consumo de combustible"),
    dcc.Graph(
        id='scatter_consumo',
        figure=px.scatter(df, x='Fuel Consumption City (L/100 km)', y='CO2 Emissions(g/km)',
                          color='Alta Emision', hover_data=['Fuel Type', 'Transmission'])
    ),

    html.H2("Histograma de emisiones por tipo de combustible"),
    dcc.Dropdown(
        id='dropdown_fuel',
        options=[{'label': f, 'value': f} for f in df['Fuel Type'].unique()],
        value=df['Fuel Type'].unique()[0],
        clearable=False
    ),
    dcc.Graph(id='histograma_emisiones')
])

# Callback: histograma interactivo
@app.callback(
    Output('histograma_emisiones', 'figure'),
    Input('dropdown_fuel', 'value')
)
def update_histograma(fuel_type):
    filtered = df[df['Fuel Type'] == fuel_type]
    fig = px.histogram(filtered, x='CO2 Emissions(g/km)', nbins=30,
                       title=f"Emisiones para combustible: {fuel_type}",
                       labels={'CO2 Emissions(g/km)': 'CO2 Emisiones'})
    return fig

# Ejecutar la app
if __name__ == '__main__':
    app.run(debug=True)