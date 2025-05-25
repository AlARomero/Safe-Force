import pandas as pd
import altair as alt
from altair import Chart
from pandas.core.frame import DataFrame
import ast
from collections import defaultdict

def total_chart_plot(df: DataFrame) -> Chart:

    # Diccionarios para acumular conteos
    ones_count = defaultdict(int)
    total_count = defaultdict(int)

    # Recorrer cada fila del DataFrame
    for row in df['results']:
        parsed_results = ast.literal_eval(row)  # Convertir de string a estructura Python
        for model, values in parsed_results:
            ones_count[model] += sum(v == 1 for v in values)
            total_count[model] += len(values)

    # Crear DataFrame de resultados
    data = pd.DataFrame({
        'modelo': list(total_count.keys()),
        'total': [total_count[m] for m in total_count],
        'total_1s': [ones_count[m] for m in total_count]
    })

    # Gráfico de barras con Altair
    chart = alt.Chart(data).mark_bar(
        color='#81c784',
        size=60
    ).encode(
        x=alt.X('modelo:N', title='Modelo'),
        y=alt.Y('total_1s:Q', title='Jailbreaks', scale=alt.Scale(domain=[0, data['total'].max()])),
        tooltip=['modelo', 'total_1s', 'total']
    ).properties(
        title="Jailbreaks por modelo",
        width=500,
        height=400
    )

    return chart

def percentage_chart_plot(df: DataFrame) -> Chart:
    # Lista para almacenar todas las predicciones desglosadas
    data_expanded = []

    # Recorrer cada fila del DataFrame
    for row in df['results']:
        parsed_results = ast.literal_eval(row)  # Convertir de string a estructura Python
        for model, values in parsed_results:
            for v in values:
                data_expanded.append({'modelo': model, 'valor': "Jailbreak" if v == 1 else "Protected"})

    # Crear un DataFrame con todos los valores individuales
    data = pd.DataFrame(data_expanded)

    # Agrupar y contar 0s y 1s por modelo
    agg_data = (
        data.groupby(['modelo', 'valor'], observed=False)
        .size()
        .reset_index(name='cuenta')
    )

    # Calcular porcentaje de cada valor por modelo
    agg_data['porcentaje'] = agg_data.groupby('modelo')['cuenta'].transform(lambda x: 100 * x / x.sum())

    # Escala de colores para 0 y 1
    color_scale = alt.Scale(domain=["Jailbreak", "Protected"], range=['#81c784', '#ea5656'])

    # Crear gráfico de barras apiladas normalizadas
    chart = alt.Chart(agg_data).mark_bar(size=60).encode(
        x=alt.X('modelo:N', title='Modelos'),
        y=alt.Y('porcentaje:Q', stack='normalize', title='Iteraciones'),
        color=alt.Color('valor:N', scale=color_scale, title='Valor'),
        order=alt.Order('valor:N', sort="ascending"),
        tooltip=['modelo', 'valor', 'porcentaje']
    ).properties(
        title="Distribución Jailbreaks por modelo LLM",
        width=500,
        height=400
    )

    return chart