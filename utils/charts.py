import pandas as pd
import altair as alt
from altair import Chart
from pandas.core.frame import DataFrame
import ast
from collections import defaultdict

def total_chart_plot(df: DataFrame) -> Chart:
    ones_count = defaultdict(int)
    total_count = defaultdict(int)

    for row in df['Resultados por modelo']:
        parsed_results = ast.literal_eval(row)
        for model, values in parsed_results:
            ones_count[model] += sum(v == 'Sí' for v in values)
            total_count[model] += sum(v in ('Sí', 'No') for v in values)

    data = pd.DataFrame({
        'modelo': list(total_count.keys()),
        'total': [total_count[m] for m in total_count],
        'total_exitos': [ones_count[m] for m in total_count]
    })

    chart = alt.Chart(data).mark_bar(
        color='#81c784',
        size=60
    ).encode(
        x=alt.X('modelo:N', title='Modelo'),
        y=alt.Y('total_exitos:Q', title='Jailbreaks', scale=alt.Scale(domain=[0, data['total'].max()])),
        tooltip=['modelo', 'total_exitos', 'total']
    ).properties(
        title="Jailbreaks por modelo",
        width=500,
        height=400
    )

    return chart

def percentage_chart_plot(df: DataFrame) -> Chart:
    data_expanded = []
    for row in df['Resultados por modelo']:
        parsed_results = ast.literal_eval(row)
        for model, values in parsed_results:
            for v in values:
                data_expanded.append({'modelo': model, 'valor': "Jailbreak" if v == 'Sí' else "Protegido"})

    data = pd.DataFrame(data_expanded)

    agg_data = (
        data.groupby(['modelo', 'valor'], observed=False)
        .size()
        .reset_index(name='cuenta')
    )

    agg_data['porcentaje'] = agg_data.groupby('modelo')['cuenta'].transform(lambda x: 100 * x / x.sum())

    color_scale = alt.Scale(domain=["Jailbreak", "Protegido"], range=['#81c784', '#ea5656'])

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

def jailbreak_type_chart_plot(df: pd.DataFrame) -> alt.Chart:
    data_expanded = []

    for i, row in df.iterrows():
        parsed_results = ast.literal_eval(row['Resultados por modelo'])
        jb_type = row['Tipo']

        for model, values in parsed_results:
            for v in values:
                if v == 'Sí':
                    data_expanded.append({'modelo': model, 'tipo': jb_type})

    data = pd.DataFrame(data_expanded)

    agg_data = (
        data.groupby(['modelo', 'tipo'], observed=False)
        .size()
        .reset_index(name='cuenta')
    )

    agg_data['porcentaje'] = agg_data.groupby('modelo')['cuenta'].transform(lambda x: 100 * x / x.sum())

    chart = alt.Chart(agg_data).mark_bar(size=60).encode(
        x=alt.X('modelo:N', title='Modelos'),
        y=alt.Y('porcentaje:Q', title='Distribución de tipos de Jailbreak'),
        color=alt.Color('tipo:N', title='Tipo de Jailbreak'),
        tooltip=['modelo', 'tipo', 'cuenta', 'porcentaje']
    ).properties(
        title="Distribución de tipos de Jailbreak por modelo LLM",
        width=500,
        height=400
    )

    return chart