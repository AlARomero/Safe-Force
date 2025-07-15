from argparse import ArgumentError
import streamlit as st
import pandas as pd
import time
import config
import ollama
from main import run as run_graph
from utils.charts import total_chart_plot, percentage_chart_plot, jailbreak_type_chart_plot

@st.cache_data
def get_local_models():
    return [f"ollama/{model['model']}" for model in ollama.list()["models"]]

def app():
    st.set_page_config(page_title="Safe-Force", page_icon="🛡️", layout="wide")
    if 'started' not in st.session_state:
        st.session_state.started = False
    if 'done' not in st.session_state:
        st.session_state.done = False
    if 'opciones_agregadas' not in st.session_state:
        st.session_state.opciones_agregadas = []
    st.markdown("<div style='font-size: 48px; font-weight: bold;'>Safe-Force: Fuzzer Agent for Jailbreak LLM</div>", unsafe_allow_html=True)
    st.markdown("-----------------------")
    st.markdown("  \n")
    modelos_predefinidos = get_local_models() + [target for target in config.TARGETS if not target.__contains__("ollama/")]

    if not st.session_state.started:
        col1, col2, col3 = st.columns([1, 0.7, 2.3])
        with col1:
            with st.expander("Cómo añadir los modelos"):
                st.write("""Se debe introducir inicialmente la API con la que se gestiona el modelo:
                        'openai' para Openai, 'azure' para AzureOpenAI y 'ollama' para Ollama.
                        Si no se añade ninguno de los previos, se considera que se quiere realizar una ejecución local
                        a través de la API de HuggingFace.""")
            st.markdown("Selecciona algoritmos de jailbreak:")
            st.session_state.algoritmos = st.multiselect(label="Algoritmos disponibles", label_visibility="collapsed", accept_new_options=True, options=modelos_predefinidos)
        with col2:
            archivo = st.file_uploader("Cargar CSV", type="csv", label_visibility="collapsed")
        if archivo is not None:
            try:
                df = pd.read_csv(archivo)
                if list(df.columns) != ['Prompt padre', 'Entrada', 'Preguntas', 'Tipo', 'Respuestas por modelo', 'Mutador', 'Política de selección', 'Resultados por modelo']:
                    raise ArgumentError(None, "Error, csv inválido")
                st.session_state.df = df
                st.session_state.chart_percentage = percentage_chart_plot(st.session_state.df)
                st.session_state.chart_bare_results = total_chart_plot(st.session_state.df)
                st.session_state.chart_types = jailbreak_type_chart_plot(st.session_state.df)
                st.session_state.started = True
                st.session_state.done = True
                st.rerun()
            except Exception as e:
                st.info(e)

        st.markdown("  \n")
        st.markdown("  \n")

        if st.button("Iniciar"):
            st.session_state.started = True
            st.rerun()

    if st.session_state.started:
        if not st.session_state.done:
            st.button("Iniciar", disabled=True)

            with st.status("Procesando pasos...", expanded=True) as status:
                if any(algoritmo.__contains__("ollama/") for algoritmo in st.session_state.algoritmos):
                    st.write("🔎 Detectados modelos Ollama")
                    for modelo in [algoritmo for algoritmo in st.session_state.algoritmos if algoritmo.__contains__("ollama/")]:
                        st.write(f"📥 Instalando {modelo}")
                        try:
                            ollama.pull(modelo.split("ollama/")[1])
                        except:
                            st.write(f"❌ Ha ocurrido un error durante la descarga del modelo {modelo}")
                            time.sleep(0.5)
                            st.session_state.done = False
                            st.session_state.started = False
                            st.rerun()
                    st.write("✅ Modelos instalados")

                st.write("🔍 Ejecutando Safe-Force...")
                estado_final_grafo = run_graph(st.session_state.algoritmos)
                if estado_final_grafo is None:
                    st.write("❌ Ha ocurrido un error durante la ejecución")
                    time.sleep(0.5)
                    st.session_state.done = False
                    st.session_state.started = False
                else:
                    st.write("✅ Ejecución finalizada...")
                    time.sleep(0.5)
                    st.write("📊 Creando visualización")
                    try:
                        st.session_state.df = pd.read_csv(config.RESULT_FILE_PATH)
                        st.session_state.chart_percentage = percentage_chart_plot(st.session_state.df)
                        st.session_state.chart_bare_results = total_chart_plot(st.session_state.df)
                        st.session_state.chart_types = jailbreak_type_chart_plot(st.session_state.df)
                        status.update(label="Finalizado", state="complete")
                        time.sleep(0.5)
                        st.session_state.done = True
                    except:
                        st.write("❌ Ha ocurrido un error durante la obtención de resultados")
                        time.sleep(0.5)
                        st.session_state.done = False
                        st.session_state.started = False
                        st.cache_data.clear()
            st.rerun()

        if st.session_state.done:
            if st.button("Reiniciar"):
                st.session_state.chart_percentage = None
                st.session_state.chart_bare_results = None
                st.session_state.chart_types = None
                st.session_state.done = False
                st.session_state.started = False
                st.cache_data.clear()
                st.rerun()
            st.markdown("<div style='font-size: 32px; font-weight: bold;'>Datos Generados</div>", unsafe_allow_html=True)
            st.markdown("-----------------------")

            # Crear columnas para centrar la gráfica y la tabla
            col1, col2 = st.columns([2, 3])  # Ajusta el ancho de las columnas
            with col1:
                st.altair_chart(st.session_state.chart_percentage, use_container_width=True)
                st.altair_chart(st.session_state.chart_bare_results, use_container_width=True)
            with col2:
                st.altair_chart(st.session_state.chart_types, use_container_width=True)
                st.dataframe(st.session_state.df)

            # Botón para descargar CSV
            csv = st.session_state.df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Descargar CSV",
                data=csv,
                file_name='resultados.csv',
                mime='text/csv'
            )

if __name__ == "__main__":
    app()