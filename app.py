from argparse import ArgumentError

import streamlit as st
import pandas as pd
import time
import config
from main import run as run_graph
from utils.charts import total_chart_plot, percentage_chart_plot


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

    modelos_predefinidos = config.TARGETS

    if not st.session_state.started:
        col1, col2, col3 = st.columns([1, 0.7, 2.3])
        with col1:
            with st.expander("Cómo añadir los modelos"):
                st.write("""Para añadir modelos se debe introducir inicialmente la api con la que se gestionará el mismo:
                        openai para Openai, azure para Azure OpenAI y ollama para Ollama.
                        En caso de no añadir ninguno de los anteriores se considerará que se quiere una ejecución local
                        a través de la api HuggingFace.""")
            st.markdown("Selecciona algoritmos de jailbreak:")
            st.session_state.algoritmos = st.multiselect(label="Algoritmos disponibles", label_visibility="collapsed", accept_new_options=True, options=modelos_predefinidos)
        with col2:
            archivo = st.file_uploader("Cargar CSV", type="csv", label_visibility="collapsed")
        if archivo is not None:
            try:
                df = pd.read_csv(archivo)
                if list(df.columns) != ['prompt', 'response', 'parent', 'mutator', 'selection_policy', 'results']:
                    raise ArgumentError(None, "Error, csv invalido")
                st.session_state.df = df
                st.session_state.chart_percentage = percentage_chart_plot(st.session_state.df)
                st.session_state.chart_bare_results = total_chart_plot(st.session_state.df)
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
                # AQUI SE METE LA LOGICA DE LA APP
                st.write("🔍 Ejecutando Safe-Force...")
                estado_final_grafo = run_graph(st.session_state.algoritmos)
                if estado_final_grafo is None:
                    st.write("❌ Ha ocurrido un error durante la ejecución")
                    time.sleep(0.5)
                    st.session_state.done = False
                    st.session_state.started = False
                else:
                    st.write("✅ Ejecucion finalizada...")
                    time.sleep(0.5)
                    st.write("📊 Creando visualizacion")
                    try:
                        st.session_state.df = pd.read_csv(config.RESULT_FILE_PATH)
                        st.session_state.chart_percentage = percentage_chart_plot(st.session_state.df)
                        st.session_state.chart_bare_results = total_chart_plot(st.session_state.df)
                        status.update(label="Finalizado", state="complete")
                        time.sleep(0.5)
                        st.session_state.done = True
                    except:
                        st.write("❌ Ha ocurrido un error durante la obtencion de resultados")
                        time.sleep(0.5)
                        st.session_state.done = False
                        st.session_state.started = False
            st.rerun()

        if st.session_state.done:
            if st.button("Reiniciar"):
                st.session_state.done = False
                st.session_state.started = False
                st.rerun()

            st.markdown("<div style='font-size: 32px; font-weight: bold;'>Datos Generados</div>", unsafe_allow_html=True)
            st.markdown("-----------------------")

            # Crear columnas para centrar la gráfica y la tabla
            col1, col2 = st.columns([2, 3])  # Ajusta el ancho de las columnas
            with col1:
                st.altair_chart(st.session_state.chart_percentage, use_container_width=True)
                st.altair_chart(st.session_state.chart_bare_results, use_container_width=True)

            with col2:
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