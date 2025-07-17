# Safe-Force: Guía de Instalación y Puesta en Marcha

Este documento detalla los pasos necesarios para instalar y ejecutar el sistema **Safe-Force**.

---

## 1. Instalación de Docker

El sistema se despliega mediante contenedores. Es imprescindible instalar Docker.

➡️ Visita la [web oficial de Docker](https://www.docker.com) y descarga **Docker**.

---

## 2. Descarga del Proyecto

➡️ Clona el repositorio:

```bash
git clone https://github.com/AlARomero/Safe-Force.git
```

---

## 3. Configuración de Variables de Entorno

Crea un archivo `.env` en la raíz del proyecto utilizando como plantilla el archivo existente `.env.template`.

Las variables a configurar son:

| Variable | Descripción |
|----------|-------------|
| `OPENAI_API_KEY` | Clave API válida para usar los servicios de OpenAI (opcional). |
| `DEVICE` | Define el dispositivo a usar: `cuda` (GPU Nvidia) o `cpu` (por defecto). |
| `AZURE_OPENAI_API_KEY` | Clave API para usar Azure OpenAI (opcional). |
| `AZURE_OPENAI_ENDPOINT` | Endpoint del servicio de Azure OpenAI. |
| `QUESTIONS_FILE` | Ruta al archivo con las preguntas a utilizar. |
| `PROMPT_TEMPLATES_FILE` | Ruta al archivo con las semillas (*prompt templates*). |
| `TARGETS` | Objetivos por defecto (no requiere modificación). |
| `POSSIBLE_MUTATORS` | Mutadores utilizados, separados por `|`. |
| `QUESTIONS_USED_NUMBER` | Número de preguntas por plantilla. |
| `STRATEGIST_TEMPERATURE` | Temperatura del agente estratega. |
| `MAX_ITERATIONS` | Número máximo de iteraciones del proceso. |
| `ENERGY` | Número de semillas generadas por iteración. |
| `OLLAMA_HOST` | Dirección del servicio Ollama (no modificar). |
| `RESULTS_PATH` | Ruta donde se guardarán los resultados (no modificar). |

---

## 4. Levantar el Servicio

Abre una terminal en la ruta del proyecto descargado y ejecuta:

```bash
docker compose up
```

Docker descargará las dependencias necesarias y lanzará los contenedores. Este proceso puede demorar. En caso de errores relacionados con health checks, vuelve a ejecutar el mismo comando.

## 5. Acceso a la Aplicación

Una vez finalizado el proceso de inicialización, accede a la aplicación desde:
➡️ http://localhost:8501