#!/bin/sh

set -e

echo "⏳ Esperando a que el servidor Ollama esté listo..."

until ollama list > /dev/null 2>&1; do
  sleep 2
done

echo "✅ Ollama está activo. Esperando a que el modelo llama3:8b esté listo..."
ollama pull "llama3:8b"

until ollama list | grep -q "llama3:8b"; do
  sleep 3
done

echo "🎉 Modelo llama3:8b disponible. Continuando..."
exec "$@"