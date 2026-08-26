#!/bin/bash

# Verificar que se ha pasado una ruta como argumento
if [ -z "$1" ]; then
  echo "Uso: $0 /ruta/a/la/carpeta"
  exit 1
fi

DIRECTORIO="$1"

# Verificar si el directorio existe
if [ ! -d "$DIRECTORIO" ]; then
  echo "Error: El directorio '$DIRECTORIO' no existe."
  exit 1
fi

# Buscar y borrar los archivos que coincidan con la regla
find "$DIRECTORIO" -type f -name "memory_*.csv" -delete

echo "Proceso completado. Se han eliminado todos los archivos que coinciden."