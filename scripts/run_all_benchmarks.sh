#!/bin/bash

# Comprobar parámetro obligatorio de tamaño
GRID_SIZE=$1

if [ -z "$GRID_SIZE" ]; then
    echo "Error: Debes indicar el tamaño de la malla a ejecutar."
    echo "Uso: $0 <3x3|4x4|5x5|8x8>"
    exit 1
fi

# Definir la raíz del proyecto y exportar PYTHONPATH
PROJECT_ROOT="/home/mbelda/Documents/GitHub/mbelda/ESL-CGRA-simulator"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Comando ejecutor con Conda
PYTHON_CMD="conda run -n core-v-mini-mcu python3"

# Rutas del proyecto
BASE_BENCHMARKS_DIR="$PROJECT_ROOT/benchmarks/compigra_kernel/nonkernel_residual"
TARGET_DIR="$BASE_BENCHMARKS_DIR/$GRID_SIZE"
SCRIPTS_DIR="$PROJECT_ROOT/scripts"
SUMMARY_CSV="$BASE_BENCHMARKS_DIR/summary_metrics.csv"
HEADER_FUNC="void mmul_base(int inputX[NI][NK], int inputY[NK][NJ], int output[NI][NJ])"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

if [ ! -d "$TARGET_DIR" ]; then
    echo -e "${RED}Error: El directorio para la arquitectura '$GRID_SIZE' no existe: $TARGET_DIR${NC}"
    exit 1
fi

# Crear la cabecera del CSV acumulativo si aún no existe
if [ ! -f "$SUMMARY_CSV" ]; then
    echo "Grid_Size,Benchmark,Execution_Cycles,Config_Cycles,Total_Cycles" > "$SUMMARY_CSV"
fi

echo -e "${BLUE}======================================================${NC}"
echo -e "${BLUE} PROCESANDO Y EJECUTANDO BENCHMARKS: Malla $GRID_SIZE ${NC}"
echo -e "${BLUE}======================================================${NC}"

find "$TARGET_DIR" -type f -name "*.sat" | while read -r sat_path; do
    folder_path=$(dirname "$sat_path")
    sat_filename=$(basename "$sat_path")
    version_tag="${sat_filename%.sat}"
    
    # Nombre de archivo que espera cgra.py: instructions_<version_tag>.csv
    inst_csv_filename="instructions_${version_tag}.csv"
    inst_csv_path="$folder_path/$inst_csv_filename"

    # Regex insensible a mayúsculas/minúsculas para extraer I, J, K del primer bloque
    if [[ "$sat_filename" =~ _[iI]([0-9]+)_[jJ]([0-9]+)_[kK]([0-9]+) ]]; then
        NI="${BASH_REMATCH[1]}"
        NJ="${BASH_REMATCH[2]}"
        NK="${BASH_REMATCH[3]}"
    else
        # Dimensiones por defecto si no contiene dimensiones explícitas (ej. out_4.sat)
        NI=24
        NJ=24
        NK=24
    fi

    echo -e "\n${BLUE}[+] Benchmark detectado (${GRID_SIZE}):${NC} $version_tag (NI=$NI, NJ=$NJ, NK=$NK)"

    # PASO 1: Generar el archivo instructions_...csv desde el .sat
    echo -e "  ${YELLOW}1. Generando CSV de instrucciones desde SAT...${NC}"
    $PYTHON_CMD "$SCRIPTS_DIR/sat_to_csv.py" -i "$sat_path" -o "$inst_csv_filename"

    if [ ! -f "$inst_csv_path" ]; then
        echo -e "  ${RED}Error: No se pudo generar el archivo CSV en '$inst_csv_path'${NC}"
        continue
    fi

    # PASO 2: Transformar CSV y exportar la configuración JSON
    echo -e "  ${YELLOW}2. Procesando instrucciones y generando config JSON...${NC}"
    json_cfg="$folder_path/${version_tag}_cfg.json"
    $PYTHON_CMD "$SCRIPTS_DIR/compi_to_OE_csv.py" "$inst_csv_path" --header "$HEADER_FUNC" --json-out "$json_cfg"

    # PASO 3: Ejecutar la simulación pasando la carpeta contenedora
    echo -e "  ${YELLOW}3. Ejecutando test de simulación...${NC}"
    $PYTHON_CMD "$SCRIPTS_DIR/run_test_generic.py" \
        --version-tag "$version_tag" \
        --config-json "$json_cfg" \
        --benchmark-dir "$folder_path" \
        --ni "$NI" --nj "$NJ" --nk "$NK"

    # Limpieza de archivo temporal JSON de configuración
    rm -f "$json_cfg"

    # PASO 4: Analizar log y extraer métricas
    log_file="$folder_path/execution_${version_tag}.log"

    if [ -f "$log_file" ]; then
        if grep -q "END" "$log_file"; then
            exec_cycles=$(grep "Execution accurate cycles:" "$log_file" | awk -F': ' '{print $2}' | tr -d '\r')
            config_cycles=$(grep "Config cycles:" "$log_file" | awk -F': ' '{print $2}' | tr -d '\r')
            total_cycles=$(grep "Total cycles:" "$log_file" | awk -F': ' '{print $2}' | tr -d '\r')

            echo "${GRID_SIZE},${version_tag},${exec_cycles},${config_cycles},${total_cycles}" >> "$SUMMARY_CSV"
            echo -e "  ${GREEN}✔ Métricas guardadas en summary_metrics.csv (Exec: ${exec_cycles}, Config: ${config_cycles})${NC}"
        else
            echo -e "  ${RED}⚠ Advertencia: El log no finalizó correctamente (falta la palabra 'END').${NC}"
        fi
    else
        echo -e "  ${RED}⚠ Advertencia: No se encontró el archivo de log '$log_file'.${NC}"
    fi
done

echo -e "\n${GREEN}======================================================${NC}"
echo -e "${GREEN} ¡Flujo completado con éxito para la arquitectura $GRID_SIZE! ${NC}"
echo -e "${GREEN} Resumen guardado en: $SUMMARY_CSV ${NC}"
echo -e "${GREEN}======================================================${NC}"