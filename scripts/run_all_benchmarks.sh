#!/bin/bash

GRID_SIZE=$1
TARGET_FOLDER=$2

if [ -z "$GRID_SIZE" ] || [ -z "$TARGET_FOLDER" ]; then
    echo "Error: Debes indicar el tamaño de la malla y el tipo de carpeta a ejecutar."
    echo "Uso: $0 <3x3|4x4|5x5|8x8> <kernel|baseline|nonkernel>"
    exit 1
fi

if [[ "$TARGET_FOLDER" != "kernel" && "$TARGET_FOLDER" != "baseline" && "$TARGET_FOLDER" != "nonkernel" ]]; then
    echo "Error: Tipo de carpeta inválido. Opciones válidas: kernel, baseline, nonkernel"
    exit 1
fi

PROJECT_ROOT="/home/mbelda/Documents/GitHub/mbelda/ESL-CGRA-simulator"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

PYTHON_CMD="conda run -n core-v-mini-mcu python3"

BASE_BENCHMARKS_DIR="$PROJECT_ROOT/benchmarks/compigra_kernel/nonkernel_residual"
TARGET_DIR="$BASE_BENCHMARKS_DIR/$GRID_SIZE/$TARGET_FOLDER"
SCRIPTS_DIR="$PROJECT_ROOT/scripts"
SUMMARY_CSV="$BASE_BENCHMARKS_DIR/summary_metrics.csv"
HEADER_FUNC="void mmul_base(int inputX[NI][NK], int inputY[NK][NJ], int output[NI][NJ])"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

if [ ! -d "$TARGET_DIR" ]; then
    echo -e "${RED}Error: El directorio destino no existe: $TARGET_DIR${NC}"
    exit 1
fi

if [ ! -f "$SUMMARY_CSV" ]; then
    echo "Grid_Size,Benchmark,Execution_Cycles,Config_Cycles,Total_Cycles,Status" > "$SUMMARY_CSV"
fi

if [ "$TARGET_FOLDER" == "kernel" ]; then
    RUNNER_SCRIPT="$SCRIPTS_DIR/run_test_kernel.py"
else
    RUNNER_SCRIPT="$SCRIPTS_DIR/run_test_generic.py"
fi

echo -e "${BLUE}======================================================${NC}"
echo -e "${BLUE} PROCESANDO: Malla $GRID_SIZE | Modo: $TARGET_FOLDER ${NC}"
echo -e "${BLUE} Ruta Target: $TARGET_DIR ${NC}"
echo -e "${BLUE} Ejecutor Python: $(basename $RUNNER_SCRIPT) ${NC}"
echo -e "${BLUE}======================================================${NC}"

find "$TARGET_DIR" -type f -name "*.sat" | while read -r sat_path; do
    folder_path=$(dirname "$sat_path")
    sat_filename=$(basename "$sat_path")
    base_version_tag="${sat_filename%.sat}"
    
    base_inst_csv_filename="instructions_${base_version_tag}.csv"
    base_inst_csv_path="$folder_path/$base_inst_csv_filename"

    # PASO 0: Normalización (solo si no es modo kernel)
    if [ "$TARGET_FOLDER" != "kernel" ]; then
        echo -e "  ${YELLOW}0. Normalizando formato del archivo .sat...${NC}"
        $PYTHON_CMD -c '
import sys, re
sat_file = sys.argv[1]
with open(sat_file, "r") as f:
    lines = f.readlines()
out_lines = []
for line in lines:
    line_str = line.strip()
    if not line_str: continue
    match = re.search(r"\b(\d+)\b", line_str)
    if match and ("time" in line_str.lower() or line_str.startswith("T")):
        out_lines.append(f"T = {match.group(1)}")
    else:
        instructions = re.split(r"\s{2,}|\t+", line_str)
        for instr in instructions:
            clean = instr.strip()
            if clean: out_lines.append(clean)
if not out_lines or out_lines[-1] != "T = 0":
    out_lines.append("T = 0")
with open(sat_file, "w") as f:
    f.write("\n".join(out_lines) + "\n")
' "$sat_path"
    else
        echo -e "  ${YELLOW}0. Omitiendo normalización de .sat (Modo kernel)${NC}"
    fi

    # PASO 1: Generar CSV de instrucciones base
    echo -e "  ${YELLOW}1. Generando CSV de instrucciones ($base_inst_csv_filename)...${NC}"
    $PYTHON_CMD "$SCRIPTS_DIR/sat_to_csv.py" -i "$sat_path" -o "$base_inst_csv_filename"

    if [ ! -f "$base_inst_csv_path" ]; then
        echo -e "  ${RED}Error: No se pudo generar el archivo CSV en '$base_inst_csv_path'${NC}"
        echo "${GRID_SIZE},${base_version_tag},N/A,N/A,N/A,FAILED_SAT_CONVERSION" >> "$SUMMARY_CSV"
        continue
    fi

    # LÓGICA DE EJECUCIÓN
    if [ "$TARGET_FOLDER" == "kernel" ]; then
        # Lista fija de canales y de NKs para la categoría kernel
        CHANNELS=("c1:9:12" "c2:15:31" "c3:24:64" "c4:47:64")
        NK_VALUES=(11 23 33 65)

        for ch in "${CHANNELS[@]}"; do
            IFS=":" read -r c_name curr_NI curr_NJ <<< "$ch"
            for curr_NK in "${NK_VALUES[@]}"; do
                curr_tag="${base_version_tag}_${c_name}_k${curr_NK}"

                echo -e "\n${BLUE}[+] Procesando Kernel (${GRID_SIZE}):${NC} $curr_tag (NI=$curr_NI, NJ=$curr_NJ, NK=$curr_NK)"

                # Crear copia del CSV de instrucciones con la etiqueta completa
                channel_inst_csv_path="$folder_path/instructions_${curr_tag}.csv"
                cp "$base_inst_csv_path" "$channel_inst_csv_path"

                # Generar Config JSON
                json_cfg="$folder_path/${curr_tag}_cfg.json"
                $PYTHON_CMD "$SCRIPTS_DIR/compi_to_OE_csv.py" "$channel_inst_csv_path" --header "$HEADER_FUNC" --json-out "$json_cfg"

                # Ejecutar
                $PYTHON_CMD "$RUNNER_SCRIPT" \
                    --version-tag "$curr_tag" \
                    --config-json "$json_cfg" \
                    --benchmark-dir "$folder_path" \
                    --ni "$curr_NI" --nj "$curr_NJ" --nk "$curr_NK"

                # Limpieza
                rm -f "$json_cfg" "$channel_inst_csv_path"

                # Guardar métricas
                log_file="$folder_path/execution_${curr_tag}.log"
                if [ -f "$log_file" ] && grep -q "END" "$log_file"; then
                    exec_cycles=$(grep "Execution accurate cycles:" "$log_file" | awk -F': ' '{print $2}' | tr -d '\r')
                    config_cycles=$(grep "Config cycles:" "$log_file" | awk -F': ' '{print $2}' | tr -d '\r')
                    total_cycles=$(grep "Total cycles:" "$log_file" | awk -F': ' '{print $2}' | tr -d '\r')

                    echo "${GRID_SIZE},${curr_tag},${exec_cycles},${config_cycles},${total_cycles},SUCCESS" >> "$SUMMARY_CSV"
                    echo -e "  ${GREEN}✔ Métricas guardadas (${curr_tag})${NC}"
                else
                    echo "${GRID_SIZE},${curr_tag},N/A,N/A,N/A,FAILED_EXECUTION" >> "$SUMMARY_CSV"
                    echo -e "  ${RED}⚠ FAILED (${curr_tag})${NC}"
                fi
            done
        done
    else
        # LÓGICA HABITUAL PARA BASELINE Y NONKERNEL
        if [[ "$sat_filename" =~ _[iI]([0-9]+)_[jJ]([0-9]+)_[kK]([0-9]+) ]]; then
            NI="${BASH_REMATCH[1]}"; NJ="${BASH_REMATCH[2]}"; NK="${BASH_REMATCH[3]}"
        else
            g_val="${GRID_SIZE%%x*}"
            NI="$g_val"; NJ="$g_val"; NK="$g_val"
        fi

        json_cfg="$folder_path/${base_version_tag}_cfg.json"
        $PYTHON_CMD "$SCRIPTS_DIR/compi_to_OE_csv.py" "$base_inst_csv_path" --header "$HEADER_FUNC" --json-out "$json_cfg"

        $PYTHON_CMD "$RUNNER_SCRIPT" \
            --version-tag "$base_version_tag" \
            --config-json "$json_cfg" \
            --benchmark-dir "$folder_path" \
            --ni "$NI" --nj "$NJ" --nk "$NK"

        rm -f "$json_cfg"

        log_file="$folder_path/execution_${base_version_tag}.log"
        if [ -f "$log_file" ] && grep -q "END" "$log_file"; then
            exec_cycles=$(grep "Execution accurate cycles:" "$log_file" | awk -F': ' '{print $2}' | tr -d '\r')
            config_cycles=$(grep "Config cycles:" "$log_file" | awk -F': ' '{print $2}' | tr -d '\r')
            total_cycles=$(grep "Total cycles:" "$log_file" | awk -F': ' '{print $2}' | tr -d '\r')

            echo "${GRID_SIZE},${base_version_tag},${exec_cycles},${config_cycles},${total_cycles},SUCCESS" >> "$SUMMARY_CSV"
            echo -e "  ${GREEN}✔ Métricas guardadas en summary_metrics.csv${NC}"
        else
            echo "${GRID_SIZE},${base_version_tag},N/A,N/A,N/A,FAILED_EXECUTION" >> "$SUMMARY_CSV"
            echo -e "  ${RED}⚠ FAILED.${NC}"
        fi
    fi
done

echo -e "\n${GREEN}======================================================${NC}"
echo -e "${GREEN} ¡Flujo completado para $GRID_SIZE/$TARGET_FOLDER! ${NC}"
echo -e "${GREEN} Resumen guardado en: $SUMMARY_CSV ${NC}"
echo -e "${GREEN}======================================================${NC}"