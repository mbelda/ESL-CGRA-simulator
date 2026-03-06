#!/bin/bash

set -e

# === PATHS ===
OE_SIM_YUXUAN="/home/mbelda/Documents/GitHub/yuxuan/ESL-CGRA-simulator"
OE_SIM="/home/mbelda/Documents/GitHub/mbelda/CGRAs/ESL-CGRA-simulator"

SAT_TO_CSV_SCRIPT="$OE_SIM/scripts/sat_to_csv.py"

# === MAPPINGS ===
declare -A FOLDER_MAP
FOLDER_MAP["examples_BLAS"]="blas"
FOLDER_MAP["examples_SAT_ILP"]="satilp"
FOLDER_MAP["examples_unroll"]="unroll"

BENCHMARKS=("PCA" "Kalman_1" "Kalman_2")

WORKDIR="$(pwd)"

echo "Working directory: $WORKDIR"
echo "-------------------------------------"

for SRC_FOLDER in "${!FOLDER_MAP[@]}"; do

    DST_FOLDER="${FOLDER_MAP[$SRC_FOLDER]}"

    for BENCH in "${BENCHMARKS[@]}"; do

        SRC_PATH="$OE_SIM_YUXUAN/$SRC_FOLDER/$BENCH"

        [ -d "$SRC_PATH" ] || continue

        for SAT_FILE in "$SRC_PATH"/*.sat; do

            [ -e "$SAT_FILE" ] || continue

            FILENAME=$(basename "$SAT_FILE")

            # Extraer N de out_3_....sat
            if [[ "$FILENAME" =~ out_([0-9]+)_ ]]; then
                N="${BASH_REMATCH[1]}"
            else
                echo "No se pudo extraer N de $FILENAME"
                continue
            fi

            # Solo 3x3, 4x4, 5x5
            if [[ "$N" != "3" && "$N" != "4" && "$N" != "5" ]]; then
                continue
            fi

            echo "Procesando $FILENAME (N=$N)"

            # === 1️⃣ Copiar .sat al directorio actual ===
            cp "$SAT_FILE" "$WORKDIR/$FILENAME"

            LOCAL_SAT="$WORKDIR/$FILENAME"
            LOCAL_CSV="$WORKDIR/instructions_${FILENAME%.sat}.csv"

            # === 2️⃣ Ejecutar parser en local ===
            python3 "$SAT_TO_CSV_SCRIPT" "$LOCAL_SAT" "$LOCAL_CSV"

            # === 3️⃣ Crear destino ===
            DEST_BASE="$OE_SIM/benchmarks/compigra_blas_paper/$DST_FOLDER/$BENCH/${N}x${N}"
            mkdir -p "$DEST_BASE"

            mv "$LOCAL_CSV" "$DEST_BASE/"

            # === 4️⃣ Borrar .sat local ===
            rm "$LOCAL_SAT"

            echo "✔ Guardado en $DEST_BASE/"
            echo "-------------------------------------"

        done
    done
done

echo "Proceso terminado correctamente."
