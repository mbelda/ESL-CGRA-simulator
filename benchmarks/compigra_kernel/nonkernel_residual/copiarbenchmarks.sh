#!/bin/bash

# Definición de rutas origen y destino
SRC_DIR="/home/mbelda/Documents/GitHub/yuxuan/ESL-CGRA-simulator/examples_nonkernel_residual"
DST_DIR="/home/mbelda/Documents/GitHub/mbelda/ESL-CGRA-simulator/benchmarks/compigra_kernel/nonkernel_residual"

# Colores para avisos en terminal
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

if [ ! -d "$SRC_DIR" ]; then
    echo -e "${RED}Error: El directorio origen no existe: $SRC_DIR${NC}"
    exit 1
fi

echo -e "${BLUE}Iniciando el proceso de copia preservando el nombre completo de origen...${NC}\n"

for folder_path in "$SRC_DIR"/*/; do
    [ -d "$folder_path" ] || continue
    
    folder_name=$(basename "$folder_path")
    
    # 1. Determinar la categoría principal
    category=""
    case "$folder_name" in
        baseline_*)  category="baseline" ;;
        kernel_*)    category="kernel" ;;
        nonkernel_*) category="nonkernel" ;;
        *)           continue ;;
    esac
    
    # 2. Extraer el tamaño gN
    grid_size=""
    g_val=""
    if [[ "$folder_name" =~ _g([0-9]+)(_|$) ]]; then
        g_val="${BASH_REMATCH[1]}"
        grid_size="${g_val}x${g_val}"
    else
        continue
    fi
    
    # Permitir carpetas 3x3, 4x4, 5x5 y 8x8
    if [[ ! "$grid_size" =~ ^(3x3|4x4|5x5|8x8)$ ]]; then
        continue
    fi

    # Ruta de la carpeta destino
    target_dir="$DST_DIR/$grid_size/$category"
    mkdir -p "$target_dir"
    
    # 3. Seleccionar el archivo .sat adecuado
    shopt -s nullglob
    sat_files=("$folder_path"/*.sat)
    shopt -u nullglob

    target_sat=""

    if [ ${#sat_files[@]} -eq 1 ]; then
        target_sat="${sat_files[0]}"
    elif [ ${#sat_files[@]} -gt 1 ]; then
        for f in "${sat_files[@]}"; do
            fname=$(basename "$f")
            if [[ "$fname" =~ ^out_${g_val}_.+ ]]; then
                target_sat="$f"
                break
            fi
        done
        if [ -z "$target_sat" ]; then
            for f in "${sat_files[@]}"; do
                fname=$(basename "$f")
                if [ "$fname" != "out_${g_val}.sat" ]; then
                    target_sat="$f"
                    break
                fi
            done
        fi
    fi

    # 4. Copiar utilizando el nombre íntegro de la carpeta para evitar cualquier colisión
    if [ -n "$target_sat" ] && [ -f "$target_sat" ]; then
        file_name=$(basename "$target_sat")
        
        # El nombre del archivo se construye con la raíz base y toda la metadata del nombre de la carpeta
        final_name="${folder_name}.sat"
        dest_file="$target_dir/$final_name"
        
        # Control de colisiones / duplicados
        if [ -f "$dest_file" ]; then
            echo -e "${YELLOW}[ALERTA] Conflicto de nombre:${NC} '$final_name' ya existe en '$grid_size/$category/'"
            conflict_filename="CONFLICTO_${final_name}"
            echo -e "         Guardando como: ${RED}$conflict_filename${NC}"
            cp "$target_sat" "$target_dir/$conflict_filename"
        else
            cp "$target_sat" "$dest_file"
            echo -e "${GREEN}[OK] Copiado:${NC} $file_name -> $grid_size/$category/$final_name"
        fi
    fi

done

echo -e "\n${GREEN}¡Proceso finalizado con éxito!${NC}"