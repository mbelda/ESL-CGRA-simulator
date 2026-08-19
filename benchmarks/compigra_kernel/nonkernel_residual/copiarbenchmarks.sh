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

echo -e "${BLUE}Iniciando el proceso de copia con gestión de ficheros .sat únicos...${NC}\n"

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

    # 3. Extraer la subcategoría/sufijo tras parámetros c/g/k si existe
    suffix=""
    if [[ "$folder_name" =~ _g[0-9]+(_k[0-9]+)?(_c[0-9]+)?_(.+) ]]; then
        suffix="${BASH_REMATCH[3]}"
    elif [[ "$folder_name" =~ _c[0-9]+_g[0-9]+(_k[0-9]+)?_(.+) ]]; then
        suffix="${BASH_REMATCH[2]}"
    fi

    # Limpieza del sufijo para quitar residuos de cX, gX, kX
    if [ -n "$suffix" ]; then
        suffix=$(echo "$suffix" | sed -E 's/^(c[0-9]+_|g[0-9]+_|k[0-9]+_)+//g')
    fi

    # Ruta de la carpeta destino
    target_dir="$DST_DIR/$grid_size/$category"
    mkdir -p "$target_dir"
    
    # 4. Seleccionar el archivo .sat adecuado
    shopt -s nullglob
    sat_files=("$folder_path"/*.sat)
    shopt -u nullglob

    target_sat=""

    if [ ${#sat_files[@]} -eq 1 ]; then
        # REGLA: Si solo hay 1 archivo .sat, lo tomamos directamente sin importar el nombre
        target_sat="${sat_files[0]}"
    elif [ ${#sat_files[@]} -gt 1 ]; then
        # Si hay más de uno, buscamos el que tiene información extendida (out_N_*.sat)
        for f in "${sat_files[@]}"; do
            fname=$(basename "$f")
            if [[ "$fname" =~ ^out_${g_val}_.+ ]]; then
                target_sat="$f"
                break
            fi
        done
        # Si no coincidió con el patrón extendido, tomamos el primero que no sea el out_N.sat simple
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

    # 5. Copiar el archivo seleccionado si existe
    if [ -n "$target_sat" ] && [ -f "$target_sat" ]; then
        file_name=$(basename "$target_sat")
        
        # Construir el nombre final incluyendo el sufijo de subcategoría si aplica
        if [ -n "$suffix" ]; then
            base_name="${file_name%.sat}"
            final_name="${base_name}_${suffix}.sat"
        else
            final_name="$file_name"
        fi
        
        dest_file="$target_dir/$final_name"
        
        # Control de colisiones / duplicados
        if [ -f "$dest_file" ]; then
            echo -e "${YELLOW}[ALERTA] Conflicto de nombre:${NC} '$final_name' ya existe en '$grid_size/$category/'"
            conflict_filename="CONFLICTO_${folder_name}_${final_name}"
            echo -e "         Guardando como: ${RED}$conflict_filename${NC}"
            cp "$target_sat" "$target_dir/$conflict_filename"
        else
            cp "$target_sat" "$dest_file"
            echo -e "${GREEN}[OK] Copiado:${NC} $file_name -> $grid_size/$category/$final_name"
        fi
    fi

done

echo -e "\n${GREEN}¡Proceso finalizado con éxito!${NC}"