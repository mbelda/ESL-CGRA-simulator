#!/usr/bin/env python3
import csv
import sys
import argparse
import re

def parse_kernel(csv_path):
    """
    Lee el archivo CSV de un kernel y lo estructura en una lista de diccionarios.
    Cada elemento representa un ciclo con su id original y su matriz 4x4 de instrucciones.
    """
    cycles = []
    current_cycle_id = None
    current_rows = []

    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            # Detectar cabecera de ciclo (ej: "0,,," o "0")
            if len(row) > 0 and (row[0].isdigit() or (row[0] == '' and len(row) > 1 and row[1] == '')):
                # Si ya teníamos un ciclo acumulado, lo guardamos
                if current_cycle_id is not None and len(current_rows) == 4:
                    cycles.append({
                        'orig_id': current_cycle_id,
                        'matrix': current_rows
                    })
                # Inicializar nuevo ciclo
                current_cycle_id = int(row[0]) if row[0].isdigit() else 0
                current_rows = []
            else:
                # Asegurar que la fila tenga 4 columnas (rellenar con NOP si falta)
                cleaned_row = [instr.strip() for instr in row[:4]]
                while len(cleaned_row) < 4:
                    cleaned_row.append("NOP")
                current_rows.append(cleaned_row)
        
        # Guardar el último ciclo del archivo
        if current_cycle_id is not None and len(current_rows) == 4:
            cycles.append({
                'orig_id': current_cycle_id,
                'matrix': current_rows
            })
            
    return cycles

def clean_and_offset_kernel(cycles, cycle_offset, is_last_kernel):
    """
    1. Reemplaza 'EXIT' por 'NOP'.
    2. Elimina ciclos compuestos exclusivamente por 'NOP'.
    3. Genera el mapeo de IDs viejos a IDs nuevos.
    """
    processed_cycles = []
    old_to_new_id_map = {}
    current_new_id = cycle_offset

    for cycle in cycles:
        matrix = cycle['matrix']
        orig_id = cycle['orig_id']
        
        # 1. Reemplazar EXIT por NOP
        new_matrix = []
        is_all_nop = True
        for row in matrix:
            new_row = []
            for instr in row:
                if instr.upper() == "EXIT":
                    instr = "NOP"
                if instr.upper() != "NOP":
                    is_all_nop = False
                new_row.append(instr)
            new_matrix.append(new_row)
        
        # 2. Si el ciclo está lleno de NOPs, se elimina (salvo si es el último del flujo total por diseño)
        if is_all_nop and not (is_last_kernel and cycle == cycles[-1]):
            # No se añade a la lista, pero mapeamos su ID al siguiente ciclo válido disponible
            old_to_new_id_map[orig_id] = current_new_id
            continue
        
        # Asignar el nuevo ID de ciclo secuencializado
        old_to_new_id_map[orig_id] = current_new_id
        processed_cycles.append({
            'new_id': current_new_id,
            'matrix': new_matrix,
            'orig_id': orig_id # Guardado temporal para resolver los saltos
        })
        current_new_id += 1

    return processed_cycles, old_to_new_id_map, current_new_id

def update_branch_offsets(processed_cycles, old_to_new_id_map, global_id_map):
    """
    Busca instrucciones de salto (BEQ, BNE, BLT, BGE) y actualiza sus literales numéricos
    utilizando el diccionario de traducción de ciclos global.
    """
    # Regex para capturar saltos. Ejemplo típico: BEQ R1, R2, 14 o BNE R4, 12
    branch_regex = re.compile(r'\b(BEQ|BNE|BLT|BGE)\b', re.IGNORECASE)
    
    for cycle in processed_cycles:
        for r in range(4):
            for c in range(4):
                instr = cycle['matrix'][r][c]
                if branch_regex.search(instr):
                    # Extraer todos los números de la instrucción (el último suele ser el ciclo destino)
                    numbers = re.findall(r'\d+', instr)
                    if numbers:
                        old_target = int(numbers[-1])
                        # Buscamos en el mapa del kernel actual. Si no está, usamos el offset relativo
                        if old_target in old_to_new_id_map:
                            new_target = old_to_new_id_map[old_target]
                        else:
                            # Fallback por si salta a un ciclo que fue eliminado (lo manda al nuevo mapeado)
                            new_target = global_id_map.get(old_target, old_target)
                        
                        # Reemplazar el último número en el string de la instrucción
                        # Esto busca el último número exacto y lo cambia
                        parts = instr.rsplit(str(old_target), 1)
                        cycle['matrix'][r][c] = str(new_target).join(parts)

def save_merged_kernel(merged_cycles, output_path):
    """Escribe el resultado final en el CSV de salida respetando el formato."""
    with open(output_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for cycle in merged_cycles:
            # Escribir cabecera de ciclo "ID,,,"
            writer.writerow([cycle['new_id'], '', '', ''])
            # Escribir las 4 filas de instrucciones
            for row in cycle['matrix']:
                writer.writerow(row)

def main():
    parser = argparse.ArgumentParser(description="Fusión y renombrado dinámico de instrucciones CGRA desde archivos CSV.")
    parser.add_argument('--inputs', nargs='+', required=True, help="Lista de archivos CSV de entrada en orden de fusión.")
    parser.add_argument('--output', required=True, help="Nombre del archivo CSV de salida resultante.")
    
    args = parser.parse_args()
    
    if len(args.inputs) < 2:
        print("Error: Se necesitan al menos 2 kernels de entrada para realizar la fusión.")
        sys.exit(1)

    total_merged_cycles = []
    current_offset = 0

    for idx, csv_path in enumerate(args.inputs):
        print(f"Procesando: {csv_path}...")
        is_last_kernel = (idx == len(args.inputs) - 1)
        
        # 1. Parsear estructura del CSV
        raw_cycles = parse_kernel(csv_path)
        
        # 2. Quitar EXITs, limpiar ciclos vacíos de NOPs y aplicar desfase de ciclos
        processed_cycles, old_to_new_map, next_offset = clean_and_offset_kernel(
            raw_cycles, current_offset, is_last_kernel
        )
        
        # 3. Corrección de saltos locales en el bloque actual antes de acoplarlo
        update_branch_offsets(processed_cycles, old_to_new_map, old_to_new_map)
        
        # Acumular al dataset global
        total_merged_cycles.extend(processed_cycles)
        current_offset = next_offset

    # 4. Exportar el resultado final consolidado
    save_merged_kernel(total_merged_cycles, args.output)
    print(f"\n¡Éxito! Kernel unificado generado correctamente en: '{args.output}' con un total de {current_offset} ciclos.")

if __name__ == "__main__":
    main()