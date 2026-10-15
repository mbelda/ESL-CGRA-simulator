import csv
import sys
import os

def analizar_uso_cgra(ruta_archivo):
    # Verificar si el archivo existe
    if not os.path.exists(ruta_archivo):
        print(f"Error: El archivo '{ruta_archivo}' no existe.")
        sys.exit(1)
        
    ciclos = []
    ciclo_actual = None
    filas_ciclo_actual = []
    
    # 1. Leer y agrupar el archivo por bloques de ciclos
    with open(ruta_archivo, mode='r', encoding='utf-8') as f:
        lector = csv.reader(f)
        for fila in lector:
            if not fila or all(celda.strip() == '' for celda in fila):
                continue 
            
            primera_celda = fila[0].strip()
            if primera_celda.isdigit():
                if ciclo_actual is not None:
                    ciclos.append((ciclo_actual, filas_ciclo_actual))
                
                ciclo_actual = int(primera_celda)
                filas_ciclo_actual = []
            else:
                instrucciones_limpias = [inst.strip().strip('"').upper() for inst in fila]
                while len(instrucciones_limpias) < 4:
                    instrucciones_limpias.append('')
                filas_ciclo_actual.append(instrucciones_limpias[:4])
        
        if ciclo_actual is not None:
            ciclos.append((ciclo_actual, filas_ciclo_actual))

    total_nops = 0
    ciclos_validos_conteo = 0

    # 2. Contar NOPs solo en ciclos válidos
    for num_ciclo, filas in ciclos:
        # Si el ciclo contiene 'EXIT', se ignora por completo
        tiene_exit = any('EXIT' in inst for fila in filas for inst in fila)
        
        if tiene_exit:
            print(f"Ciclo {num_ciclo} ignorado por contener la instrucción 'EXIT'.")
            continue 
        
        ciclos_validos_conteo += 1
        
        for fila in filas:
            for inst in fila:
                if inst == 'NOP' or not inst:  # Considera NOP o celda vacía como no-uso
                    total_nops += 1

    # 3. Aplicar la fórmula solicitada
    # Slots totales = número de ciclos válidos * 16 (4 filas x 4 columnas por ciclo)
    slots_totales = ciclos_validos_conteo * 16
    
    if slots_totales > 0:
        instrucciones_utiles = slots_totales - total_nops
        porcentaje_uso = (instrucciones_utiles / slots_totales) * 100
    else:
        instrucciones_utiles = 0
        porcentaje_uso = 0.0

    # 4. Mostrar resultados
    print("\n=== METRICAS DE USO DEL CGRA ===")
    print(f"Ciclos procesados (válidos): {ciclos_validos_conteo}")
    print(f"Slots totales disponibles ({ciclos_validos_conteo} x 16): {slots_totales}")
    print(f"Instrucciones NOP totales: {total_nops}")
    print(f"Instrucciones útiles ejecutadas: {instrucciones_utiles}")
    print(f"Porcentaje de uso de los PEs: {porcentaje_uso:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python script.py <nombre_del_archivo.csv>")
        sys.exit(1)
        
    archivo_csv = sys.argv[1]
    analizar_uso_cgra(archivo_csv)