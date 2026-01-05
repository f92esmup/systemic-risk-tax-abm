import numpy as np
import glob
import os

def diagnosticar():
    print(">>> INICIANDO DIAGNÓSTICO UNIVERSAL DE DATOS <<<")
    
    # Buscar en todas las carpetas de output
    rutas = glob.glob("output_data/**/*.npz", recursive=True)
    
    if not rutas:
        print(" [!] ALERTA: No se encontraron archivos .npz en output_data/")
        print("     Solución: Ejecuta 'python principal.py' para generar datos.")
        return

    print(f" Se encontraron {len(rutas)} archivos de simulación.")
    
    # Analizar el último archivo modificado (el más reciente)
    ultimo_archivo = max(rutas, key=os.path.getmtime)
    print(f" Inspeccionando el archivo más reciente: {ultimo_archivo}")
    
    try:
        datos = np.load(ultimo_archivo)
        claves = list(datos.keys())
        print(f" Claves detectadas: {claves}")
        
        # 1. Verificar Idioma de los Datos
        if "matriz_interbancaria" in claves:
            print(" [OK] Formato: ESPAÑOL (matriz_interbancaria detectada)")
            matriz = datos["matriz_interbancaria"]
        elif "L_bb" in claves:
            print(" [WARN] Formato: INGLÉS (L_bb detectada). Los scripts nuevos podrían fallar.")
            matriz = datos["L_bb"]
        else:
            print(" [ERROR] No se encuentra matriz de préstamos (ni 'L_bb' ni 'matriz_interbancaria').")
            return

        # 2. Verificar Contenido (¿Hay datos o ceros?)
        # Matriz shape: (Steps, B, B)
        if len(matriz) == 0:
             print(" [!] ALERTA CRÍTICA: La matriz está vacía (0 pasos).")
             return

        ultimo_paso = matriz[-1]
        total_prestamos = np.sum(ultimo_paso)
        num_enlaces = np.count_nonzero(ultimo_paso)
        
        print(f" Estadísticas del último paso (t={len(matriz)}):")
        print(f"  - Volumen Total Prestado: {total_prestamos:.2f}")
        print(f"  - Número de Enlaces Activos: {num_enlaces}")
        
        if total_prestamos == 0:
            print(" [!] ALERTA CRÍTICA: La simulación corrió pero NO hubo préstamos (Volumen 0).")
            print("     Causa probable: Tasas de interés muy altas, exceso de liquidez inicial o lógica de matching rota.")
        else:
            print(" [OK] La simulación contiene datos válidos para graficar.")
            
    except Exception as e:
        print(f" [ERROR] Fallo al leer el archivo: {e}")

if __name__ == "__main__":
    diagnosticar()
