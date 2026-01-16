import os
import pandas as pd
import numpy as np


class SimulationLogger:
    def __init__(self):
        # Buffers para acumular datos en memoria
        # Estructura: self.agents_buffer[agent_type] = lista de dataframes
        self.agents_buffer = {}
        self.networks_buffer = {}  # self.networks_buffer[network_name] = lista de dataframes (edgelist o wrapper denso)
        self.globals_buffer = []  # lista de dataframes o diccionarios

    def log_step(self, t, agents_data, networks_data):
        """
        Acumula los datos del paso t en los buffers.
        agents_data: dict {nombre: pd.DataFrame}
        networks_data: dict {nombre: np.array (matriz)}
        """
        # 1. Agentes
        for name, df in agents_data.items():
            if name == "globals":
                # Manejo especial para globales (métricas escalares)
                df_copy = df.copy()
                df_copy["t"] = t
                self.globals_buffer.append(df_copy)
                continue

            if name not in self.agents_buffer:
                self.agents_buffer[name] = []

            # Copiar y añadir columna de tiempo
            df_copy = df.copy()
            df_copy["t"] = t
            self.agents_buffer[name].append(df_copy)

        # 2. Redes (Matrices)
        for name, matrix in networks_data.items():
            if name not in self.networks_buffer:
                self.networks_buffer[name] = []

            # Estrategia: Guardar siempre como Edgelist (source, target, weight, t)
            # Esto es mucho más eficiente para matrices dispersas y para análisis posterior en grafos.
            rows, cols = np.nonzero(matrix)
            vals = matrix[rows, cols]

            if len(vals) > 0:
                df_edge = pd.DataFrame({"source": rows, "target": cols, "weight": vals})
                df_edge["t"] = t
                self.networks_buffer[name].append(df_edge)

    def flush(self, run_id, output_dir="outputdata"):
        """
        Escribe los buffers concatenados a disco en formato Parquet.
        Estructura: output_dir/run_{id}/{component}.parquet
        """
        # Si run_id ya empieza con run_, no duplicar prefijo
        if run_id.startswith("run_"):
            dirname = run_id
        else:
            dirname = f"run_{run_id}"

        run_dir = os.path.join(output_dir, dirname)

        os.makedirs(run_dir, exist_ok=True)

        # print(f"   [Logger] Flushing data for run {run_id}...")

        # 1. Agentes y Globales
        # Unir buffer de agentes
        for name, buffer_list in self.agents_buffer.items():
            if not buffer_list:
                continue

            full_df = pd.concat(buffer_list, ignore_index=True)

            # Optimización de tipos si es posible
            # ...

            save_path = os.path.join(run_dir, f"{name}.parquet")
            full_df.to_parquet(save_path, compression="snappy")

        # Globales se procesaron como "agents" si venian en el dict,
        # pero si los separamos en self.globals_buffer:
        if self.globals_buffer:
            full_globals = pd.concat(self.globals_buffer, ignore_index=True)
            full_globals.to_parquet(
                os.path.join(run_dir, "globals.parquet"), compression="snappy"
            )

        # 2. Redes
        for name, buffer_list in self.networks_buffer.items():
            if not buffer_list:
                continue

            full_network_df = pd.concat(buffer_list, ignore_index=True)
            save_path = os.path.join(run_dir, f"{name}.parquet")
            full_network_df.to_parquet(save_path, compression="snappy")

        # Limpiar buffers tras flush
        self.clear()
        # print("   [Logger] Done.")

    def clear(self):
        self.agents_buffer = {}
        self.networks_buffer = {}
        self.globals_buffer = []
