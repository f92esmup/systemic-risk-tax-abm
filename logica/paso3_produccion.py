import numpy as np
from parametros import Parametros


def ejecutar_paso3(modelo):
    """
    Paso 3: Producción y Asignación de Capital (Trabajo).
    - Contratación / Despido de trabajadores (Mercado Laboral).
    - Actualización de producción real basada en trabajadores obtenidos.
    """

    # 1. Current state
    current_workers = np.sum(modelo.matriz_laboral, axis=0)  # (F,)

    # Target was set in Step 1
    target_workers = modelo.estado_firmas[:, Parametros.IDX_FIRM_WORKERS].astype(int)

    delta = target_workers - current_workers

    # Identify firms that need change

    # --- FIRING (delta < 0) ---
    firing_firms = np.where(delta < 0)[0]
    for f in firing_firms:
        n_fire = abs(delta[f])
        # Find current employees: (H,) boolean mask -> indices
        employee_indices = np.where(modelo.matriz_laboral[:, f] == 1)[0]

        if len(employee_indices) > 0:
            n_fire = min(n_fire, len(employee_indices))
            fired_indices = modelo.rng.choice(
                employee_indices, size=n_fire, replace=False
            )
            # Update Matrix
            modelo.matriz_laboral[fired_indices, f] = 0

    # --- HIRING (delta > 0) ---
    hiring_firms = np.where(delta > 0)[0]

    # Identify unemployed pool (Dynamic based on firings just happened)
    # Sum rows: if 0, unemployed.
    employment_status = np.sum(modelo.matriz_laboral, axis=1)
    unemployed_indices = np.where(employment_status == 0)[0]

    # Shuffle unemployed pool once
    modelo.rng.shuffle(unemployed_indices)
    pool_ptr = 0
    total_unemployed = len(unemployed_indices)

    for f in hiring_firms:
        n_hire = delta[f]

        # Check availability
        remaining_in_pool = total_unemployed - pool_ptr
        if remaining_in_pool <= 0:
            break  # No more workers

        # Hire
        actual_hire = min(n_hire, remaining_in_pool)
        new_hires = unemployed_indices[pool_ptr : pool_ptr + actual_hire]

        # Update Matrix
        modelo.matriz_laboral[new_hires, f] = 1
        pool_ptr += actual_hire

    # Final Sync: Update Firm State to match actual Matrix (Rationing)
    real_workers = np.sum(modelo.matriz_laboral, axis=0)
    modelo.estado_firmas[:, Parametros.IDX_FIRM_WORKERS] = real_workers

    # Update Production and Wages based on REAL workers
    modelo.estado_firmas[:, Parametros.IDX_FIRM_PROD] = real_workers * Parametros.alpha
    modelo.estado_firmas[:, Parametros.IDX_FIRM_WAGES] = real_workers * Parametros.WAGE
