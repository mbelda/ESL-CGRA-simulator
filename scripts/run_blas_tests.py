import os
import re
import sys
from pathlib import Path
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

GREEN = "\033[0;32m"
BLUE = "\033[0;34m"
YELLOW = "\033[1;33m"
RED = "\033[0;31m"
NC = "\033[0m"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BLAS_DIR = PROJECT_ROOT / "benchmarks" / "compigra_kernel" / "blas"
CGRA_PY_PATH = PROJECT_ROOT / "cgra.py"
SUMMARY_CSV = BLAS_DIR / "summary_metrics.csv"


def update_cgra_py(n_rows: int, n_cols: int):
    if not CGRA_PY_PATH.exists():
        return
    with open(CGRA_PY_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    new_content = re.sub(r"^N_ROWS\s*=\s*\d+", f"N_ROWS = {n_rows}", content, flags=re.MULTILINE)
    new_content = re.sub(r"^N_COLS\s*=\s*\d+", f"N_COLS = {n_cols}", new_content, flags=re.MULTILINE)
    with open(CGRA_PY_PATH, "w", encoding="utf-8") as f:
        f.write(new_content)


def init_summary_csv():
    if not SUMMARY_CSV.exists():
        with open(SUMMARY_CSV, "w", encoding="utf-8") as f:
            f.write("Benchmark_Type,Grid_Size,Version,Execution_Cycles,Config_Cycles,Total_Cycles,Memory_Cycles,Arithmetic_Cycles,Status\n")


def parse_metrics_from_output(output_text):
    metrics = {
        "exec_cycles": "N/A",
        "config_cycles": "N/A",
        "total_cycles": "N/A",
        "memory_cycles": "N/A",
        "arithmetic_cycles": "N/A",
    }
    for line in output_text.splitlines():
        line = line.strip()
        if "Execution accurate cycles:" in line:
            metrics["exec_cycles"] = line.split(":")[-1].strip()
        elif "Config cycles:" in line:
            metrics["config_cycles"] = line.split(":")[-1].strip()
        elif "Total cycles:" in line:
            metrics["total_cycles"] = line.split(":")[-1].strip()
        elif "Memory cycles:" in line:
            metrics["memory_cycles"] = line.split(":")[-1].strip()
        elif "Arithmetic cycles:" in line:
            metrics["arithmetic_cycles"] = line.split(":")[-1].strip()
    return metrics


def run_all_tests():
    init_summary_csv()

    # REGEX ACTUALIZADA: Solo empareja si NO tiene _ctx en el nombre
    instr_pattern = re.compile(r"^instructions_(\d+)_(IJK\d+)\.csv$")

    print(f"{BLUE}======================================================{NC}")
    print(f"{BLUE} PROCESANDO BENCHMARKS BLAS                           {NC}")
    print(f"{BLUE} Ruta Base: {BLAS_DIR} {NC}")
    print(f"{BLUE}======================================================{NC}\n")

    for root, dirs, files in os.walk(BLAS_DIR):
        root_path = Path(root)

        notebooks = list(root_path.glob("test_*.ipynb"))
        if not notebooks:
            continue

        nb_path = notebooks[0]
        rel_parts = root_path.relative_to(BLAS_DIR).parts
        benchmark_type = rel_parts[0] if len(rel_parts) > 0 else "unknown"
        grid_size = rel_parts[1] if len(rel_parts) > 1 else "unknown"

        for file in sorted(files):
            # Omitir directamente cualquier archivo que contenga '_ctx'
            if "_ctx" in file:
                continue

            match = instr_pattern.match(file)
            if match:
                cgra_n = int(match.group(1))
                ijk_str = match.group(2)
                data_size = int(re.search(r"\d+", ijk_str).group())

                version_str = f"_{cgra_n}_{ijk_str}"

                print(f"{BLUE}[+] Ejecutando ({benchmark_type} | {grid_size}):{NC} {version_str} (CSV: {file})")

                update_cgra_py(n_rows=cgra_n, n_cols=cgra_n)

                try:
                    with open(nb_path, "r", encoding="utf-8") as f:
                        nb = nbformat.read(f, as_version=4)

                    # Reemplazar asignaciones locales
                    for cell in nb.cells:
                        if cell.cell_type == "code":
                            lines = cell.source.splitlines()
                            new_lines = []
                            for line in lines:
                                line_strip = line.strip()
                                if line_strip.startswith("SIZE =") or line_strip.startswith("SIZE="):
                                    new_lines.append(f"SIZE = {data_size}")
                                elif line_strip.startswith("kernel_name =") or line_strip.startswith("kernel_name="):
                                    new_lines.append("kernel_name = '.'")
                                elif line_strip.startswith("version =") or line_strip.startswith("version="):
                                    new_lines.append(f"version = '{version_str}'")
                                else:
                                    new_lines.append(line)
                            cell.source = "\n".join(new_lines)

                    # Inyección en la cabecera
                    override_code = (
                        f"import sys\n"
                        f"if '{PROJECT_ROOT}' not in sys.path:\n"
                        f"    sys.path.insert(0, '{PROJECT_ROOT}')\n"
                        f"import cgra\n"
                        f"cgra.N_ROWS = {cgra_n}\n"
                        f"cgra.N_COLS = {cgra_n}\n"
                        f"SIZE = {data_size}\n"
                        f"dim = {data_size}\n"
                        f"kernel_name = '.'\n"
                        f"version = '{version_str}'\n"
                        f"CGRA_N_ROWS = {cgra_n}\n"
                        f"CGRA_N_COLS = {cgra_n}\n"
                    )

                    nb.cells.insert(0, nbformat.v4.new_code_cell(source=override_code))

                    os.chdir(root_path)

                    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
                    executed_nb, _ = ep.preprocess(nb, {"metadata": {"path": str(root_path)}})

                    os.chdir(PROJECT_ROOT)

                    nb_output_text = ""
                    for cell in executed_nb.cells:
                        if cell.cell_type == "code" and "outputs" in cell:
                            for out in cell["outputs"]:
                                if out.output_type == "stream":
                                    nb_output_text += out.text
                                elif out.output_type in ["execute_result", "display_data"]:
                                    if "text/plain" in out.get("data", {}):
                                        nb_output_text += out["data"]["text/plain"] + "\n"

                    metrics = parse_metrics_from_output(nb_output_text)

                    with open(SUMMARY_CSV, "a", encoding="utf-8") as f:
                        f.write(
                            f"{benchmark_type},{grid_size},{version_str},"
                            f"{metrics['exec_cycles']},{metrics['config_cycles']},{metrics['total_cycles']},"
                            f"{metrics['memory_cycles']},{metrics['arithmetic_cycles']},SUCCESS\n"
                        )

                    print(f"  {GREEN}✔ Métricas guardadas ({version_str}):{NC}")
                    print(f"     - Exec: {metrics['exec_cycles']} | Config: {metrics['config_cycles']} | Total: {metrics['total_cycles']}")
                    print(f"     - Memory: {metrics['memory_cycles']} | Arith: {metrics['arithmetic_cycles']}\n")

                except Exception as e:
                    os.chdir(PROJECT_ROOT)
                    print(f"  {RED}⚠ FAILED ({version_str}): {e}{NC}\n")
                    with open(SUMMARY_CSV, "a", encoding="utf-8") as f:
                        f.write(f"{benchmark_type},{grid_size},{version_str},N/A,N/A,N/A,N/A,N/A,FAILED_EXECUTION\n")

    print(f"{GREEN}======================================================{NC}")
    print(f"{GREEN} ¡Flujo completado! CSV: {SUMMARY_CSV} {NC}")
    print(f"{GREEN}======================================================{NC}")


if __name__ == "__main__":
    run_all_tests()