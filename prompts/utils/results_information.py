from matplotlib import pyplot as plt

from graph.exploit_result import ExploitResult


def summarize_results(results: list[ExploitResult]) -> dict[str, dict[str, int]]:
    summary = {}
    total_success = 0
    total_count = 0
    for result in results:
        method = result.method.split(" | ")[0]
        if method not in summary:
            summary[method] = {"success": 0, "total": 0}
        summary[method]["total"] += 1
        total_count += 1
        if result.success:
            summary[method]["success"] += 1
            total_success += 1
    summary["__TOTAL__"] = {"success": total_success, "total": total_count}
    return summary

def print_summary(summary: dict[str, dict[str, int]]):
    print("\n--- Resumen de aciertos ---")
    for method, stats in summary.items():
        if method == "__TOTAL__":
            continue
        success = stats["success"]
        total = stats["total"]
        percentage = (success / total * 100) if total > 0 else 0.0
        print(f"[{method}] Aciertos: {success} / {total} ({percentage:.2f}%)")
    global_stats = summary["__TOTAL__"]
    global_success = global_stats["success"]
    global_total = global_stats["total"]
    global_percentage = (global_success / global_total * 100) if global_total > 0 else 0.0
    print("\n[GLOBAL] Aciertos totales: {} / {} ({:.2f}%)".format(
        global_success, global_total, global_percentage
    ))

def plot_summary(summary: dict[str, dict[str, int]]):
    methods = []
    success_rates = []
    for method, stats in summary.items():
        if method == "__TOTAL__":
            continue
        total = stats["total"]
        success = stats["success"]
        rate = (success / total * 100) if total > 0 else 0.0
        methods.append(method)
        success_rates.append(rate)
    global_stats = summary.get("__TOTAL__", {"success": 0, "total": 0})
    global_rate = (global_stats["success"] / global_stats["total"] * 100) if global_stats["total"] > 0 else 0.0
    methods.append("GLOBAL")
    success_rates.append(global_rate)
    plt.figure(figsize=(8, 5))
    bars = plt.bar(methods, success_rates, color=['steelblue' if m != "GLOBAL" else 'darkred' for m in methods])
    plt.xlabel("Método")
    plt.ylabel("Tasa de éxito (%)")
    plt.title("Tasa de éxito por método de explotación")
    plt.ylim(0, 100)
    for bar, rate in zip(bars, success_rates):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 2, f"{rate:.1f}%", ha='center', va='bottom')
    plt.tight_layout()
    plt.show()