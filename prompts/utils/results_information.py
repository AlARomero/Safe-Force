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
    counts = []
    for method, stats in summary.items():
        if method == "__TOTAL__":
            continue
        total = stats["total"]
        success = stats["success"]
        rate = (success / total * 100) if total > 0 else 0.0
        methods.append(method)
        success_rates.append(rate)
        counts.append((success, total))
    sorted_data = sorted(zip(methods, success_rates, counts), key=lambda x: x[1], reverse=True)
    methods = [m for m, _, _ in sorted_data]
    success_rates = [r for _, r, _ in sorted_data]
    counts = [c for _, _, c in sorted_data]
    global_stats = summary.get("__TOTAL__", {"success": 0, "total": 0})
    global_rate = (global_stats["success"] / global_stats["total"] * 100) if global_stats["total"] > 0 else 0.0
    methods.append("GLOBAL")
    success_rates.append(global_rate)
    counts.append((global_stats["success"], global_stats["total"]))
    method_colors = {
        'PromptInject': '#4e79a7',
        'MasterKey': '#59a14f',
        'GPTFuzzer': '#b07aa1',
        'GLOBAL': '#e15759'
    }
    plt.figure(figsize=(10, 6))
    colors = [method_colors.get(m, '#797979') for m in methods]  # gris para métodos no definidos
    bars = plt.bar(methods, success_rates, color=colors)
    plt.xlabel("Método", fontsize=12)
    plt.ylabel("Tasa de éxito (%)", fontsize=12)
    plt.title("Tasa de éxito por método de explotación", fontsize=14, pad=20)
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar, rate, (success, total), method in zip(bars, success_rates, counts, methods):
        yval = bar.get_height()
        text = f"{rate:.1f}%\n({success}/{total})"
        va = 'bottom' if yval < 90 else 'top'
        color = 'black' if method != 'GLOBAL' else 'white'
        plt.text(bar.get_x() + bar.get_width() / 2,
                 yval + 2 if va == 'bottom' else yval - 2,
                 text,
                 ha='center',
                 va=va,
                 color=color,
                 fontsize=10)
    plt.tight_layout()
    plt.show()