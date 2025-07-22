import pandas as pd
import matplotlib.pyplot as plt

# Lecture du fichier CSV
df = pd.read_csv("all_inference_multi_gpu_results.csv", na_values=[""])


def plot_comparison(df):
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import textwrap

    required_columns = {"Model", "GPU", "Error"}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        print(f"Missing required columns: {missing}")
        return

    df["Error"] = df["Error"].replace("", pd.NA)
    df_no_errors = df[df["Error"].isna()]
    if df_no_errors.empty:
        print("No valid data without errors to plot.")
        return

    metrics = [
        "Average FPS", "Model size (MB)", "Parameters",
        "mAP@0.5", "Precision", "Recall", "F1 score"
    ]
    available_metrics = [m for m in metrics if m in df_no_errors.columns]

    percent_metrics = {"mAP@0.5", "Precision", "Recall", "F1 score"}

    unit_map = {
        "mAP@0.5": "(%)",
        "Precision": "(%)",
        "Recall": "(%)",
        "F1 score": "(%)",
        "Model size": "(MB)",
        "Average FPS": "(FPS)",
        "Parameters": "(count)"
    }

    def wrap_label(text, width=20):
        return '\n'.join(textwrap.wrap(text, width=width))

    if not available_metrics:
        print("None of the expected metrics are available.")
        return

    # Bar plots with highlighting max for percent_metrics
    for metric in available_metrics:
        plt.figure(figsize=(12, 6))
        labels = df_no_errors["Model"] + " (" + df_no_errors["GPU"] + ")"
        values = pd.to_numeric(df_no_errors[metric], errors='coerce')

        valid = ~values.isna()
        labels = labels[valid]
        values = values[valid]

        bars = plt.bar(labels, values, color="lightgreen", edgecolor="black")
        plt.title(f"{metric} per Model/GPU", fontsize=12)
        plt.ylabel(metric, fontsize=10)
        plt.xlabel("Model (GPU)", fontsize=10)
        plt.xticks(rotation=30, ha='right', fontsize=8)

        # Find max value index for highlighting if metric in percent_metrics
        max_idx = None
        if metric in percent_metrics:
            max_idx = values.idxmax()

        for i, bar in enumerate(bars):
            height = bar.get_height()
            if not pd.isna(height):
                # Highlight max value in yellow and bold
                if max_idx is not None and labels.index[i] == max_idx:
                    color = 'gold'
                    weight = 'bold'
                    size = 9
                else:
                    color = 'black'
                    weight = 'normal'
                    size = 7

                plt.annotate(f'{height:.3f}',
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3),
                             textcoords="offset points",
                             ha='center', va='bottom',
                             fontsize=size,
                             color=color,
                             fontweight=weight)

        plt.tight_layout()
        plt.savefig(f"chart_{metric.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
        plt.close()

    # Pie charts with highlighting max value for percent_metrics
    for metric in available_metrics:
        pie_data = df_no_errors.dropna(subset=[metric])
        if pie_data.empty:
            print(f"No valid data for pie chart of '{metric}'. Skipping.")
            continue

        labels = pie_data["Model"] + " (" + pie_data["GPU"] + ")"
        if metric in percent_metrics:
            sizes = pd.to_numeric(pie_data[metric], errors='coerce').fillna(0) * 100
        else:
            sizes = pd.to_numeric(pie_data[metric], errors='coerce').fillna(0)

        if len(labels) > 8:
            top_n = sizes.nlargest(8)
            sizes = top_n
            labels = labels[top_n.index]

        plt.figure(figsize=(10, 10))
        wedges, texts = plt.pie(
            sizes,
            labels=['']*len(labels),
            startangle=90,
            wedgeprops={'edgecolor': 'black'}
        )

        # Find index of max value for highlighting
        max_idx = sizes.idxmax() if metric in percent_metrics else None

        for i, wedge in enumerate(wedges):
            ang = (wedge.theta2 + wedge.theta1) / 2
            x = 0.6 * np.cos(np.deg2rad(ang))
            y = 0.6 * np.sin(np.deg2rad(ang))

            rotation = ang
            if ang > 90 and ang < 270:
                rotation = ang + 180
                ha = 'right'
            else:
                ha = 'left'

            wrapped_label = wrap_label(labels.iloc[i])

            val_str = f"{sizes.iloc[i]:.1f}"
            if metric in percent_metrics:
                val_str += "%"

            # Highlight max value text color gold else white
            color = 'gold' if (max_idx is not None and sizes.index[i] == max_idx) else 'white'
            weight = 'bold' if color == 'gold' else 'normal'

            plt.text(
                x, y,
                f"{wrapped_label}\n{val_str}",
                ha='center',
                va='center',
                rotation=rotation,
                rotation_mode='anchor',
                fontsize=9,
                color=color,
                fontweight=weight
            )

        unit = unit_map.get(metric, "")
        plt.title(f"{metric} Distribution {unit}", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"pie_{metric.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
        plt.close()

    print("✅ All charts generated with max values highlighted for specified metrics.")


plot_comparison(df)
print("\n✅ Inference results saved to 'inference_multi_gpu_results.csv'")
print(df)