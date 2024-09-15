import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime

class Visualization:
    def __init__(self, output_dir="test_results"):
        self.base_output_dir = output_dir
        self.output_dir = self._create_unique_output_dir()

    def _create_unique_output_dir(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_output_dir = os.path.join(self.base_output_dir, f"run_{timestamp}")
        if not os.path.exists(unique_output_dir):
            os.makedirs(unique_output_dir)
        return unique_output_dir

    def visualize_results(self, results):
        df = pd.DataFrame(results)

        csv_path = os.path.join(self.output_dir, "results.csv")
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")

        for temperature in df["temperature"].unique():
            subset_df = df[df["temperature"] == temperature]
            known_counts = subset_df["is_known"].value_counts()
            correct_counts = subset_df.groupby("is_known")["is_correct"].value_counts().unstack().fillna(0)
            correct_when_known_counts = subset_df[subset_df["is_known"] == True]["is_correct"].value_counts()
            score_distribution = subset_df["score"].dropna()

            fig, axes = plt.subplots(3, 2, figsize=(16, 24))

            sns.barplot(x=known_counts.index, y=known_counts.values, ax=axes[0, 0], hue=known_counts.index, palette="viridis", legend=False)
            axes[0, 0].set_title(f"Known Questions (temperature={temperature})", fontsize=16)
            axes[0, 0].set_xlabel("Is Known", fontsize=14)
            axes[0, 0].set_ylabel("Count", fontsize=14)
            axes[0, 0].tick_params(axis='x', labelsize=12)
            axes[0, 0].tick_params(axis='y', labelsize=12)
            for i, v in enumerate(known_counts.values):
                axes[0, 0].text(i, v + 0.5, str(v), ha='center', fontsize=12)

            correct_counts.plot(kind='bar', stacked=True, ax=axes[0, 1], color=["#4c72b0", "#55a868"])
            axes[0, 1].set_title(f"Correct Answers (temperature={temperature})", fontsize=16)
            axes[0, 1].set_xlabel("Is Known", fontsize=14)
            axes[0, 1].set_ylabel("Count", fontsize=14)
            axes[0, 1].tick_params(axis='x', labelsize=12)
            axes[0, 1].tick_params(axis='y', labelsize=12)
            for i, (known, row) in enumerate(correct_counts.iterrows()):
                for j, v in enumerate(row):
                    axes[0, 1].text(i, row[:j + 1].sum() - v / 2, str(int(v)), ha='center', fontsize=12)

            axes[1, 0].pie(known_counts, labels=known_counts.index, autopct='%1.1f%%', startangle=140,
                           colors=sns.color_palette("viridis", len(known_counts)))
            axes[1, 0].set_title(f"Known Questions Distribution (temperature={temperature})", fontsize=16)

            correct_counts.sum().plot(kind='pie', labels=["False", "True"], autopct='%1.1f%%', startangle=140,
                                      colors=sns.color_palette("viridis", 2), ax=axes[1, 1])
            axes[1, 1].set_title(f"Correct Answers Distribution (temperature={temperature})", fontsize=16)

            sns.barplot(x=correct_when_known_counts.index, y=correct_when_known_counts.values, ax=axes[2, 0], hue=correct_when_known_counts.index, palette="viridis", legend=False)
            axes[2, 0].set_title(f"Correct Answers When Known (temperature={temperature})", fontsize=16)
            axes[2, 0].set_xlabel("Is Correct", fontsize=14)
            axes[2, 0].set_ylabel("Count", fontsize=14)
            axes[2, 0].tick_params(axis='x', labelsize=12)
            axes[2, 0].tick_params(axis='y', labelsize=12)
            for i, v in enumerate(correct_when_known_counts.values):
                axes[2, 0].text(i, v + 0.5, str(v), ha='center', fontsize=12)

            sns.histplot(score_distribution, bins=10, kde=True, ax=axes[2, 1], color="#4c72b0")
            axes[2, 1].set_title(f"Score Distribution (temperature={temperature})", fontsize=16)
            axes[2, 1].set_xlabel("Score", fontsize=14)
            axes[2, 1].set_ylabel("Frequency", fontsize=14)
            axes[2, 1].tick_params(axis='x', labelsize=12)
            axes[2, 1].tick_params(axis='y', labelsize=12)

            plt.tight_layout()
            fig_path = os.path.join(self.output_dir, f"visualization_temperature_{temperature}.png")
            plt.savefig(fig_path)
            print(f"Visualization saved to {fig_path}")
            plt.show()