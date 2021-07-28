import numpy as np
import pandas as pd
from sklearn.utils import shuffle


class AnomalyInjection:
    def __init__(self, df):
        """Init."""
        self.dataset = df.copy()
        self.columns = list(df.columns)

    def inject(self, y, column_changed, index):
        """Inject anomaly to dataset.

        Args:
            frac (float): random sample fraction from dataset
            y (categorical): column name as subtractor
            column_changed: column to changed
        """
        random_idx = index
        random_sample = self.dataset.loc[random_idx]
        z = np.random.choice(a=random_idx, size=1)[0]  # pick any random
        obs = self.dataset.loc[[z]]  # pick
        val = obs.loc[z, y]
        idx = (abs(random_sample[y] - val)).idxmax()
        x_ = random_sample.loc[idx, y]
        obs.loc[z, y] = x_
        obs.loc[z, "is_artificial"] = 1
        return obs

    def get_artificial(self, frac_slice=50, anomaly_size=20):
        """Inject artificial."""
        self.index = list(shuffle(self.dataset.index))
        res = []
        all_list = np.array_split(self.index, frac_slice)

        for i in all_list:
            slice_index = i
            for _ in range(anomaly_size):
                y, column_changed = np.random.choice(
                    list(set(self.columns) - set(["transaction_created_at_date"])),
                    size=2,
                    replace=False,
                )

                obs = self.inject(y=y, column_changed=column_changed, index=slice_index)
                res.append(obs.values.tolist()[0])

        self.columns.append("is_artificial")
        df_anomaly = pd.DataFrame(res, columns=self.columns)
        return df_anomaly


def inject_artifical_anomaly(df, frac_slice, anomaly_size):
    """Inject artificial anomaly."""
    artificial = AnomalyInjection(df)
    artificial_df = artificial.get_artificial(
        frac_slice=frac_slice, anomaly_size=anomaly_size
    )
    df["is_artificial"] = 0
    new_df = pd.concat([df, artificial_df])
    return new_df
