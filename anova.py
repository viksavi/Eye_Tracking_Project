import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import os

output_dir = "export"

# load CSV
csv_file = os.path.join(output_dir, "per_trial_with_cond.csv")
df = pd.read_csv(csv_file)

# z-scored
df['n_fixations_z'] = df.groupby('Participant_ID')['n_fixations'].transform(
    lambda x: (x - x.mean()) / x.std(ddof=1)
)
df['mean_fix_dur_ms_z'] = df.groupby('Participant_ID')['mean_fix_dur_ms'].transform(
    lambda x: (x - x.mean()) / x.std(ddof=1)
)
df['dispersion_r_z'] = df.groupby('Participant_ID')['dispersion_r'].transform(
    lambda x: (x - x.mean()) / x.std(ddof=1)
)

# ±3σ outlier
df_clean_nfix = df[(df['n_fixations_z'] >= -3) & (df['n_fixations_z'] <= 3)]
df_clean_mfd  = df[(df['mean_fix_dur_ms_z'] >= -3) & (df['mean_fix_dur_ms_z'] <= 3)]
df_clean_disp = df[(df['dispersion_r_z'] >= -3) & (df['dispersion_r_z'] <= 3)]

# 2way-ANOVA: n_fixations
#                                    sum_sq     df         F    PR(>F)
# C(image_is_true)                 0.026731    1.0  0.027222  0.869140
# C(condition)                     0.845039    1.0  0.860563  0.354852
# C(image_is_true):C(condition)    0.117406    1.0  0.119563  0.729922

model_nfix = ols('n_fixations_z ~ C(image_is_true) * C(condition)', data=df_clean_nfix).fit()
anova_nfix = sm.stats.anova_lm(model_nfix, typ=2)
print("\nTwo-way ANOVA for n_fixations (z-scored):")
print(anova_nfix)

# 2way-ANOVA: mean_fix_dur_ms
#                                    sum_sq     df         F    PR(>F)
# C(image_is_true)                 0.678525    1.0  0.851459  0.357425
# C(condition)                     2.716619    1.0  3.409001  0.066550 .
# C(image_is_true):C(condition)    0.000663    1.0  0.000832  0.977023

model_mfd = ols('mean_fix_dur_ms_z ~ C(image_is_true) * C(condition)', data=df_clean_mfd).fit()
anova_mfd = sm.stats.anova_lm(model_mfd, typ=2)
print("\nTwo-way ANOVA for mean_fix_dur_ms (z-scored):")
print(anova_mfd)

# 2way-ANOVA: dispersion_r ()
#                                    sum_sq     df         F    PR(>F)
# C(image_is_true)                 1.292023    1.0  1.413781  0.236041
# C(condition)                     0.000565    1.0  0.000618  0.980188
# C(image_is_true):C(condition)    3.061990    1.0  3.350548  0.068884 .
model_disp = ols('dispersion_r_z ~ C(image_is_true) * C(condition)', data=df_clean_disp).fit()
anova_disp = sm.stats.anova_lm(model_disp, typ=2)
print("\nTwo-way ANOVA for dispersion_r (z-scored):")
print(anova_disp)