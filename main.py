import numpy as np
import pandas as pd
import os

output_dir = "export"
os.makedirs(output_dir, exist_ok=True)

participants = {
    "P01": {
        "eye": "eye_tracking_results/p1_eye_tracking_data.csv",
        "beh": "eye_tracking_results/p1_behavioral_data.csv",
    },
    "P02": {
        "eye": "eye_tracking_results/p2_eye_tracking_data.csv",
        "beh": "eye_tracking_results/p2_behavioral_data.csv",
    },
    "P03": {
        "eye": "eye_tracking_results/p3_eye_tracking_data.csv",
        "beh": "eye_tracking_results/p3_behavioral_data.csv",
    },
    "P04": {
        "eye": "eye_tracking_results/p4_eye_tracking_data.csv",
        "beh": "eye_tracking_results/p4_behavioral_data.csv",
    },
    "P05": {
        "eye": "eye_tracking_results/p5_eye_tracking_data.csv",
        "beh": "eye_tracking_results/p5_behavioral_data.csv",
    },
    "P06": {
        "eye": "eye_tracking_results/p6_eye_tracking_data.csv",
        "beh": "eye_tracking_results/p6_behavioral_data.csv",
    },
}
def compute_velocity(df):

    df = df.sort_values('timestamp_us').copy()

    # Convert timestamps to seconds
    df['time_s'] = df['timestamp_us'] / 1_000_000.0

    dx = df['x'].diff()
    dy = df['y'].diff()
    dt = df['time_s'].diff()

    dist = np.sqrt(dx**2 + dy**2)

    # Avoid division by zero
    dt[dt == 0] = np.nan
    velocity = dist / dt  # pixels per second

    df['velocity'] = velocity.fillna(0.0)
    return df

def ivt_detect_fixations_and_saccades(
    df, vel_threshold=500.0, min_fix_duration_s=0.05
):

    if 'velocity' not in df.columns or 'time_s' not in df.columns:
        df = compute_velocity(df)
    else:
        df = df.sort_values('time_s').copy()

    df['is_fixation'] = df['velocity'] < vel_threshold

    df['group_id'] = (
        df['is_fixation'] != df['is_fixation'].shift(fill_value=df['is_fixation'].iloc[0])
    ).cumsum()

    fix_rows = []
    sac_rows = []
    fix_id = 0
    sac_id = 0

    for group_id, group in df.groupby('group_id'):
        start_time = group['time_s'].iloc[0]
        end_time   = group['time_s'].iloc[-1]
        duration   = end_time - start_time  # seconds

        x_centroid = group['x'].mean()
        y_centroid = group['y'].mean()

        if group['is_fixation'].iloc[0]:
            # FIXATION
            if duration >= min_fix_duration_s:
                fix_rows.append({
                    'fix_id': fix_id,
                    'start_time_s': start_time,
                    'end_time_s': end_time,
                    'duration_s': duration,
                    'x': x_centroid,
                    'y': y_centroid,
                    'n_samples': len(group),
                })
                df.loc[group.index, 'fixation_id'] = fix_id
                fix_id += 1
        else:
            # SACCADE
            sac_rows.append({
                'sac_id': sac_id,
                'start_time_s': start_time,
                'end_time_s': end_time,
                'duration_s': duration,
                'x_start': group['x'].iloc[0],
                'y_start': group['y'].iloc[0],
                'x_end': group['x'].iloc[-1],
                'y_end': group['y'].iloc[-1],
                'n_samples': len(group),
            })
            df.loc[group.index, 'saccade_id'] = sac_id
            sac_id += 1

    fixations = pd.DataFrame(fix_rows)
    saccades  = pd.DataFrame(sac_rows)
    return fixations, saccades, df

def process_one_participant_eye(participant_id, filepath,
                               vel_threshold=500.0,
                               min_fix_duration_s=0.05):

    df_eye = pd.read_csv(filepath)

    # Keep only valid gaze samples
    df_eye = df_eye[df_eye['validity'] == 'Valid'].copy()

    if 'Participant_ID' not in df_eye.columns:
        df_eye['Participant_ID'] = participant_id

    # Detect fixations
    fix_df, sacc_df, df_out = ivt_detect_fixations_and_saccades(
        df_eye,
        vel_threshold=vel_threshold,
        min_fix_duration_s=min_fix_duration_s
    )

    if 'Image' in df_out.columns:
        img_map = (
            df_out.groupby('fixation_id')['Image']
            .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0])
        )
        fix_df = fix_df.merge(
            img_map.rename('Image'),
            left_on='fix_id',
            right_index=True,
            how='left'
        )
    else:
        fix_df['Image'] = 'Unknown'

    fix_df['Participant_ID'] = participant_id
    fix_df['duration_ms'] = fix_df['duration_s'] * 1000.0

    return fix_df

def process_one_participant_beh(participant_id, filepath):

    beh = pd.read_csv(filepath).copy()

    if 'Participant_ID' not in beh.columns:
        beh['Participant_ID'] = participant_id

    return beh

all_fixations_list = []
all_behaviour_list = []

for pid, files in participants.items():
    print(f"Processing participant {pid} ...")

    eye_file = files["eye"]
    beh_file = files["beh"]

    # Eye
    fix_p = process_one_participant_eye(pid, eye_file)
    all_fixations_list.append(fix_p)
    print(f"  -> {len(fix_p)} fixations")

    # Behaviour
    beh_p = process_one_participant_beh(pid, beh_file)
    all_behaviour_list.append(beh_p)
    print(f"  -> {len(beh_p)} behavioural trials")

# Combine across participants
all_fixations = pd.concat(all_fixations_list, ignore_index=True)
all_behaviour = pd.concat(all_behaviour_list, ignore_index=True)

print("\nCombined fixations:", all_fixations.head())


print("\nCombined behavioural data:"), all_behaviour.head()

def trial_metrics(fix_trial):

    f = fix_trial.sort_values('start_time_s')

    n_fix = len(f)
    mfd = f['duration_ms'].mean() if n_fix > 0 else np.nan
    total_dwell = f['duration_ms'].sum() if n_fix > 0 else 0.0

    if n_fix > 1:
        disp_x = f['x'].std()
        disp_y = f['y'].std()
    else:
        disp_x = 0.0
        disp_y = 0.0

    disp_r = np.sqrt(disp_x**2 + disp_y**2)

    if n_fix > 1:
        dx = f['x'].diff().fillna(0)
        dy = f['y'].diff().fillna(0)
        step_dist = np.sqrt(dx**2 + dy**2)
        path_len = step_dist.sum()
    else:
        path_len = 0.0

    return pd.Series({
        'n_fixations': n_fix,
        'mean_fix_dur_ms': mfd,
        'total_dwell_ms': total_dwell,
        'dispersion_x': disp_x,
        'dispersion_y': disp_y,
        'dispersion_r': disp_r,
        'scanpath_len_px': path_len,
    })

# per_trial = (
#     all_fixations
#     .groupby(['Participant_ID', 'Image'], as_index=False)
#     .apply(trial_metrics, include_groups=False)
#     .reset_index(drop=True)
# )

per_trial = (
    all_fixations
    .groupby(['Participant_ID', 'Image'])
    .apply(trial_metrics)
    .reset_index()
)


print("Per-trial eye metrics (first rows):", per_trial.head())

data = per_trial.merge(
    all_behaviour,
    on=['Participant_ID', 'Image'],
    how='inner'
)

print("Merged eye + behavioural data:", data.head())
print(f"\nNumber of merged rows: {len(data)}")
data['condition'] = np.where(data['Rating'] >= 3, 'familiar', 'unfamiliar')

print("Condition counts:")
print(data['condition'].value_counts())


summary_by_cond = (
    data
    .groupby('condition')
    .agg({
        'n_fixations': ['mean', 'std'],
        'mean_fix_dur_ms': ['mean', 'std'],
        'total_dwell_ms': ['mean', 'std'],
        'dispersion_r': ['mean', 'std'],
        'scanpath_len_px': ['mean', 'std'],
    })
)

print("Condition-wise eye metrics (mean ± SD):", summary_by_cond)

per_subject = (
    data
    .groupby('Participant_ID')
    .agg({
        'n_fixations': 'sum',           # total fixations
        'mean_fix_dur_ms': 'mean',      # average MFD across their images
        'total_dwell_ms': 'sum',        # total viewing time
        'dispersion_r': 'mean',         # average dispersion
        'scanpath_len_px': 'mean',      # average scanpath length
    })
    .rename(columns={'mean_fix_dur_ms': 'MFD_ms'})
    .reset_index()
)

print("Per-subject summary:", per_subject)

group_summary = pd.Series({
    'n_fixations_mean': per_subject['n_fixations'].mean(),
    'n_fixations_sd':   per_subject['n_fixations'].std(ddof=1),

    'MFD_ms_mean': per_subject['MFD_ms'].mean(),
    'MFD_ms_sd':   per_subject['MFD_ms'].std(ddof=1),

    'total_dwell_ms_mean': per_subject['total_dwell_ms'].mean(),
    'total_dwell_ms_sd':   per_subject['total_dwell_ms'].std(ddof=1),

    'dispersion_r_mean': per_subject['dispersion_r'].mean(),
    'dispersion_r_sd':   per_subject['dispersion_r'].std(ddof=1),

    'scanpath_len_px_mean': per_subject['scanpath_len_px'].mean(),
    'scanpath_len_px_sd':   per_subject['scanpath_len_px'].std(ddof=1),
})

print("Group-level summary (across participants):", group_summary.to_frame(name='value'))

per_subject_cond = (
    data
    .groupby(['Participant_ID', 'condition'])
    .agg({
        'n_fixations': 'sum',           # total fixations
        'mean_fix_dur_ms': 'mean',      # average MFD across images
        'total_dwell_ms': 'sum',        # total viewing time
        'dispersion_r': 'mean',         # average dispersion
        'scanpath_len_px': 'mean',      # average scanpath length
    })
    .rename(columns={'mean_fix_dur_ms': 'MFD_ms'})
    .reset_index()
)

print("Per-subject vs condition summary:")
print(per_subject_cond)

# export
# 1. per_trial
# add familiarity and condition column
per_trial['image_is_true'] = per_trial['Image'].str.startswith('true\\')
trial_rating = all_behaviour[['Participant_ID', 'Image', 'Rating']].drop_duplicates()

per_trial_with_cond = per_trial.merge(
    trial_rating,
    on=['Participant_ID', 'Image'],
    how='left'
)
per_trial_with_cond['condition'] = np.where(
    per_trial_with_cond['Rating'] >= 3, 'familiar', 'unfamiliar'
)

per_trial_file_enhanced = os.path.join(output_dir, "per_trial_with_cond.csv")
per_trial_with_cond.to_csv(per_trial_file_enhanced, index=False)
print(f"Saved enhanced per-trial metrics to {per_trial_file_enhanced}")

# 2. merged eye + behavioural data
merged_file = os.path.join(output_dir, "merged_eye_behavior.csv")
data.to_csv(merged_file, index=False)
print(f"Saved merged eye + behavioural data to {merged_file}")

# 3. summary by condition
summary_cond_file = os.path.join(output_dir, "summary_by_condition.csv")
summary_by_cond.to_csv(summary_cond_file)
print(f"Saved condition-wise summary to {summary_cond_file}")

# 4. per-subject summary
per_subject_file = os.path.join(output_dir, "per_subject_summary.csv")
per_subject.to_csv(per_subject_file, index=False)
print(f"Saved per-subject summary to {per_subject_file}")

# 5. group-level summary (Series -> DataFrame)
group_summary_file = os.path.join(output_dir, "group_summary.csv")
group_summary.to_frame(name='value').to_csv(group_summary_file)
print(f"Saved group-level summary to {group_summary_file}")

# 6. condition vs subject
per_subject_cond_file = os.path.join(output_dir, "per_subject_per_condition_summary.csv")
per_subject_cond.to_csv(per_subject_cond_file, index=False)
print(f"Saved per-subject × condition summary to {per_subject_cond_file}")