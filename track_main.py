def initiate_tracks(measurements, threshold_km=1.0):
    """Initiate tracks and group measurements based on proximity."""
    tracks = []
    track_id = 0

    for measurement in measurements:
        x, y, z, mt = measurement
        associated = False
        for track in tracks:
            last_measurement = track[-1]
            lx, ly, lz, lmt = last_measurement
            distance = np.sqrt((x - lx)**2 + (y - ly)**2 + (z - lz)**2)
            if distance <= threshold_km * 1000:  # Convert km to meters
                track.append(measurement)
                associated = True
                break

        if not associated:
            tracks.append([measurement])
            track_id += 1

    return tracks

def main():
    """Main processing loop."""
    kalman_filter = CVFilter()
    csv_file_path = 'ttk_84_2.csv'

    try:
        measurements = read_measurements_from_csv(csv_file_path)
    except FileNotFoundError:
        print(f"File not found: {csv_file_path}")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    if not measurements:
        print("No measurements found in the CSV file.")
        return

    tracks = initiate_tracks(measurements)
    cov_inv = np.linalg.inv(np.eye(state_dim))  # Example covariance inverse matrix

    updated_states = []

    for group_idx, track_group in enumerate(tracks):
        print(f"Processing group {group_idx + 1}/{len(tracks)}")

        track_states = []
        reports = []

        for i, (x, y, z, mt) in enumerate(track_group):
            if i == 0:
                # Initialize filter state with the first measurement
                kalman_filter.initialize_filter_state(x, y, z, 0, 0, 0, mt)
            elif i == 1:
                # Initialize filter state with the second measurement and compute velocity
                prev_x, prev_y, prev_z = track_group[i-1][:3]
                dt = mt - track_group[i-1][3]
                vx = (x - prev_x) / dt
                vy = (y - prev_y) / dt
                vz = (z - prev_z) / dt
                kalman_filter.initialize_filter_state(x, y, z, vx, vy, vz, mt)
            else:
                kalman_filter.predict_step(mt)
                kalman_filter.initialize_measurement_for_filtering(x, y, z, mt)

            filtered_state = kalman_filter.Sf.flatten()[:3]
            r, az, el = cart2sph(filtered_state[0], filtered_state[1], filtered_state[2])
            track_states.append(filtered_state)
            reports.append([x, y, z])
            updated_states.append((mt, r, az, el))

        # Perform clustering, hypothesis generation, and association
        hypotheses, probabilities = perform_clustering_hypothesis_association(track_states, reports, cov_inv)

        # After association, use the most likely report for track update
        max_associations, _ = find_max_associations(hypotheses, probabilities, reports)
        for report_idx, track_idx in enumerate(max_associations):
            if track_idx != -1:
                report = reports[report_idx]
                kalman_filter.update_step(report)
                filtered_state = kalman_filter.Sf.flatten()[:3]
                r, az, el = cart2sph(filtered_state[0], filtered_state[1], filtered_state[2])
                updated_states.append((track_group[report_idx][3], r, az, el))

    # Plotting all data together
    plot_track_data(updated_states)

if __name__ == "__main__":
    main()
