# information about dataset

### Force Data (`force.csv`):

- Contains **6 features**:
  - `FP1_x`, `FP2_x`, `FP1_y`, `FP2_y`, `FP1_z`, `FP2_z`
  - Represent the ground reaction forces (in 3 axes: x, y, z) for two force plates.

### Marker Data (`marker.csv`):

- Contains **24 features**:
  - `L_FCC_x`, `L_FM1_x`, ..., `R_FM5_z`
  - Represent the 3D positions of markers placed on the left (L) and right (R) foot in three axes (x, y, z).

---

### Next Steps: Preprocessing and Feature Engineering

We will proceed with the following steps:

1. **Preprocessing**:

   - Combine both datasets to create a unified feature set.
   - Normalize/scale the data to ensure all features are on the same scale.

2. **Feature Engineering**:

   - Add aggregated features (e.g., average force, magnitude of marker movements).
   - Drop any unnecessary or redundant features.

3. **Model Training**:
   - Train a **binary classification model**:
     - Define labels: Generate a synthetic binary label for demonstration (e.g., "Normal" vs. "Abnormal").
   - Evaluate the model performance.

The preprocessing is complete, and the dataset is now ready for modeling. Here's what we achieved so far:

1. **Combined Features**:
   - Concatenated force and marker data into a single dataset with 30 features.
2. **Generated Labels**:

   - Created a synthetic binary label (`Normal=1`, `Abnormal=0`) for demonstration purposes.

3. **Scaled Features**:
   - Standardized all features to ensure they are on a comparable scale.
