import numpy as np

def extract_features(csi_matrix):
    """
    csi_matrix:
        Shape = (Number_of_Packets, 128)

    Returns:
        Feature Vector
    """

    features = []

    # Mean
    features.extend(np.mean(csi_matrix, axis=0))

    # Standard Deviation
    features.extend(np.std(csi_matrix, axis=0))

    # Maximum
    features.extend(np.max(csi_matrix, axis=0))

    # Minimum
    features.extend(np.min(csi_matrix, axis=0))

    # Range
    features.extend(np.ptp(csi_matrix, axis=0))

    # RMS (Root Mean Square)
    features.extend(np.sqrt(np.mean(np.square(csi_matrix), axis=0)))

    return np.array(features)