from scipy.stats.qmc import LatinHypercube, scale
import numpy as np
import tensorflow as tf

def gen_ini_LHS(sim_input):
    n_variables = len(sim_input.opt_variable_name)
    lhs = LatinHypercube(d=n_variables)
    samples = lhs.random(n=sim_input.lhs_sample_num)
    scaled_samples = scale(samples, sim_input.opt_variable_lower, sim_input.opt_variable_upper)
    header = np.array(sim_input.opt_variable_name).reshape(1, -1)
    combined_samples = np.vstack([header, scaled_samples])
    return combined_samples

def check_overlap(new_sample, existing_samples, threshold=1e-3):
    """
    Check if the new sample overlaps with existing samples.
    """
    for sample in existing_samples:
        if np.all(np.abs(new_sample - sample) < threshold):
            return True
    return False

def gen_additional_LHS(existing_samples, num_new_samples, opt_variable_lower, opt_variable_upper, max_attempts=1000, batch_size=100):
    """
    Generates additional LHS samples while considering existing samples.
    """
    num_variables = len(opt_variable_lower)
    lhs = LatinHypercube(d=num_variables)
    
    # Convert existing samples to float
    headers = existing_samples[0]
    data = existing_samples[1:]
    data = np.array(data, dtype=float)
    
    new_samples = []
    attempts = 0

    while len(new_samples) < num_new_samples and attempts < max_attempts:
        lhs_samples = lhs.random(batch_size)
        scaled_samples = scale(lhs_samples, l_bounds=opt_variable_lower, u_bounds=opt_variable_upper)
        
        for scaled_sample in scaled_samples:
            if len(new_samples) >= num_new_samples:
                break
            if not check_overlap(scaled_sample, data) and not check_overlap(scaled_sample, new_samples):
                new_samples.append(scaled_sample)

        attempts += 1

    if len(new_samples) < num_new_samples:
        print(f"Only {len(new_samples)} out of {num_new_samples} samples were generated without overlap.")
    new_samples_with_headers = np.vstack([headers, new_samples])
    return new_samples_with_headers


def sequential_sampling(valid_samples, error_samples, num_new_samples, opt_variable_lower, opt_variable_upper):
    # Check available devices
    print("Available devices:")
    for device in tf.config.list_physical_devices():
        print(device)

    # Ensure TensorFlow is using the GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        except RuntimeError as e:
            print(e)
    
    # Separate header and data
    headers = valid_samples[0]
    valid_data = np.array(valid_samples[1:], dtype=np.float64)
    error_data = np.array(error_samples[1:], dtype=np.float64)
    print('Seq : Initial sample data processing completed.')

    # Normalize valid and error samples
    opt_variable_lower_tf = tf.constant(opt_variable_lower, dtype=tf.float64)
    opt_variable_upper_tf = tf.constant(opt_variable_upper, dtype=tf.float64)
    
    normalized_valid_samples = (valid_data - opt_variable_lower_tf) / (opt_variable_upper_tf - opt_variable_lower_tf)
    normalized_error_samples = (error_data - opt_variable_lower_tf) / (opt_variable_upper_tf - opt_variable_lower_tf)
    
    # Combine valid and error samples, marking valid as 0 and error as 1
    all_samples = tf.concat([normalized_valid_samples, normalized_error_samples], axis=0)
    all_labels = tf.concat([tf.zeros(len(normalized_valid_samples)), tf.ones(len(normalized_error_samples))], axis=0)
    
    # Define a simple neural network model as the surrogate model
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(all_samples.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train the surrogate model
    model.fit(all_samples, all_labels, epochs=100, verbose=0)
    print('Seq : Surrogate model training completed.')
    
    # Acquisition function: Predict using the surrogate model
    def acquisition_function(samples):
        return model.predict(samples)
    
    # Generate new samples using the acquisition function
    new_samples = []
    for _ in range(num_new_samples):
        # Randomly initialize new sample candidates
        random_points = tf.random.uniform(shape=(1000, len(opt_variable_lower)), minval=0, maxval=1, dtype=tf.float64)
        
        # Predict acquisition values
        acquisition_values = acquisition_function(random_points)
        
        # Select the sample with the best (lowest) acquisition value
        best_idx = tf.argmin(acquisition_values).numpy().item()  # Extract scalar value
        new_sample = random_points[best_idx]
        
        # Add the new sample to the list
        new_samples.append(new_sample)
        
        # Update the training data with the new sample and retrain the model
        all_samples = tf.concat([all_samples, [new_sample]], axis=0)
        all_labels = tf.concat([all_labels, [0]], axis=0)
        model.fit(all_samples, all_labels, epochs=10, verbose=0)
    
    new_samples = tf.stack(new_samples)
    
    # Scale new_samples to the specified bounds
    new_samples = opt_variable_lower_tf + new_samples * (opt_variable_upper_tf - opt_variable_lower_tf)
    new_samples = new_samples.numpy()
    
    # Add header to new samples
    new_samples_with_header = np.vstack([headers, new_samples])
    print('Seq : New samples generation completed.')
    
    return new_samples_with_header