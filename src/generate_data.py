"""generate training and prime image datasets for orthographic dnn experiments."""

from utils.data_generate.orchestrate import init_create_prime_data, init_create_train_data

if __name__ == "__main__":
    init_create_train_data(dummy=False, random=False)
    init_create_prime_data(position_correction=False)
