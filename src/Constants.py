class Constants:
    ALGORITHMS = {'PPO', 'SAC'}
    COMMON_PARAMS = {
        'env',
        'gamma',
        'fcnet_activation',
        'fcnet_hiddens',
        'train_batch_size',
        'evaluation_interval'
    }

    MODEL_PARAMS = {
        'fcnet_activation',
        'fcnet_hiddens',
        'policy_layers',
        'q_value_layers',
        'policy_layers',
    }

    PPO_SPECIFIC_PARAMS = {
        'clip_param',
        'lambda',
        'num_sgd_iter',
        'sgd_minibatch_size',
        'vf_clip_param',
        'kl_target',
        'clip_param',
        'lr'
    }

    SAC_SPECIFIC_PARAMS = {
        'actor_learning_rate',
        'critic_learning_rate',
        'entropy_learning_rate',
        'policy_layers',
        'q_value_layers',
        'initial_alpha',
        'buffer_size',
        'tau',
        'learning_starts',
    }

    OPTIMIZATION_PARAMS = {
        'actor_learning_rate',
        'critic_learning_rate',
        'entropy_learning_rate'
    }

    # Env variables used in evaluation results data extraction
    ENV_PROGRESS_FILE_PATH = 'PROGRESS_FILE_PATH'
    ENV_EVALUATION_RESULTS_DIR_PATH = 'EVALUATION_RESULTS_FILE_PATH'
    LOGS_DIRECTORY = '/tensorboard_logs'
    LOGGER_NAME = 'rllib_runner_logger'

    # Indexes for values in Humanoid-v2 state vector
    HEIGHT_INDEX = 0
    TILT_INDEX = 3  # In qpos its under index 5, but observation cuts first two elements
    X_AXIS_ROTATION_INDEX = 4  # In qpos its under index 6, but observation cuts first two elements

    HEIGHT_NOMINAL_VALUE = 1.4
    TILT_NOMINAL_VALUE = 0.1
    X_AXIS_ROTATION_NOMINAL_VALUE = 0.0

