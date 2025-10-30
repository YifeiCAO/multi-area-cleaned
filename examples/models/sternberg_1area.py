from __future__ import division
import numpy as np
from pycog import tasktools
import pdb

learning_rate = 5e-5
max_gradient_norm = 0.1

# =======================
# Task-related parameters
# =======================

# Sternberg task parameters
set_sizes = [2, 4, 6]  # Number of items to remember
n_items = 8  # Total number of possible items (e.g., digits 0-7)
nconditions = len(set_sizes) * 2  # set_size * match/non-match

# Catch probability
pcatch = 0.1

# =================
# Network structure
# =================
Nin = n_items + 1  # n_items for item identity + 1 for probe signal
N = 300  # number of recurrent units
Nout = 2  # match / non-match

tau = 50
tau_in = 50
dt = 10
rectify_inputs = False

train_brec = True
train_bout = False
baseline_in = 0

# noise
vin = 0.10**2
# Diagonal covariance matrix for input noise
var_in_diag = [vin] * Nin
var_in = np.diag(var_in_diag)
var_rec = 0.05**2

# L2 regularization
lambda2_in = 1
lambda2_rec = 1
lambda2_out = 1
lambda2_r = 0
lambda_Omega = 2

# hidden activation
hidden_activation = 'rectify'

# Timing parameters (in ms)
T_catch = 2000
fixation_duration = 200  # Initial fixation period
item_duration = 300  # Duration to present each item
delay_duration = [1000, 2000]  # Variable delay period
probe_duration = 500  # Duration of probe presentation
response_duration = 500  # Time allowed for response
post_delay_set = 300

# E/I - ei an N vectors of 1's and -1's, EXC indices of +1's, INH indices of -1's
ei, EXC, INH = tasktools.generate_ei(N)

Nexc = len(EXC)
Ninh = len(INH)

# Single area: all neurons in one area
EXC_SENSORY = EXC[:Nexc]
INH_SENSORY = INH[:Ninh]

# Input connectivity: all neurons receive input
Cin = np.zeros((N, Nin))
Cin[list(EXC_SENSORY) + list(INH_SENSORY), :] = 1

rng = np.random.RandomState(1000)
ff_prop = 0.1
fb_prop = 0.05

# Build recurrent connectivity matrix (fully connected within area)
Crec = np.zeros((N, N))

for i in EXC_SENSORY:
    Crec[i, list(EXC_SENSORY)] = 1
    Crec[i, i] = 0
    Crec[i, list(INH_SENSORY)] = np.sum(Crec[i, list(EXC_SENSORY)]) / len(INH_SENSORY)

for i in INH_SENSORY:
    Crec[i, list(EXC_SENSORY)] = 1
    Crec[i, list(INH_SENSORY)] = np.sum(Crec[i, list(EXC_SENSORY)]) / (len(INH_SENSORY) - 1)
    Crec[i, i] = 0

Crec /= np.linalg.norm(Crec, axis=1)[:, np.newaxis]

# Output connectivity: read from all excitatory neurons
Cout = np.zeros((Nout, N))
Cout[:, list(EXC_SENSORY)] = 1


def generate_trial(rng, dt, params):
    """
    Generate a Sternberg working memory trial.
    
    Trial structure:
    1. Fixation period
    2. Sequential presentation of items (encoding phase)
    3. Delay period (working memory maintenance)
    4. Probe presentation
    5. Response period
    """
    
    catch_trial = False
    
    if params['name'] in ['gradient', 'test']:
        if params.get('catch', rng.rand() < pcatch):
            catch_trial = True
        else:
            set_size = params.get('set_size', rng.choice(set_sizes))
            is_match = params.get('is_match', rng.choice([0, 1]))
    
    elif params['name'] == 'validation':
        b = params['minibatch_index'] % (nconditions + 1)
        if b == 0:
            catch_trial = True
        else:
            k0, k1 = tasktools.unravel_index(b - 1, (len(set_sizes), 2))
            set_size = set_sizes[k0]
            is_match = k1
    else:
        raise ValueError("Unknown trial type.")
    
    # ======
    # Epochs
    # ======
    
    if catch_trial:
        epochs = {
            'fixation': (0, T_catch),
            'T': T_catch
        }
        trial_info = {'catch': True, 'dt': dt, 'epochs': epochs}
    
    else:
        # Calculate timing
        fixation = fixation_duration
        encoding_start = fixation
        encoding_duration = set_size * item_duration
        encoding_end = encoding_start + encoding_duration
        
        delay_dur = int(rng.uniform(delay_duration[0], delay_duration[1]) // dt * dt)
        delay_end = encoding_end + delay_dur
        
        probe_end = delay_end + probe_duration
        response_end = probe_end + response_duration
        post_delay = post_delay_set
        
        T = response_end + post_delay
        T = int((T // dt) * dt)
        
        # Generate memory set (unique items)
        memory_set = rng.choice(n_items, size=set_size, replace=False)
        
        # Generate probe
        if is_match:
            probe_item = rng.choice(memory_set)
        else:
            non_memory_items = [i for i in range(n_items) if i not in memory_set]
            probe_item = rng.choice(non_memory_items)
        
        # Define epochs
        epochs = {
            'fixation': (0, fixation),
            'encoding': (encoding_start, encoding_end),
            'delay': (encoding_end, delay_end),
            'probe': (delay_end, probe_end),
            'response': (probe_end, response_end),
            'post_delay': (response_end, T),
            'T': T
        }
        
        # Store item presentation times
        item_epochs = {}
        for i, item in enumerate(memory_set):
            item_start = encoding_start + i * item_duration
            item_end = item_start + item_duration
            item_epochs[f'item_{i}'] = (item_start, item_end)
        
        trial_info = {
            'set_size': set_size,
            'memory_set': memory_set,
            'probe_item': probe_item,
            'is_match': is_match,
            'choice': is_match,  # 1 for match, 0 for non-match
            'delay_duration': delay_dur,
            'catch': False,
            'dt': dt,
            'epochs': epochs,
            'item_epochs': item_epochs
        }
    
    # ==========
    # Trial info
    # ==========
    
    t, e = tasktools.get_epochs_idx(dt, epochs)
    trial = {'t': t, 'epochs': epochs}
    trial['info'] = trial_info
    
    # ======
    # Inputs
    # ======
    
    X = np.zeros((len(t), Nin))
    
    if not catch_trial:
        # Present each item sequentially during encoding
        for i, item in enumerate(memory_set):
            item_start = encoding_start + i * item_duration
            item_end = item_start + item_duration
            item_idx = tasktools.get_idx(t, (item_start, item_end))
            X[item_idx, item] = 1.0
        
        # Present probe
        X[e['probe'], probe_item] = 1.0
        # Signal probe period with additional input channel
        X[e['probe'], -1] = 1.0
    
    trial['inputs'] = X
    
    # ======
    # Output
    # ======
    
    if params.get('target_output', False):
        Y = np.zeros((len(t), Nout))
        M = np.zeros_like(Y)
        
        if catch_trial:
            # During catch trials, maintain baseline
            Y[:] = 0.5
            M[:] = 1
        else:
            # Output during response period
            if is_match:
                Y[e['response'], 1] = 1.0  # Match output
                Y[e['response'], 0] = 0.0
            else:
                Y[e['response'], 0] = 1.0  # Non-match output
                Y[e['response'], 1] = 0.0
            
            # Mask: care about all periods
            M[e['fixation'] + e['encoding'] + e['delay'] + e['probe'] + e['response'] + e['post_delay'], :] = 1
        
        trial['outputs'] = Y
        trial['mask'] = M
    
    return trial


def performance_sternberg(trials, z):
    """
    Performance measure for Sternberg task.
    Accuracy is computed based on match/non-match classification.
    """
    post_delay = trials[0]['info']['dt'] * 5  # small buffer
    dt = trials[0]['info']['dt']
    
    ends = [len(trial['t']) - 1 for trial in trials]
    # Check output at end of response period (before post_delay)
    choices = [np.argmax(z[ends[i] - int(post_delay // dt) - 1, i]) for i, end in enumerate(ends)]
    
    correct = []
    for choice, trial in zip(choices, trials):
        if trial['info']['catch']:
            continue
        # choice 1 = match, choice 0 = non-match
        correct.append(choice == trial['info']['is_match'])
    
    if len(correct) == 0:
        return 0.0
    
    return 100 * sum(correct) / len(correct)


# Performance measure
performance = performance_sternberg

# Termination criterion
TARGET_PERFORMANCE = 75


def terminate(pcorrect_history):
    return np.mean(pcorrect_history[-1:]) > TARGET_PERFORMANCE


# Validation dataset
n_validation = 100 * (nconditions + 1)


